from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from hashlib import sha256
import threading
import time
from typing import Any

from pydantic import ValidationError

from finagent.packages.common.config import Settings, get_settings
from finagent.packages.common.errors import ToolTimeoutError
from finagent.packages.common.sqlite_store import dumps
from finagent.packages.common.store import Store, build_store
from finagent.packages.market_data.adapters import MarketDataProvider, build_market_data_provider
from finagent.packages.orchestrator.models import SourceRef, ToolCall, ToolResult
from finagent.packages.quant_engine.metrics import calc_portfolio_metrics, run_scenario_analysis
from finagent.packages.rag.gateway import ResearchRetriever, build_research_retriever
from finagent.packages.risk.rules import RiskService
from finagent.packages.tools.registry import ToolContext, ToolRegistry, ToolSpec
from finagent.packages.tools.schemas import (
    CNAnnouncementsInput,
    ComplianceCheckInput,
    ForexRateInput,
    HistoricalPricesInput,
    ImpactAnalysisInput,
    MacroIndicatorInput,
    PositionValueInput,
    PortfolioMetricsInput,
    ResearchSearchInput,
    ScenarioAnalysisInput,
    SecFilingsInput,
    TickerMarketInput,
    UserPortfolioInput,
)


def _hash_args(arguments: dict[str, Any]) -> str:
    return sha256(dumps(arguments).encode("utf-8")).hexdigest()


class ToolExecutor:
    _progress: threading.local = threading.local()

    def __init__(self, registry: ToolRegistry, store: Store | None = None) -> None:
        self.registry = registry
        self.store = store

    @staticmethod
    def set_progress_callback(cb: Callable[[str, str], None] | None) -> None:
        ToolExecutor._progress.cb = cb

    @staticmethod
    def _notify_progress(tool_name: str, phase: str) -> None:
        cb: Callable[[str, str], None] | None = getattr(ToolExecutor._progress, "cb", None)
        if cb:
            try:
                cb(tool_name, phase)
            except Exception:
                pass

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        started = time.perf_counter()
        status = "error"
        source_ts = None
        try:
            spec = self.registry.get(call.tool_name)
            try:
                parsed = spec.input_model.model_validate(call.arguments)
            except ValidationError as exc:
                return self._finish(
                    call,
                    context,
                    started,
                    status="error",
                    error=f"工具参数校验失败：{exc.errors()}",
                )

            try:
                ToolExecutor._notify_progress(call.tool_name, "start")
                payload = self._run_handler_with_policy(spec, parsed, context)
            except Exception as exc:
                ToolExecutor._notify_progress(call.tool_name, "error")
                fallback_payload, fallback_error = self._run_fallback(spec, call.arguments, context, exc)
                if fallback_payload is None:
                    if fallback_error is not None:
                        return self._finish(
                            call,
                            context,
                            started,
                            status=status,
                            error=f"{exc}；降级工具 {spec.fallback_tool} 也执行失败：{fallback_error}",
                            source_ts=source_ts,
                        )
                    raise
                payload = fallback_payload
            ToolExecutor._notify_progress(call.tool_name, "end")
            return self._finish_payload(call, context, started, payload)
        except Exception as exc:
            return self._finish(call, context, started, status=status, error=str(exc), source_ts=source_ts)

    def _finish_payload(
        self,
        call: ToolCall,
        context: ToolContext,
        started: float,
        payload: dict[str, Any],
    ) -> ToolResult:
        payload = dict(payload)
        sources = [
            item if isinstance(item, SourceRef) else SourceRef.model_validate(item)
            for item in payload.pop("sources", [])
        ]
        source_ts = payload.pop("source_ts", None)
        status = payload.pop("status", "success")
        if status not in {"success", "error"}:
            return self._finish(
                call,
                context,
                started,
                status="error",
                error=f"工具返回了无效状态：{status}",
                source_ts=source_ts,
            )
        return self._finish(
            call,
            context,
            started,
            status=status,
            data=payload,
            sources=sources,
            source_ts=source_ts,
        )

    def _run_fallback(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolContext,
        original_error: Exception,
    ) -> tuple[dict[str, Any] | None, Exception | None]:
        if not spec.fallback_tool or not _is_timeout_error(original_error):
            return None, None
        try:
            fallback_spec = self.registry.get(spec.fallback_tool)
            parsed = fallback_spec.input_model.model_validate(arguments)
            payload = self._run_handler_with_policy(fallback_spec, parsed, context)
        except Exception as exc:
            return None, exc
        payload = dict(payload)
        payload["fallback"] = {
            "from_tool": spec.name,
            "fallback_tool": fallback_spec.name,
            "reason": str(original_error),
        }
        return payload, None

    def _run_handler_with_policy(self, spec: ToolSpec, parsed: Any, context: ToolContext) -> dict[str, Any]:
        attempts = max(1, spec.max_retries + 1)
        last_error: Exception | None = None
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            for attempt in range(1, attempts + 1):
                future = executor.submit(spec.handler, parsed, context)
                try:
                    payload = future.result(timeout=spec.timeout_seconds)
                    return payload
                except FutureTimeoutError:
                    future.cancel()
                    last_error = TimeoutError(f"{spec.name} 在 {spec.timeout_seconds:g} 秒内未返回结果")
                except Exception as exc:
                    last_error = exc
                if attempt < attempts:
                    continue
            assert last_error is not None
            raise last_error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _finish(
        self,
        call: ToolCall,
        context: ToolContext,
        started: float,
        *,
        status: str,
        data: dict[str, Any] | None = None,
        sources: list[SourceRef] | None = None,
        source_ts: Any = None,
        error: str | None = None,
    ) -> ToolResult:
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            status=status,  # type: ignore[arg-type]
            data=data or {},
            sources=sources or [],
            source_ts=source_ts,
            latency_ms=latency_ms,
            error=error,
        )
        self._log(call, context, result, started)
        return result

    def _log(self, call: ToolCall, context: ToolContext, result: ToolResult, started: float) -> None:
        if not self.store:
            return
        self.store.log_tool_call(
            context.tenant_id,
            context.task_id,
            call.tool_call_id,
            call.tool_name,
            _hash_args(call.arguments),
            result.status,
            result.latency_ms,
            result.source_ts.isoformat() if result.source_ts else None,
        )


def build_default_registry(
    settings: Settings | None = None,
    store: Store | None = None,
    market_data_provider: MarketDataProvider | None = None,
    research_retriever: ResearchRetriever | None = None,
) -> ToolRegistry:
    settings = settings or get_settings()
    if store is None:
        store = build_store(settings)
    market_data = market_data_provider or build_market_data_provider(settings, store)
    retriever = research_retriever or build_research_retriever(settings)
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_realtime_stock",
            description="Get realtime stock quote.",
            input_model=TickerMarketInput,
            handler=market_data.get_realtime_stock,
            fallback_tool="get_delayed_quote",
        )
    )
    registry.register(
        ToolSpec(
            name="get_delayed_quote",
            description="Get delayed quote from the latest historical close.",
            input_model=TickerMarketInput,
            handler=market_data.get_delayed_quote,
        )
    )
    registry.register(
        ToolSpec(
            name="get_historical_prices",
            description="Get historical prices for deterministic quant calculations.",
            input_model=HistoricalPricesInput,
            handler=market_data.get_historical_prices,
        )
    )
    registry.register(
        ToolSpec(
            name="macro_indicator",
            description="Get macro indicator snapshot.",
            input_model=MacroIndicatorInput,
            handler=market_data.get_macro_indicator,
        )
    )
    registry.register(
        ToolSpec(
            name="get_forex_rate",
            description="Get a foreign exchange reference rate through the macro gateway.",
            input_model=ForexRateInput,
            handler=lambda payload, context: _get_forex_rate(market_data, payload, context),
        )
    )
    registry.register(
        ToolSpec(
            name="fetch_sec_filings",
            description="Fetch SEC filing sections.",
            input_model=SecFilingsInput,
            handler=market_data.fetch_sec_filings,
        )
    )
    registry.register(
        ToolSpec(
            name="fetch_cn_announcements",
            description="Fetch China-market announcements.",
            input_model=CNAnnouncementsInput,
            handler=market_data.fetch_cn_announcements,
        )
    )
    registry.register(
        ToolSpec(
            name="web_search",
            description="Search research reports and news with paragraph-level snippets.",
            input_model=ResearchSearchInput,
            handler=retriever.web_search,
        )
    )
    registry.register(
        ToolSpec(
            name="get_user_portfolio",
            description="Read tenant/user scoped portfolio snapshot.",
            input_model=UserPortfolioInput,
            handler=market_data.get_user_portfolio,
        )
    )
    registry.register(
        ToolSpec(
            name="calc_portfolio_metrics",
            description="Calculate deterministic portfolio metrics.",
            input_model=PortfolioMetricsInput,
            handler=calc_portfolio_metrics,
        )
    )
    registry.register(
        ToolSpec(
            name="calc_position_value",
            description="Calculate deterministic position market value.",
            input_model=PositionValueInput,
            handler=_calc_position_value,
        )
    )
    registry.register(
        ToolSpec(
            name="run_scenario_analysis",
            description="Run deterministic scenario stress analysis.",
            input_model=ScenarioAnalysisInput,
            handler=run_scenario_analysis,
        )
    )
    registry.register(
        ToolSpec(
            name="analyze_impact",
            description="Document-compatible qualitative macro impact analyzer.",
            input_model=ImpactAnalysisInput,
            handler=_analyze_impact,
        )
    )
    risk_service = RiskService()
    registry.register(
        ToolSpec(
            name="compliance_rule_check",
            description="Run rule-based compliance checks.",
            input_model=ComplianceCheckInput,
            handler=lambda payload, context: risk_service.tool_check(payload, context),
        )
    )
    return registry


def _is_timeout_error(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ToolTimeoutError))


def _calc_position_value(payload: PositionValueInput, context: ToolContext) -> dict[str, Any]:
    return {
        "market_value": round(payload.quantity * payload.price, 2),
        "quantity": payload.quantity,
        "price": payload.price,
        "currency": payload.currency,
        "sources": [
            SourceRef(
                source_type="calculation",
                title="Position value calculation engine",
                freshness="snapshot",
            )
        ],
    }


def _get_forex_rate(
    market_data: MarketDataProvider,
    payload: ForexRateInput,
    context: ToolContext,
) -> dict[str, Any]:
    macro = market_data.get_macro_indicator(
        MacroIndicatorInput(
            indicator=f"{payload.base.lower()}_{payload.quote.lower()}_fx",
            region="GLOBAL",
            latest=True,
        ),
        context,
    )
    normalized = dict(macro)
    rate = normalized.get("rate", normalized.get("value", normalized.get("price")))
    normalized["base"] = payload.base
    normalized["quote"] = payload.quote
    normalized["pair"] = f"{payload.base}/{payload.quote}"
    if rate is not None:
        normalized["rate"] = rate
    return normalized


def _analyze_impact(payload: ImpactAnalysisInput, context: ToolContext) -> dict[str, Any]:
    sector = payload.sector or "broad_market"
    lower_event = payload.macro_event.lower()
    if "rate_hike" in lower_event or "hike" in lower_event or "fomc" in lower_event:
        sector_impact = "Rate hikes may support bank net interest margins while raising credit and valuation risks."
    elif "rate_cut" in lower_event or "cut" in lower_event:
        sector_impact = "Rate cuts may reduce funding pressure but can signal weaker growth expectations."
    else:
        sector_impact = "Impact should be assessed through earnings sensitivity, valuation, liquidity, and risk appetite."
    stock_specific = [
        {
            "ticker": ticker,
            "sensitivity": "high" if sector.lower() in {"financials", "banks"} else "medium",
            "impact_path": sector_impact,
        }
        for ticker in payload.holdings
    ]
    return {
        "macro_event": payload.macro_event,
        "sector": sector,
        "sector_impact": sector_impact,
        "holdings": payload.holdings,
        "stock_specific": stock_specific,
        "sources": [
            SourceRef(
                source_type="calculation",
                title="Macro impact analysis policy",
                freshness="snapshot",
            )
        ],
    }
