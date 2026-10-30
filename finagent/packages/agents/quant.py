from __future__ import annotations

from finagent.packages.agents.base import execute_tool
from finagent.packages.orchestrator.models import AgentOutput, AgentState
from finagent.packages.tools.executor import ToolExecutor
from finagent.packages.tools.registry import ToolContext


class QuantAgent:
    name = "quant_agent"

    def run(
        self,
        state: AgentState,
        executor: ToolExecutor,
        context: ToolContext,
        *,
        scenario: str | None = None,
        user_input: str = "",
    ) -> AgentOutput:
        sector = self._sector_from_user_input(user_input)
        scenario = scenario or self._scenario_from_context(state, user_input, sector)
        portfolio = execute_tool(
            state=state,
            executor=executor,
            context=context,
            tool_name="get_user_portfolio",
            arguments={"account_scope": context.account_scope, "sector": sector},
        )
        if portfolio.status == "error":
            output = AgentOutput(
                agent_name=self.name,
                summary="Portfolio holdings could not be loaded.",
                findings=[portfolio.error or "Portfolio tool returned an error."],
                tool_call_ids=[portfolio.tool_call_id],
                sources=portfolio.sources,
                confidence=0.1,
                metadata={"portfolio_error": portfolio.model_dump(mode="json")},
            )
            state.agent_outputs[self.name] = output
            return output

        holdings = portfolio.data.get("holdings", [])
        if not holdings:
            scope = f" in sector {sector}" if sector else ""
            output = AgentOutput(
                agent_name=self.name,
                summary="Portfolio snapshot returned no holdings for the requested scope.",
                findings=[f"No portfolio holdings were available{scope}; deterministic metrics were not calculated."],
                tool_call_ids=[portfolio.tool_call_id],
                sources=portfolio.sources,
                confidence=0.35,
                metadata={
                    "portfolio": portfolio.data,
                    "analysis_context": {
                        "sector": sector,
                        "scenario": scenario,
                    },
                },
            )
            state.agent_outputs[self.name] = output
            return output

        positions = [self._position_for_metrics(item) for item in holdings]
        tool_ids = [portfolio.tool_call_id]
        sources = [*portfolio.sources]
        price_history_results = []
        for position in positions:
            history = execute_tool(
                state=state,
                executor=executor,
                context=context,
                tool_name="get_historical_prices",
                arguments={
                    "ticker": position["ticker"],
                    "market": position.get("market", "US"),
                    "period": "6mo",
                    "frequency": "daily",
                    "adjusted": True,
                },
            )
            tool_ids.append(history.tool_call_id)
            sources.extend(history.sources)
            price_history_results.append(history.model_dump(mode="json"))
            if history.status == "success":
                position["price_history"] = self._closing_prices(history.data)
        metrics = execute_tool(
            state=state,
            executor=executor,
            context=context,
            tool_name="calc_portfolio_metrics",
            arguments={"positions": positions, "risk_free_rate": 0.02},
        )
        tool_ids.append(metrics.tool_call_id)
        sources.extend(metrics.sources)
        findings = [
            f"Portfolio snapshot value: {portfolio.data.get('total_value', 0):,.2f} {portfolio.data.get('currency', 'USD')}.",
        ]
        if sector:
            findings.append(f"Sector scope: {sector}.")
        if metrics.status == "error":
            findings.append(metrics.error or "Portfolio metrics tool returned an error.")
        else:
            findings.extend(
                [
                    f"Max drawdown estimate: {metrics.data.get('max_drawdown', 0):.2%}.",
                    f"VaR 95 estimate: {metrics.data.get('var_95', 0):,.2f}.",
                ]
            )
        metadata = {
            "portfolio": portfolio.data,
            "price_history": price_history_results,
            "metrics": metrics.data,
            "analysis_context": {
                "sector": sector,
                "scenario": scenario,
            },
        }
        if scenario:
            stress = execute_tool(
                state=state,
                executor=executor,
                context=context,
                tool_name="run_scenario_analysis",
                arguments={"positions": positions, "scenario": scenario, "shock_bps": 25},
            )
            tool_ids.append(stress.tool_call_id)
            sources.extend(stress.sources)
            if stress.status == "error":
                findings.append(stress.error or "Scenario analysis returned an error.")
            else:
                findings.append(f"Scenario estimated P/L: {stress.data.get('estimated_total_pnl', 0):,.2f}.")
            metadata["scenario"] = stress.data
            impact = execute_tool(
                state=state,
                executor=executor,
                context=context,
                tool_name="analyze_impact",
                arguments={
                    "macro_event": scenario,
                    "sector": sector,
                    "holdings": [position["ticker"] for position in positions],
                },
            )
            tool_ids.append(impact.tool_call_id)
            sources.extend(impact.sources)
            if impact.status == "error":
                findings.append(impact.error or "Macro impact analysis returned an error.")
            else:
                findings.append(impact.data.get("sector_impact", "Macro impact analysis completed."))
            metadata["impact_analysis"] = impact.data
        output = AgentOutput(
            agent_name=self.name,
            summary="Deterministic portfolio metrics and scenario calculations completed.",
            findings=findings,
            tool_call_ids=tool_ids,
            sources=sources,
            confidence=0.82,
            metadata=metadata,
        )
        state.agent_outputs[self.name] = output
        return output

    @staticmethod
    def _position_for_metrics(item: dict) -> dict:
        return {
            "ticker": item["ticker"],
            "market": item.get("market", "US"),
            "quantity": item["quantity"],
            "price": item["price"],
            "cost_price": item["cost_price"],
            "sector": item.get("sector"),
        }

    @staticmethod
    def _closing_prices(data: dict) -> list[float]:
        prices = data.get("prices") or []
        if not isinstance(prices, list):
            return []
        closes: list[float] = []
        for item in prices:
            if isinstance(item, dict):
                value = item.get("adjusted_close") or item.get("close")
            else:
                value = item
            if isinstance(value, int | float):
                closes.append(float(value))
        return closes

    @staticmethod
    def _sector_from_user_input(text: str) -> str | None:
        lowered = text.lower()
        sector_patterns = (
            ("Financials", ("bank", "banks", "financial", "\u94f6\u884c", "\u91d1\u878d", "\u5238\u5546", "\u4fdd\u9669")),
            ("Technology", ("tech", "technology", "semiconductor", "chip", "\u79d1\u6280", "\u82af\u7247", "\u534a\u5bfc\u4f53")),
            ("Energy", ("energy", "oil", "gas", "\u80fd\u6e90", "\u539f\u6cb9", "\u6cb9\u6c14")),
            ("Consumer", ("consumer", "retail", "\u6d88\u8d39", "\u96f6\u552e")),
            ("Healthcare", ("healthcare", "health", "pharma", "\u533b\u836f", "\u533b\u7597")),
            ("Materials", ("materials", "metals", "\u6750\u6599", "\u6709\u8272", "\u91d1\u5c5e")),
            ("Utilities", ("utilities", "utility", "\u516c\u7528\u4e8b\u4e1a", "\u7535\u529b")),
        )
        for sector, tokens in sector_patterns:
            if any(token in lowered or token in text for token in tokens):
                return sector
        return None

    @staticmethod
    def _scenario_from_context(state: AgentState, user_input: str, sector: str | None) -> str | None:
        macro_output = state.agent_outputs.get("macro_agent")
        if macro_output is None:
            return None
        macro_text = _macro_context_text(macro_output, user_input)
        if _has_any(macro_text, ("rate_cut", "cut", "\u964d\u606f")):
            return "rate_cut_25bps"
        if _has_any(macro_text, ("rate_hike", "hike", "fomc", "fed", "+25bps", "\u52a0\u606f", "\u7f8e\u8054\u50a8")):
            return "rate_hike_25bps"
        if _has_any(macro_text, ("usd", "dollar", "fx", "forex", "\u7f8e\u5143", "\u6c47\u7387")):
            return "usd_strength"
        if _has_any(macro_text, ("oil", "crude", "brent", "\u539f\u6cb9", "\u6cb9\u4ef7")):
            return "oil_shock"
        if sector is not None:
            return "sector_stress"
        return "sector_stress"


def _macro_context_text(macro_output: AgentOutput, user_input: str) -> str:
    metadata = macro_output.metadata
    request = metadata.get("macro_request") if isinstance(metadata, dict) else None
    macro_data = metadata.get("macro_data") if isinstance(metadata, dict) else None
    parts = [user_input, macro_output.summary, " ".join(macro_output.findings)]
    if isinstance(request, dict):
        parts.extend(str(value) for value in request.values())
    if isinstance(macro_data, dict):
        parts.extend(str(value) for value in macro_data.values())
    return " ".join(parts).lower()


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)
