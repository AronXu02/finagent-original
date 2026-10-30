from __future__ import annotations

import re

from finagent.packages.agents.asset_resolver import resolve_asset_alias
from finagent.packages.agents.base import execute_tool
from finagent.packages.orchestrator.models import AgentOutput, AgentState
from finagent.packages.tools.executor import ToolExecutor
from finagent.packages.tools.registry import ToolContext


US_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5})(?![A-Za-z0-9])")
CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,5})(?![A-Za-z0-9])")
LISTED_NUMBER_RE = re.compile(r"(?<!\d)(\d{5,6})(?:\.(SH|SZ|HK))?(?!\d)")
QUANTITY_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:shares?|股)", re.IGNORECASE)
FOREX_PAIR_RE = re.compile(r"(?<![A-Za-z])([A-Z]{3})\s*/?\s*([A-Z]{3})(?![A-Za-z])")
TICKER_STOPWORDS = {
    "ASK",
    "CN",
    "ETF",
    "FX",
    "HK",
    "JPY",
    "NYSE",
    "PRICE",
    "QUOTE",
    "RATE",
    "SSE",
    "STOCK",
    "USD",
    "US",
}


class MarketAgent:
    name = "market_agent"

    def run(self, state: AgentState, executor: ToolExecutor, context: ToolContext, user_input: str) -> AgentOutput:
        lowered = user_input.lower()
        if "jpy" in lowered or "\u65e5\u5143" in user_input or "\u6c47\u7387" in user_input:
            base, quote = self._extract_forex_pair(user_input) or ("USD", "JPY")
            result = execute_tool(
                state=state,
                executor=executor,
                context=context,
                tool_name="get_forex_rate",
                arguments={"base": base, "quote": quote},
            )
            finding = f"{base}/{quote} reference rate: {result.data.get('rate')}."
            tool_call_ids = [result.tool_call_id]
            sources = result.sources
            metadata = {"quote": result.data}
        else:
            target = self._extract_quote_target(user_input)
            if target is None:
                output = AgentOutput(
                    agent_name=self.name,
                    summary="行情查询需要明确的证券代码。",
                    findings=[
                        "请先提供 ticker 或上市证券代码，再调用行情工具。"
                    ],
                    confidence=0.25,
                    metadata={"input": user_input, "missing": "ticker"},
                )
                state.agent_outputs[self.name] = output
                return output
            ticker, market = target
            result = execute_tool(
                state=state,
                executor=executor,
                context=context,
                tool_name="get_realtime_stock",
                arguments={"ticker": ticker, "market": market},
            )
            finding = f"{ticker} reference price: {result.data.get('price')} {result.data.get('currency', 'USD')}."
            tool_call_ids = [result.tool_call_id]
            sources = [*result.sources]
            metadata = {"quote": result.data}
            quantity = self._extract_quantity(user_input)
            price = result.data.get("price")
            if quantity is not None and isinstance(price, int | float):
                value = execute_tool(
                    state=state,
                    executor=executor,
                    context=context,
                    tool_name="calc_position_value",
                    arguments={
                        "quantity": quantity,
                        "price": float(price),
                        "currency": result.data.get("currency", "USD"),
                    },
                )
                tool_call_ids.append(value.tool_call_id)
                sources.extend(value.sources)
                if value.status == "error":
                    finding = f"{finding} Position value calculation failed: {value.error}."
                else:
                    finding = f"{finding} Position market value: {value.data.get('market_value'):,.2f} {value.data.get('currency', 'USD')}."
                metadata["position_value"] = value.data
        output = AgentOutput(
            agent_name=self.name,
            summary="Market snapshot retrieved.",
            findings=[finding],
            tool_call_ids=tool_call_ids,
            sources=sources,
            confidence=0.8,
            metadata=metadata,
        )
        state.agent_outputs[self.name] = output
        return output

    @staticmethod
    def _extract_quote_target(text: str) -> tuple[str, str] | None:
        upper = text.upper()
        listed = LISTED_NUMBER_RE.search(upper)
        if listed:
            ticker = listed.group(0)
            suffix = listed.group(2)
            if suffix == "HK" or (suffix is None and len(listed.group(1)) == 5):
                return ticker, "HK"
            return ticker, "CN"
        cashtag = CASHTAG_RE.search(text)
        if cashtag:
            token = cashtag.group(1).upper()
            if token not in TICKER_STOPWORDS:
                return token, "US"
        alias = resolve_asset_alias(text, markets=("US", "CN", "HK"))
        if alias is not None:
            return alias
        for match in US_TICKER_RE.finditer(text):
            token = match.group(1).upper()
            if token not in TICKER_STOPWORDS:
                return token, "US"
        return None

    @staticmethod
    def _extract_quantity(text: str) -> float | None:
        match = QUANTITY_RE.search(text)
        if not match:
            return None
        return float(match.group(1))

    @staticmethod
    def _extract_forex_pair(text: str) -> tuple[str, str] | None:
        upper = text.upper()
        for match in FOREX_PAIR_RE.finditer(upper):
            base, quote = match.group(1), match.group(2)
            if base != quote and {base, quote} <= {"USD", "JPY", "EUR", "GBP", "CNH", "CNY", "HKD"}:
                return base, quote
        return None
