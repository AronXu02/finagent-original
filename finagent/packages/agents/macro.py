from __future__ import annotations

from dataclasses import dataclass
import re

from finagent.packages.agents.base import execute_tool
from finagent.packages.orchestrator.models import AgentOutput, AgentState
from finagent.packages.tools.executor import ToolExecutor
from finagent.packages.tools.registry import ToolContext


@dataclass(frozen=True)
class MacroRequest:
    indicator: str
    region: str
    research_query: str
    theme: str


class MacroAgent:
    name = "macro_agent"

    def run(self, state: AgentState, executor: ToolExecutor, context: ToolContext, user_input: str = "") -> AgentOutput:
        request = self._request_from_input(user_input)
        macro = execute_tool(
            state=state,
            executor=executor,
            context=context,
            tool_name="macro_indicator",
            arguments={"indicator": request.indicator, "region": request.region, "latest": True},
        )
        research = execute_tool(
            state=state,
            executor=executor,
            context=context,
            tool_name="web_search",
            arguments={
                "query": request.research_query,
                "top_k": 3,
            },
        )
        findings = self._findings_for(request)
        if research.status == "error":
            findings.append(research.error or "Macro research retrieval failed.")
        else:
            findings.append("Macro research/news snippets were retrieved for cross-checking.")
        observed = macro.data.get("event") or macro.data.get("indicator") or request.indicator
        summary = f"Macro event assessed with an auditable macro data source: {observed}."
        output = AgentOutput(
            agent_name=self.name,
            summary=summary,
            findings=findings,
            tool_call_ids=[macro.tool_call_id, research.tool_call_id],
            sources=[*macro.sources, *research.sources],
            confidence=0.78,
            metadata={
                "macro_request": {
                    "indicator": request.indicator,
                    "region": request.region,
                    "theme": request.theme,
                    "research_query": request.research_query,
                },
                "macro_data": macro.data,
                "research": research.data,
            },
        )
        state.agent_outputs[self.name] = output
        return output

    @staticmethod
    def _request_from_input(user_input: str) -> MacroRequest:
        text = user_input.strip()
        lowered = text.lower()
        region = _region_from_text(text)
        if _contains_any(lowered, text, ("rate cut", "cut rates", "\u964d\u606f")):
            indicator = "central_bank_rate_cut"
            theme = "rate_cut"
        elif _contains_any(lowered, text, ("rate hike", "hike rates", "fomc", "fed", "\u7f8e\u8054\u50a8", "\u52a0\u606f")):
            indicator = "fomc_rate_decision"
            theme = "rate_hike"
            region = "US" if region == "GLOBAL" else region
        elif _contains_any(lowered, text, ("cpi", "inflation", "\u901a\u80c0", "\u7269\u4ef7")):
            indicator = "cpi_yoy"
            theme = "inflation"
        elif _contains_any(lowered, text, ("pmi", "\u5236\u9020\u4e1a", "\u91c7\u8d2d\u7ecf\u7406")):
            indicator = "pmi"
            theme = "growth"
        elif _contains_any(lowered, text, ("gdp", "\u7ecf\u6d4e\u589e\u957f")):
            indicator = "gdp_growth"
            theme = "growth"
        elif _contains_any(lowered, text, ("nonfarm", "unemployment", "jobs", "\u975e\u519c", "\u5931\u4e1a", "\u5c31\u4e1a")):
            indicator = "labor_market"
            theme = "labor"
        elif _contains_any(lowered, text, ("oil", "crude", "brent", "\u539f\u6cb9", "\u6cb9\u4ef7")):
            indicator = "brent_crude_oil"
            theme = "commodity"
            region = "GLOBAL"
        elif _contains_any(lowered, text, ("usd", "jpy", "cny", "fx", "forex", "\u6c47\u7387", "\u7f8e\u5143", "\u65e5\u5143")):
            indicator = "usd_jpy_fx"
            theme = "fx"
            region = "GLOBAL"
        else:
            indicator = "macro_policy_snapshot"
            theme = "macro_policy"
        return MacroRequest(
            indicator=indicator,
            region=region,
            research_query=_research_query(text, indicator, theme),
            theme=theme,
        )

    @staticmethod
    def _findings_for(request: MacroRequest) -> list[str]:
        base = [f"Macro indicator requested: {request.indicator} ({request.region})."]
        if request.theme in {"rate_hike", "rate_cut"}:
            base.append("Rate-sensitive assets should be checked through discount rates, funding costs, and credit quality.")
        elif request.theme == "inflation":
            base.append("Inflation shocks can affect margins, real yields, valuation multiples, and central-bank expectations.")
        elif request.theme == "growth":
            base.append("Growth indicators should be mapped to earnings sensitivity, demand durability, and cyclical exposure.")
        elif request.theme == "labor":
            base.append("Labor-market changes can alter growth expectations, wage pressure, and policy reaction functions.")
        elif request.theme == "commodity":
            base.append("Commodity shocks should be mapped to input costs, energy exposure, inflation expectations, and margins.")
        elif request.theme == "fx":
            base.append("FX moves should be mapped to translation effects, import costs, exporters, and foreign-currency liabilities.")
        else:
            base.append("Macro transmission should be assessed through earnings, liquidity, valuation, and risk appetite.")
        return base


def _contains_any(lowered: str, original: str, tokens: tuple[str, ...]) -> bool:
    return any(token in lowered or token in original for token in tokens)


def _region_from_text(text: str) -> str:
    lowered = text.lower()
    if _contains_any(lowered, text, ("china", "pboc", "cny", "\u4e2d\u56fd", "a\u80a1", "\u4eba\u6c11\u5e01")):
        return "CN"
    if _contains_any(lowered, text, ("hong kong", "hkd", "\u6e2f\u80a1", "\u9999\u6e2f")):
        return "HK"
    if _contains_any(lowered, text, ("us", "usa", "fed", "fomc", "usd", "\u7f8e\u56fd", "\u7f8e\u8054\u50a8", "\u7f8e\u5143")):
        return "US"
    return "GLOBAL"


def _research_query(user_input: str, indicator: str, theme: str) -> str:
    compact = re.sub(r"\s+", " ", user_input).strip()
    if compact:
        return f"{compact[:160]} macro transmission {theme} earnings valuation portfolio impact"
    return f"{indicator} macro transmission earnings valuation portfolio impact"
