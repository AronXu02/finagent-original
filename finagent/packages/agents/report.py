from __future__ import annotations

import re

from finagent.packages.agents.asset_resolver import resolve_asset_alias
from finagent.packages.agents.base import execute_tool
from finagent.packages.orchestrator.models import AgentOutput, AgentState
from finagent.packages.tools.executor import ToolExecutor
from finagent.packages.tools.registry import ToolContext


TICKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5})(?![A-Za-z0-9])")
LISTED_NUMBER_RE = re.compile(r"(?<!\d)(\d{5,6})(?:\.(SH|SZ))?(?!\d)")
TICKER_STOPWORDS = {
    "FACTORS",
    "FILING",
    "FORM",
    "ITEM",
    "K",
    "Q",
    "RISK",
    "SEC",
    "US",
}


class ReportAgent:
    name = "report_agent"

    def run(self, state: AgentState, executor: ToolExecutor, context: ToolContext, user_input: str) -> AgentOutput:
        cn_ticker = self._extract_cn_ticker(user_input)
        if cn_ticker is None:
            cn_alias = resolve_asset_alias(user_input, markets=("CN",))
            if cn_alias is not None:
                cn_ticker = cn_alias[0]
        if cn_ticker is not None:
            announcement = execute_tool(
                state=state,
                executor=executor,
                context=context,
                tool_name="fetch_cn_announcements",
                arguments={"ticker": cn_ticker, "doc_type": self._cn_doc_type(user_input)},
            )
            if announcement.status == "error":
                output = AgentOutput(
                    agent_name=self.name,
                    summary=f"{cn_ticker} 公告检索失败。",
                    findings=[announcement.error or "A 股公告工具返回错误。"],
                    tool_call_ids=[announcement.tool_call_id],
                    sources=announcement.sources,
                    confidence=0.1,
                    metadata={"announcement_error": announcement.model_dump(mode="json")},
                )
                state.agent_outputs[self.name] = output
                return output
            findings = announcement.data.get("highlights") or [
                f"已定位公告：{announcement.data.get('ticker', cn_ticker)}。",
                "形成结论前应核对公告发布时间、来源和适用市场。",
            ]
            output = AgentOutput(
                agent_name=self.name,
                summary=f"{cn_ticker} announcement parsed from a filing source.",
                findings=findings,
                tool_call_ids=[announcement.tool_call_id],
                sources=announcement.sources,
                confidence=0.72,
                metadata={"announcement": announcement.data},
            )
            state.agent_outputs[self.name] = output
            return output

        ticker = self._extract_ticker(user_input)
        if ticker is None:
            output = AgentOutput(
                agent_name=self.name,
                summary="研报/财报分析需要明确的美股 ticker 或 A 股证券代码。",
                findings=[
                    "请先提供美股 ticker 或 A 股证券代码，再调用财报/公告检索工具。"
                ],
                confidence=0.25,
                metadata={"input": user_input, "missing": "ticker"},
            )
            state.agent_outputs[self.name] = output
            return output
        filing = execute_tool(
            state=state,
            executor=executor,
            context=context,
            tool_name="fetch_sec_filings",
            arguments={"ticker": ticker, "form_type": "10-K", "section": "risk_factors"},
        )
        if filing.status == "error":
            output = AgentOutput(
                agent_name=self.name,
                summary=f"{ticker} 财报检索失败。",
                findings=[filing.error or "SEC 财报工具返回错误。"],
                tool_call_ids=[filing.tool_call_id],
                sources=filing.sources,
                confidence=0.1,
                metadata={"filing_error": filing.model_dump(mode="json")},
            )
            state.agent_outputs[self.name] = output
            return output
        findings = filing.data.get("highlights") or [
            f"已定位财报元数据：{filing.data.get('filing_url', '来源 URL 不可用')}。",
            "在形成文本结论前，应先由已配置的财报解析器完成章节级解析。",
        ]
        output = AgentOutput(
            agent_name=self.name,
            summary=f"{ticker} filing risk factors parsed from a filing source.",
            findings=findings,
            tool_call_ids=[filing.tool_call_id],
            sources=filing.sources,
            confidence=0.76,
            metadata={"filing": filing.data},
        )
        state.agent_outputs[self.name] = output
        return output

    @staticmethod
    def _extract_ticker(text: str) -> str | None:
        matches = TICKER_RE.finditer(text)
        for match in matches:
            token = match.group(1).upper()
            if token not in TICKER_STOPWORDS:
                return token
        alias = resolve_asset_alias(text, markets=("US",))
        if alias is not None:
            return alias[0]
        return None

    @staticmethod
    def _extract_cn_ticker(text: str) -> str | None:
        match = LISTED_NUMBER_RE.search(text.upper())
        if not match:
            alias = resolve_asset_alias(text, markets=("CN",))
            return alias[0] if alias is not None else None
        ticker = match.group(1)
        suffix = match.group(2)
        if suffix:
            return f"{ticker}.{suffix}"
        if ticker.startswith(("6", "9")):
            return f"{ticker}.SH"
        return f"{ticker}.SZ"

    @staticmethod
    def _cn_doc_type(text: str) -> str:
        lowered = text.lower()
        if "annual" in lowered or "年报" in text:
            return "annual_report"
        if "quarter" in lowered or "季报" in text:
            return "quarterly_report"
        if "问询" in text or "inquiry" in lowered:
            return "inquiry_letter"
        return "announcement"
