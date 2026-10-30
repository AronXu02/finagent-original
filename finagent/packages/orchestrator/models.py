from __future__ import annotations

import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from finagent.packages.common.ids import new_id
from finagent.packages.common.time import utc_now


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    created_at: datetime = Field(default_factory=utc_now)
    token_count: int = 0


class IntentType(StrEnum):
    MARKET_QUOTE = "market_quote"
    PORTFOLIO_RISK = "portfolio_risk"
    PORTFOLIO_IMPACT = "portfolio_impact"
    MACRO_EVENT = "macro_event"
    REPORT_ANALYSIS = "report_analysis"
    PREFERENCE_UPDATE = "preference_update"
    HIGH_RISK_REQUEST = "high_risk_request"
    GENERAL = "general_financial_question"
    CLARIFICATION = "clarification"


class RouteStep(StrictModel):
    node: str
    reason: str
    tools: list[str] = Field(default_factory=list)
    required: bool = True


class SourceRef(StrictModel):
    source_id: str = Field(default_factory=lambda: new_id("src"))
    source_type: str
    title: str
    url: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=utc_now)
    freshness: Literal["realtime", "delayed", "snapshot", "stale"] = "snapshot"
    checksum: str | None = None


class MemoryItemView(StrictModel):
    memory_id: str
    type: str
    content: str
    importance: float = 0.5
    decay_weight: float = 1.0
    source: str | None = None
    is_core: bool = False
    market_tag: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    valid_to: datetime | None = None


class MemoryContext(StrictModel):
    top_memories: list[MemoryItemView] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)


class TaskFrame(StrictModel):
    task_id: str
    intent: IntentType
    description: str
    checkpoint_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class InterruptionInfo(StrictModel):
    interrupted_task_id: str
    new_intent: IntentType
    reason: str
    created_at: datetime = Field(default_factory=utc_now)


def _repair_json_object(value: str) -> dict[str, Any]:
    candidates = [value.strip()]
    if candidates[0].startswith("```"):
        candidates.append(re.sub(r"^```(?:json)?\s*|\s*```$", "", candidates[0], flags=re.IGNORECASE | re.DOTALL).strip())
    candidates.append(re.sub(r",\s*([}\]])", r"\1", candidates[-1]))
    candidates.append(re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', candidates[-1]))
    candidates.append(re.sub(r"(?<!\\)'", '"', candidates[-1]))
    candidates.append(
        re.sub(
            r"\b(True|False|None)\b",
            lambda match: {"True": "true", "False": "false", "None": "null"}[match.group(1)],
            candidates[-1],
        )
    )
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            return {}
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments JSON must decode to an object")
        return parsed
    raise ValueError(f"tool arguments JSON repair failed: {last_error}")


class ToolCall(StrictModel):
    tool_call_id: str = Field(default_factory=lambda: new_id("toolcall"))
    tool_name: str
    arguments: dict[str, Any]
    idempotency_key: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("arguments", mode="before")
    @classmethod
    def repair_json_arguments(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _repair_json_object(value)
        return value


class ToolResult(StrictModel):
    tool_call_id: str
    tool_name: str
    status: Literal["success", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[SourceRef] = Field(default_factory=list)
    source_ts: datetime | None = None
    latency_ms: int = 0
    error: str | None = None


class ToolError(StrictModel):
    tool_call_id: str
    tool_name: str
    error_code: str
    message: str
    retryable: bool = False


class AgentOutput(StrictModel):
    agent_name: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskFlag(StrictModel):
    code: str
    severity: Literal["low", "medium", "high"]
    message: str
    evidence: str | None = None


class RiskReview(StrictModel):
    review_id: str = Field(default_factory=lambda: new_id("risk"))
    verdict: Literal["pass", "rewrite", "reject", "clarify"]
    flags: list[RiskFlag] = Field(default_factory=list)
    rewrite_suggestion: str | None = None
    self_reflection: list[str] = Field(default_factory=list)
    reviewed_text_hash: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class FinalResponse(StrictModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    risk_disclaimer: str
    generated_at: datetime = Field(default_factory=utc_now)


class TokenUsage(StrictModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CostUsage(StrictModel):
    amount: float = 0.0
    currency: str = "USD"


class StreamEvent(StrictModel):
    task_id: str
    event_id: str
    sequence: int
    checkpoint_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentState(StrictModel):
    trace_id: str
    tenant_id: str
    session_id: str
    user_id: str
    request_id: str
    task_id: str

    messages: list[Message]
    current_intent: IntentType | None = None
    intent_confidence: float = 0.0
    route_plan: list[RouteStep] = Field(default_factory=list)

    memory_context: MemoryContext = Field(default_factory=MemoryContext)
    short_term_summary: str | None = None
    retrieved_sources: list[SourceRef] = Field(default_factory=list)

    task_stack: list[TaskFrame] = Field(default_factory=list)
    active_task: TaskFrame | None = None
    interruption: InterruptionInfo | None = None

    pending_tools: list[ToolCall] = Field(default_factory=list)
    tool_results: dict[str, ToolResult] = Field(default_factory=dict)
    tool_errors: list[ToolError] = Field(default_factory=list)

    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)
    critic_report: RiskReview | None = None
    final_response: FinalResponse | None = None

    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    token_budget: int = 0
    cost_usage: CostUsage = Field(default_factory=CostUsage)
    status: Literal[
        "running",
        "waiting_tool",
        "needs_clarification",
        "risk_rejected",
        "completed",
        "failed",
    ] = "running"
