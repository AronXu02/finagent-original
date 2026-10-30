from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

SUPPORTED_TOOL_INPUTS: dict[str, type[BaseModel]] = {
    "get_realtime_stock": TickerMarketInput,
    "get_delayed_quote": TickerMarketInput,
    "get_historical_prices": HistoricalPricesInput,
    "macro_indicator": MacroIndicatorInput,
    "get_forex_rate": ForexRateInput,
    "fetch_sec_filings": SecFilingsInput,
    "fetch_cn_announcements": CNAnnouncementsInput,
    "web_search": ResearchSearchInput,
    "get_user_portfolio": UserPortfolioInput,
    "calc_position_value": PositionValueInput,
    "calc_portfolio_metrics": PortfolioMetricsInput,
    "run_scenario_analysis": ScenarioAnalysisInput,
    "analyze_impact": ImpactAnalysisInput,
    "compliance_rule_check": ComplianceCheckInput,
}

MASKED_LOSS_ROLES = {"system", "user", "tool"}


class SFTMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class TrajectorySample(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    messages: list[SFTMessage] = Field(min_length=2)


class SFTToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    location: str


class TrajectoryValidationResult(BaseModel):
    sample_id: str | None
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    tool_calls: list[SFTToolCall] = Field(default_factory=list)
    assistant_loss_message_indexes: list[int] = Field(default_factory=list)
    masked_roles: list[str] = Field(default_factory=lambda: sorted(MASKED_LOSS_ROLES))


class DatasetValidationReport(BaseModel):
    total: int
    valid: int
    invalid: int
    issues: list[ValidationIssue] = Field(default_factory=list)


def validate_jsonl(path: str | Path) -> DatasetValidationReport:
    issues: list[ValidationIssue] = []
    total = 0
    valid = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid_jsonl",
                        message=str(exc),
                        location=f"line:{line_number}",
                    )
                )
                continue
            result = validate_record(record, source_location=f"line:{line_number}")
            issues.extend(result.issues)
            if result.valid:
                valid += 1
    return DatasetValidationReport(total=total, valid=valid, invalid=total - valid, issues=issues)


def validate_record(record: dict[str, Any], *, source_location: str = "record") -> TrajectoryValidationResult:
    issues: list[ValidationIssue] = []
    tool_calls: list[SFTToolCall] = []
    assistant_indexes: list[int] = []
    if not isinstance(record, dict):
        return TrajectoryValidationResult(
            sample_id=None,
            valid=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    code="sample_schema_error",
                    message="JSONL records must be objects.",
                    location=source_location,
                )
            ],
        )
    sample_id = str(record.get("id")) if record.get("id") is not None else None
    try:
        sample = TrajectorySample.model_validate(record)
    except ValidationError as exc:
        return TrajectoryValidationResult(
            sample_id=sample_id,
            valid=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    code="sample_schema_error",
                    message=str(exc.errors()),
                    location=source_location,
                )
            ],
        )

    if sample.messages[0].role != "system":
        issues.append(
            ValidationIssue(
                severity="warning",
                code="missing_system_prompt",
                message="A ShareGPT SFT trajectory should start with a system message that declares tool schemas.",
                location=f"{source_location}.messages[0]",
            )
        )

    if not any(message.role == "assistant" for message in sample.messages):
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_assistant_output",
                message="At least one assistant message is required for SFT loss masking.",
                location=f"{source_location}.messages",
            )
        )

    for index, message in enumerate(sample.messages):
        location = f"{source_location}.messages[{index}]"
        if message.role == "assistant":
            assistant_indexes.append(index)
            calls, call_issues = extract_tool_calls(message.content, location=location)
            tool_calls.extend(calls)
            issues.extend(call_issues)
        elif message.role == "tool":
            issues.extend(_validate_tool_observation(message.content, location=location))
            if index == 0 or sample.messages[index - 1].role not in {"assistant", "tool"}:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="orphan_tool_observation",
                        message="Tool observations should follow an assistant tool call or another tool observation.",
                        location=location,
                    )
                )

    return TrajectoryValidationResult(
        sample_id=sample.id,
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        tool_calls=tool_calls,
        assistant_loss_message_indexes=assistant_indexes,
    )


def extract_tool_calls(content: str, *, location: str = "assistant.content") -> tuple[list[SFTToolCall], list[ValidationIssue]]:
    tool_calls: list[SFTToolCall] = []
    issues: list[ValidationIssue] = []
    for call_index, match in enumerate(TOOL_CALL_RE.finditer(content), start=1):
        call_location = f"{location}.tool_call[{call_index}]"
        try:
            raw_payload = json.loads(match.group(1))
            parsed = SFTToolCall.model_validate(raw_payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_tool_call_json",
                    message=str(exc),
                    location=call_location,
                )
            )
            continue

        canonical = canonicalize_tool_call(parsed)
        input_model = SUPPORTED_TOOL_INPUTS.get(canonical.name)
        if input_model is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unsupported_tool",
                    message=f"Tool is not registered in the runtime registry: {canonical.name}",
                    location=call_location,
                )
            )
            continue
        try:
            validated = input_model.model_validate(canonical.arguments)
        except ValidationError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="tool_schema_error",
                    message=str(exc.errors()),
                    location=call_location,
                )
            )
            continue
        tool_calls.append(SFTToolCall(name=canonical.name, arguments=validated.model_dump(mode="json")))
    if "<tool_call>" in content and not tool_calls and not any(issue.code == "invalid_tool_call_json" for issue in issues):
        issues.append(
            ValidationIssue(
                severity="error",
                code="tool_call_block_not_parsed",
                message="A tool_call tag was present but no valid tool call was parsed.",
                location=location,
            )
        )
    return tool_calls, issues


def canonicalize_tool_call(call: SFTToolCall) -> SFTToolCall:
    name = call.name
    arguments = dict(call.arguments)
    if name in {"macro_indicator", "get_macro_indicator"}:
        name = "macro_indicator"
        arguments = {
            "indicator": arguments.get("indicator") or arguments.get("event") or "fomc_rate_decision",
            "region": arguments.get("region", "US"),
            "latest": arguments.get("latest", True),
        }
    elif name in {"web_search", "search_research_reports"}:
        name = "web_search"
        arguments = {
            "query": arguments.get("query", ""),
            "ticker": arguments.get("ticker"),
            "top_k": arguments.get("top_k", 3),
        }
    elif name == "get_forex_rate":
        base = str(arguments.get("base", "USD")).strip().lower()
        quote = str(arguments.get("quote", "JPY")).strip().lower()
        arguments = {"base": base, "quote": quote}
    elif name == "fetch_sec_filings" and "year" in arguments and "fiscal_year" not in arguments:
        arguments["fiscal_year"] = arguments.pop("year")
    return SFTToolCall(name=name, arguments=arguments)


def _validate_tool_observation(content: str, *, location: str) -> list[ValidationIssue]:
    stripped = content.strip()
    if not stripped.startswith(("{", "[")):
        return []
    try:
        json.loads(stripped)
    except json.JSONDecodeError as exc:
        return [
            ValidationIssue(
                severity="error",
                code="invalid_tool_observation_json",
                message=str(exc),
                location=location,
            )
        ]
    return []
