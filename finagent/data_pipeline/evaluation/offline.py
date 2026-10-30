from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SEVERE_HALLUCINATION_RE = re.compile(
    r"(\u5fc5\u6da8|\u80af\u5b9a\u6da8|\u7a33\u8d5a|\u4fdd\u672c|\u65e0\u98ce\u9669|guaranteed|risk[- ]?free|sure profit|must rise)",
    re.IGNORECASE,
)


class ToolExecutionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: Literal["success", "error"]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    required_tools: list[str] = Field(default_factory=list)
    expected_memory_terms: list[str] = Field(default_factory=list)
    expected_final_status: Literal["completed", "risk_rejected", "needs_clarification"] = "completed"


class EvaluationPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    tool_results: list[ToolExecutionView] = Field(default_factory=list)
    final_status: str
    final_answer: str = ""
    memory_items: list[str] = Field(default_factory=list)
    risk_verdict: str | None = None
    risk_flag_codes: list[str] = Field(default_factory=list)


class OfflineEvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_count: int
    tool_success_rate: float
    goal_completion_rate: float
    memory_recall_rate: float | None
    severe_hallucination_rate: float
    missing_prediction_ids: list[str] = Field(default_factory=list)


def evaluate_offline(
    cases: list[EvaluationCase],
    predictions: list[EvaluationPrediction],
) -> OfflineEvaluationMetrics:
    prediction_by_id = {prediction.case_id: prediction for prediction in predictions}
    missing_prediction_ids = [case.case_id for case in cases if case.case_id not in prediction_by_id]
    evaluated_pairs = [(case, prediction_by_id[case.case_id]) for case in cases if case.case_id in prediction_by_id]

    required_tool_count = sum(len(case.required_tools) for case, _ in evaluated_pairs)
    successful_required_tools = sum(
        _successful_required_tools(case, prediction)
        for case, prediction in evaluated_pairs
    )
    completed_goals = sum(
        1
        for case, prediction in evaluated_pairs
        if prediction.final_status == case.expected_final_status
    )
    memory_expected_count = sum(len(case.expected_memory_terms) for case, _ in evaluated_pairs)
    memory_hits = sum(
        _memory_hits(case, prediction)
        for case, prediction in evaluated_pairs
    )
    severe_hallucinations = sum(
        1
        for _, prediction in evaluated_pairs
        if _has_severe_hallucination(prediction)
    )
    evaluated_count = len(evaluated_pairs)
    return OfflineEvaluationMetrics(
        case_count=evaluated_count,
        tool_success_rate=_rate(successful_required_tools, required_tool_count),
        goal_completion_rate=_rate(completed_goals, evaluated_count),
        memory_recall_rate=None if memory_expected_count == 0 else _rate(memory_hits, memory_expected_count),
        severe_hallucination_rate=_rate(severe_hallucinations, evaluated_count),
        missing_prediction_ids=missing_prediction_ids,
    )


def _successful_required_tools(case: EvaluationCase, prediction: EvaluationPrediction) -> int:
    successful_tools = {
        result.tool_name
        for result in prediction.tool_results
        if result.status == "success"
    }
    return sum(1 for tool_name in case.required_tools if tool_name in successful_tools)


def _memory_hits(case: EvaluationCase, prediction: EvaluationPrediction) -> int:
    memory_text = "\n".join(prediction.memory_items).lower()
    return sum(1 for term in case.expected_memory_terms if term.lower() in memory_text)


def _has_severe_hallucination(prediction: EvaluationPrediction) -> bool:
    if prediction.final_status == "risk_rejected" or prediction.risk_verdict == "reject":
        return False
    if "deterministic_return_promise" in prediction.risk_flag_codes:
        return True
    return bool(SEVERE_HALLUCINATION_RE.search(prediction.final_answer))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)
