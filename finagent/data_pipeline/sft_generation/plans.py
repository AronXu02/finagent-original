from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ScenarioKind = Literal[
    "single_tool_call",
    "multi_tool_call",
    "critic_rewrite",
    "interruption_recovery",
]


class TrajectoryScenario(BaseModel):
    kind: ScenarioKind
    target_count: int = Field(gt=0)
    intent_family: str
    required_tools: list[str] = Field(default_factory=list)
    source_mix: dict[str, float] = Field(default_factory=dict)


class TrajectoryGenerationPlan(BaseModel):
    dataset_name: str = "finagent_tool_calling_30k"
    target_total: int = 30_000
    tokenizer_cutoff_len: int = 2048
    scenarios: list[TrajectoryScenario] = Field(default_factory=list)

    @model_validator(mode="after")
    def target_counts_match(self) -> "TrajectoryGenerationPlan":
        total = sum(item.target_count for item in self.scenarios)
        if total != self.target_total:
            raise ValueError(f"scenario target counts must sum to {self.target_total}, got {total}")
        return self


DEFAULT_TRAJECTORY_SCENARIOS: tuple[TrajectoryScenario, ...] = (
    TrajectoryScenario(
        kind="single_tool_call",
        target_count=10_000,
        intent_family="quote_or_metric_lookup",
        required_tools=["get_realtime_stock", "calc_portfolio_metrics"],
        source_mix={"self_instruct": 0.6, "business_log_seed": 0.4},
    ),
    TrajectoryScenario(
        kind="multi_tool_call",
        target_count=10_000,
        intent_family="macro_micro_portfolio_impact",
        required_tools=[
            "macro_indicator",
            "web_search",
            "get_user_portfolio",
            "get_historical_prices",
            "calc_portfolio_metrics",
            "run_scenario_analysis",
            "analyze_impact",
        ],
        source_mix={"self_instruct": 0.6, "mock_tool_trace": 0.4},
    ),
    TrajectoryScenario(
        kind="critic_rewrite",
        target_count=5_000,
        intent_family="compliance_and_risk_rewrite",
        required_tools=["compliance_rule_check"],
        source_mix={"adversarial_seed": 0.5, "business_log_seed": 0.5},
    ),
    TrajectoryScenario(
        kind="interruption_recovery",
        target_count=5_000,
        intent_family="task_stack_and_resume",
        required_tools=["fetch_sec_filings", "get_forex_rate", "get_realtime_stock"],
        source_mix={"human_authored_core": 0.4, "scripted_variation": 0.6},
    ),
)


def build_default_generation_plan() -> TrajectoryGenerationPlan:
    return TrajectoryGenerationPlan(scenarios=list(DEFAULT_TRAJECTORY_SCENARIOS))
