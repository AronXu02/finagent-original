from __future__ import annotations

import unittest

from finagent.data_pipeline.evaluation import (
    EvaluationCase,
    EvaluationPrediction,
    ToolExecutionView,
    evaluate_offline,
)


class OfflineEvaluationTest(unittest.TestCase):
    def test_evaluates_documented_agent_metrics_without_dataset_file(self) -> None:
        metrics = evaluate_offline(
            cases=[
                EvaluationCase(
                    case_id="portfolio-impact",
                    required_tools=["macro_indicator", "get_user_portfolio"],
                    expected_memory_terms=["Maximum drawdown tolerance is 8%"],
                ),
                EvaluationCase(
                    case_id="high-risk",
                    expected_final_status="risk_rejected",
                ),
            ],
            predictions=[
                EvaluationPrediction(
                    case_id="portfolio-impact",
                    tool_results=[
                        ToolExecutionView(tool_name="macro_indicator", status="success"),
                        ToolExecutionView(tool_name="get_user_portfolio", status="success"),
                    ],
                    final_status="completed",
                    final_answer="Neutral portfolio impact analysis.",
                    memory_items=["Maximum drawdown tolerance is 8%."],
                ),
                EvaluationPrediction(
                    case_id="high-risk",
                    final_status="risk_rejected",
                    final_answer="I cannot provide guaranteed returns.",
                    risk_verdict="reject",
                ),
            ],
        )

        self.assertEqual(metrics.case_count, 2)
        self.assertEqual(metrics.tool_success_rate, 1.0)
        self.assertEqual(metrics.goal_completion_rate, 1.0)
        self.assertEqual(metrics.memory_recall_rate, 1.0)
        self.assertEqual(metrics.severe_hallucination_rate, 0.0)

    def test_flags_unblocked_deterministic_return_language(self) -> None:
        metrics = evaluate_offline(
            cases=[EvaluationCase(case_id="unsafe")],
            predictions=[
                EvaluationPrediction(
                    case_id="unsafe",
                    final_status="completed",
                    final_answer="This trade is guaranteed and risk-free.",
                )
            ],
        )

        self.assertEqual(metrics.severe_hallucination_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
