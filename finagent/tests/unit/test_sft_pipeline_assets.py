from __future__ import annotations

import unittest

from finagent.data_pipeline.cleaning import validate_record
from finagent.data_pipeline.sft_generation import build_default_generation_plan
from finagent.data_pipeline.training import build_training_plan


class SFTPipelineAssetsTest(unittest.TestCase):
    def test_training_assets_exist_without_generated_dataset(self) -> None:
        plan = build_training_plan(require_dataset=False)

        self.assertTrue(plan.ready_without_dataset)
        self.assertEqual(plan.missing_assets, [])
        self.assertEqual(plan.command[0:2], ["llamafactory-cli", "train"])
        self.assertTrue(plan.train_config.endswith("qwen2_5_14b_finagent_sft.yaml"))
        self.assertTrue(plan.deepspeed_config.endswith("ds_z2_config.json"))
        self.assertTrue(plan.dataset_info.endswith("dataset_info.json"))

    def test_training_plan_can_require_dataset_later(self) -> None:
        plan = build_training_plan(require_dataset=True)

        self.assertTrue(any(path.endswith("finagent_tool_calling_30k.jsonl") for path in plan.missing_assets))

    def test_generation_plan_matches_documented_30k_mix(self) -> None:
        plan = build_default_generation_plan()

        self.assertEqual(plan.target_total, 30000)
        self.assertEqual(sum(item.target_count for item in plan.scenarios), 30000)
        self.assertEqual([item.target_count for item in plan.scenarios], [10000, 10000, 5000, 5000])

    def test_document_tool_aliases_are_canonicalized_to_runtime_tools(self) -> None:
        result = validate_record(
            {
                "id": "sample_alias",
                "messages": [
                    {"role": "system", "content": "Tools: macro_indicator, fetch_sec_filings"},
                    {
                        "role": "user",
                        "content": "What is the impact of a Fed rate hike on my bank holdings?",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "<tool_call>\n"
                            '{"name": "macro_indicator", "arguments": {"event": "fomc_rate_decision", "latest": true}}\n'
                            "</tool_call>"
                        ),
                    },
                ],
            }
        )

        self.assertTrue(result.valid, result.issues)
        self.assertEqual(result.tool_calls[0].name, "macro_indicator")
        self.assertEqual(result.tool_calls[0].arguments["indicator"], "fomc_rate_decision")

    def test_invalid_tool_arguments_are_rejected(self) -> None:
        result = validate_record(
            {
                "id": "bad_metrics",
                "messages": [
                    {"role": "system", "content": "Tools: calc_portfolio_metrics"},
                    {"role": "user", "content": "Calculate weights."},
                    {
                        "role": "assistant",
                        "content": (
                            "<tool_call>\n"
                            '{"name": "calc_portfolio_metrics", "arguments": {"weights": [0.6, 0.5]}}\n'
                            "</tool_call>"
                        ),
                    },
                ],
            }
        )

        self.assertFalse(result.valid)
        self.assertIn("tool_schema_error", {issue.code for issue in result.issues})


if __name__ == "__main__":
    unittest.main()
