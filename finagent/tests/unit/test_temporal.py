from __future__ import annotations

import unittest

from finagent.packages.orchestrator.temporal import (
    TEMPORAL_ACTIVITIES,
    TEMPORAL_AGENT_ACTIVITIES,
    _temporal_next_node,
)


class TemporalWorkflowTest(unittest.TestCase):
    def test_temporal_registers_stage_activities(self) -> None:
        activity_names = {activity.__name__ for activity in TEMPORAL_ACTIVITIES}

        self.assertIn("prepare_chat_activity", activity_names)
        self.assertIn("memory_routing_activity", activity_names)
        self.assertIn("macro_agent_activity", activity_names)
        self.assertIn("quant_agent_activity", activity_names)
        self.assertIn("candidate_response_activity", activity_names)
        self.assertIn("critic_activity", activity_names)
        self.assertIn("final_response_activity", activity_names)
        self.assertEqual(TEMPORAL_AGENT_ACTIVITIES["macro_agent"].__name__, "macro_agent_activity")
        self.assertEqual(TEMPORAL_AGENT_ACTIVITIES["quant_agent"].__name__, "quant_agent_activity")

    def test_temporal_next_node_inserts_candidate_before_critic(self) -> None:
        snapshot = {
            "route_index": 0,
            "state": {
                "route_plan": [
                    {"node": "critic"},
                    {"node": "final_builder"},
                ],
            },
        }

        self.assertEqual(_temporal_next_node(snapshot), "candidate_response")

        snapshot["candidate"] = {"answer": "ok", "sources": []}
        self.assertEqual(_temporal_next_node(snapshot), "critic")


if __name__ == "__main__":
    unittest.main()
