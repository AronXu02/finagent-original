from __future__ import annotations

import unittest

from finagent.packages.agents.builders import CandidateResponseBuilder
from finagent.packages.common.errors import ExternalDependencyError
from finagent.packages.orchestrator.models import AgentOutput, AgentState, Message


class FakeLLM:
    model = "qwen-test:4b"

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response or "\u8fd9\u662f Ollama \u751f\u6210\u7684\u4e2d\u6587\u56de\u7b54\u3002"
        self.error = error
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if self.error:
            raise self.error
        return self.response


class CandidateResponseBuilderTest(unittest.TestCase):
    def test_uses_injected_llm_for_answer_text(self) -> None:
        llm = FakeLLM()
        state = self._state()

        response = CandidateResponseBuilder(llm_client=llm).build(state, "\u8bf7\u5206\u6790\u7ec4\u5408\u98ce\u9669")

        self.assertEqual(response.answer, "\u8fd9\u662f Ollama \u751f\u6210\u7684\u4e2d\u6587\u56de\u7b54\u3002")
        self.assertEqual(response.llm_provider, "ollama")
        self.assertEqual(response.llm_model, "qwen-test:4b")
        self.assertIn("\u4e2d\u6587", llm.system_prompt)
        self.assertIn("\u4e0d\u8981\u5957\u7528\u56fa\u5b9a\u4e09\u6bb5\u5f0f\u6a21\u677f", llm.system_prompt)
        self.assertIn("\u5ba1\u614e\u53e3\u5f84", llm.system_prompt)
        self.assertIn("quant_agent", llm.user_prompt)
        self.assertIn("\u5143\u6570\u636e", llm.user_prompt)
        self.assertNotIn("\u7ed3\u8bba\u6458\u8981\u3001\u4e3b\u8981\u98ce\u9669\u6216\u89c2\u5bdf\u3001\u5efa\u8bae\u7684\u4e0b\u4e00\u6b65", llm.user_prompt)

    def test_relevance_feedback_uses_llm_for_current_question(self) -> None:
        llm = FakeLLM(response="我会重新对准当前问题。")
        state = self._state()

        response = CandidateResponseBuilder(llm_client=llm).build(
            state,
            "\u4f60\u56de\u7b54\u7684\u4e0d\u592a\u76f8\u5173",
        )

        self.assertEqual(response.answer, "我会重新对准当前问题。")
        self.assertEqual(response.llm_provider, "ollama")
        self.assertEqual(response.llm_model, "qwen-test:4b")
        self.assertIn("必须先识别并回答当前用户请求", llm.system_prompt)
        self.assertIn("\u4f60\u56de\u7b54\u7684\u4e0d\u592a\u76f8\u5173", llm.user_prompt)

    def test_capability_question_uses_llm_with_history_as_memory(self) -> None:
        llm = FakeLLM(response="我会根据当前问题说明能力边界。")
        state = self._state()
        state.short_term_summary = "goals=Analyze old portfolio risk ; assets=AAPL,JPM,MSFT"
        state.messages = [
            Message(role="user", content="请分析我的组合风险"),
            Message(role="assistant", content="旧的组合风险报告"),
            Message(role="user", content="你能回答什么"),
        ]

        response = CandidateResponseBuilder(llm_client=llm).build(state, "你能回答什么")

        self.assertEqual(response.answer, "我会根据当前问题说明能力边界。")
        self.assertEqual(response.llm_provider, "ollama")
        self.assertEqual(response.llm_model, "qwen-test:4b")
        self.assertIn("用户请求：\n你能回答什么", llm.user_prompt)
        self.assertIn("历史对话记忆（仅作背景，不是当前问题）", llm.user_prompt)
        self.assertIn("旧的组合风险报告", llm.user_prompt)
        self.assertIn("先回答当前用户请求本身", llm.user_prompt)

    def test_includes_short_term_summary_and_recent_dialogue_context(self) -> None:
        llm = FakeLLM()
        state = self._state()
        state.short_term_summary = "goals=Analyze AAPL portfolio risk ; assets=AAPL"
        state.messages = [
            Message(role="user", content="请分析 AAPL 持仓风险"),
            Message(role="assistant", content="我会先检查组合和价格数据。"),
            Message(role="user", content="那这个风险呢"),
        ]

        CandidateResponseBuilder(llm_client=llm).build(state, "那这个风险呢")

        self.assertIn("短期对话记忆", llm.user_prompt)
        self.assertIn("goals=Analyze AAPL portfolio risk", llm.user_prompt)
        self.assertIn("历史对话记忆", llm.user_prompt)
        self.assertIn("用户：请分析 AAPL 持仓风险", llm.user_prompt)
        self.assertIn("助手：我会先检查组合和价格数据。", llm.user_prompt)

    def test_new_question_marks_stale_dialogue_as_memory_only(self) -> None:
        llm = FakeLLM()
        state = self._state()
        state.short_term_summary = "goals=Analyze old portfolio risk ; assets=AAPL,JPM,MSFT"
        state.messages = [
            Message(role="user", content="请分析我的组合风险"),
            Message(role="assistant", content="旧的组合风险报告"),
            Message(role="user", content="请简单说明分散投资的风险边界"),
        ]

        CandidateResponseBuilder(llm_client=llm).build(state, "请简单说明分散投资的风险边界")

        self.assertIn("短期对话记忆（仅作背景，不是当前任务）", llm.user_prompt)
        self.assertIn("历史对话记忆（仅作背景，不是当前问题）", llm.user_prompt)
        self.assertIn("旧的组合风险报告", llm.user_prompt)
        self.assertIn("先回答当前用户请求本身", llm.user_prompt)

    def test_raises_when_llm_is_missing(self) -> None:
        state = self._state()

        with self.assertRaises(ExternalDependencyError):
            CandidateResponseBuilder().build(state, "\u8bf7\u5206\u6790\u7ec4\u5408\u98ce\u9669")

    def test_propagates_llm_errors_without_fixed_answer_fallback(self) -> None:
        state = self._state()

        with self.assertRaises(RuntimeError):
            CandidateResponseBuilder(llm_client=FakeLLM(error=RuntimeError("offline"))).build(
                state,
                "\u8bf7\u5206\u6790\u7ec4\u5408\u98ce\u9669",
            )

    @staticmethod
    def _state() -> AgentState:
        state = AgentState(
            trace_id="trace",
            tenant_id="tenant",
            session_id="session",
            user_id="user",
            request_id="request",
            task_id="task",
            messages=[Message(role="user", content="\u8bf7\u5206\u6790\u7ec4\u5408\u98ce\u9669")],
        )
        state.agent_outputs["quant_agent"] = AgentOutput(
            agent_name="quant_agent",
            summary="Portfolio metrics completed.",
            findings=["VaR 95 estimate: 1200.00."],
        )
        return state


if __name__ == "__main__":
    unittest.main()
