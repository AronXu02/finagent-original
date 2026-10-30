from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from langgraph.graph.state import CompiledStateGraph

from finagent.packages.common.config import Settings
from finagent.packages.common.context import AuthContext
from finagent.packages.common.errors import ExternalDependencyError, ForbiddenError
from finagent.packages.common.sqlite_store import SQLiteStore
from finagent.packages.orchestrator.models import AgentState, IntentType, Message, TaskFrame
from finagent.packages.orchestrator.pipeline import FinAgentOrchestrator
from finagent.devtools import EmptyMarketGraphStore, InMemoryMarketDataProvider, InMemoryMemoryVectorStore


class FakeLLM:
    model = "qwen3.5:4b"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "\u8fd9\u662f Ollama \u751f\u6210\u7684\u7ec4\u5408\u98ce\u9669\u5206\u6790\u7b54\u590d\u3002"


class CapturingLLM:
    provider = "fake_llm"
    model = "qwen-meta-test"

    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "不是预设回答；我会根据当前问题调用模型生成，历史对话只作为背景记忆。"


class UnsafeLLM:
    model = "qwen3.5:4b"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "This trade is guaranteed and risk-free."


class FailingLLM:
    model = "qwen3.5:4b"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise ExternalDependencyError("Ollama 不可用")


def build_test_orchestrator(
    store: SQLiteStore,
    llm_client: object,
    settings: Settings | None = None,
) -> FinAgentOrchestrator:
    market_data = InMemoryMarketDataProvider()
    return FinAgentOrchestrator(
        store,
        settings or Settings(runtime_mode="test"),
        market_data_provider=market_data,
        memory_vector_store=InMemoryMemoryVectorStore(),
        graph_store=EmptyMarketGraphStore(),
        research_retriever=market_data,
        llm_client=llm_client,
    )


class PipelineTest(unittest.TestCase):
    def test_orchestrator_uses_compiled_langgraph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                graph = orchestrator.chat_graph.get_graph()

                self.assertIsInstance(orchestrator.chat_graph, CompiledStateGraph)
                self.assertIn("memory_retrieval", graph.nodes)
                self.assertIn("macro_agent", graph.nodes)
                self.assertIn("quant_agent", graph.nodes)
                self.assertIn("critic", graph.nodes)
                self.assertIn("final_builder", graph.nodes)
                self.assertNotIn("dispatch_agents", graph.nodes)
                self.assertIsNotNone(orchestrator.langgraph_checkpointer)

    def test_high_risk_request_is_reviewed_and_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="\u63a8\u8350\u4e00\u53ea\u660e\u5929\u80af\u5b9a\u6da8\u505c\u7684A\u80a1\uff0c\u6211\u60f3\u6ee1\u4ed3\u8fdb",
                    )
                )

                event_types = [event.event_type for event in events]
                self.assertIn("risk_review", event_types)
                compliance = next(
                    event
                    for event in events
                    if event.event_type == "tool_result"
                    and event.payload["output"]["tool_name"] == "compliance_rule_check"
                )
                final = events[-1]
                self.assertEqual(final.event_type, "final")
                self.assertIn("input", final.payload)
                self.assertIn("output", final.payload)
                self.assertEqual(
                    compliance.payload["input"]["arguments"]["user_input"],
                    "\u63a8\u8350\u4e00\u53ea\u660e\u5929\u80af\u5b9a\u6da8\u505c\u7684A\u80a1\uff0c\u6211\u60f3\u6ee1\u4ed3\u8fdb",
                )
                self.assertEqual(compliance.payload["output"]["data"]["verdict"], "reject")
                self.assertTrue(compliance.payload["output"]["data"]["user_input_checked"])
                self.assertIn("\u4e0d\u80fd\u63d0\u4f9b", final.payload["answer"])
                self.assertEqual(final.payload["status"], "risk_rejected")

    def test_rewrite_verdict_replaces_unsafe_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, UnsafeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="Please explain diversification.",
                    )
                )

                review = next(event for event in events if event.event_type == "risk_review")
                final = events[-1]

                self.assertEqual(review.payload["verdict"], "rewrite")
                self.assertEqual(review.payload["output"]["compliance_tool"]["data"]["verdict"], "rewrite")
                self.assertNotEqual(final.payload["answer"], "This trade is guaranteed and risk-free.")
                self.assertIn("\u89e6\u53d1\u5408\u89c4\u6539\u5199", final.payload["answer"])
                self.assertNotIn("guaranteed", final.payload["answer"].lower())

    def test_llm_failure_is_persisted_as_failed_task_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FailingLLM())
                auth = AuthContext(tenant_id="tenant_a", user_id="user_a")
                events = list(
                    orchestrator.run_chat(
                        auth=auth,
                        session_id="sess_a",
                        content="Please analyze my portfolio risk.",
                    )
                )

                task_id = events[-1].task_id
                task = store.validate_task_owner(auth.tenant_id, auth.user_id, task_id, "sess_a")
                replay = orchestrator.resume_events(
                    auth=auth,
                    task_id=task_id,
                    session_id="sess_a",
                    last_sequence=events[-2].sequence,
                )

                self.assertEqual(events[-1].event_type, "error")
                self.assertEqual(events[-1].payload["stage"], "candidate_response")
                self.assertEqual(events[-1].payload["output"]["code"], "external_dependency_error")
                self.assertEqual(task["status"], "failed")
                self.assertEqual(replay[-1].event_type, "error")

    def test_general_question_uses_education_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="Please explain diversification.",
                    )
                )

                routing = next(event for event in events if event.payload.get("stage") == "intent_routing")
                router = next(event for event in events if event.payload.get("stage") == "router")
                final = events[-1]

                self.assertEqual(routing.payload["output"]["intent"], "general_financial_question")
                self.assertIn("概念解释", router.payload["output"]["findings"][0])
                self.assertNotIn("time horizon", " ".join(router.payload["output"]["findings"]).lower())
                self.assertEqual(final.payload["answer"], "\u8fd9\u662f Ollama \u751f\u6210\u7684\u7ec4\u5408\u98ce\u9669\u5206\u6790\u7b54\u590d\u3002")
                self.assertGreater(final.payload["token_budget"], 0)
                self.assertGreater(final.payload["output"]["token_usage"]["prompt_tokens"], 0)
                self.assertGreater(final.payload["output"]["token_usage"]["completion_tokens"], 0)

    def test_meta_question_uses_llm_for_current_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                llm = CapturingLLM()
                orchestrator = build_test_orchestrator(store, llm)
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="你的回答是预设的吗？",
                    )
                )

                candidate = next(event for event in events if event.payload.get("stage") == "candidate_response")
                review = next(event for event in events if event.event_type == "risk_review")
                final = events[-1]

                self.assertIn("用户请求：\n你的回答是预设的吗？", llm.user_prompt)
                self.assertIn("必须先识别并回答当前用户请求", llm.system_prompt)
                self.assertEqual(candidate.payload["output"]["llm_provider"], "fake_llm")
                self.assertEqual(candidate.payload["output"]["llm_model"], "qwen-meta-test")
                self.assertEqual(review.payload["verdict"], "pass")
                self.assertEqual(
                    final.payload["answer"],
                    "不是预设回答；我会根据当前问题调用模型生成，历史对话只作为背景记忆。",
                )
                self.assertEqual(final.payload["status"], "completed")

    def test_final_event_warns_when_token_budget_crosses_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(
                    store,
                    FakeLLM(),
                    Settings(runtime_mode="test", token_budget_warning_threshold=1),
                )
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="Please explain diversification.",
                    )
                )

                final = events[-1]

                self.assertEqual(final.event_type, "final")
                self.assertTrue(final.payload["token_budget_warning"]["warning"])
                self.assertEqual(final.payload["token_budget_warning"]["threshold"], 1)
                self.assertTrue(final.payload["output"]["token_budget_warning"]["warning"])

    def test_resume_replays_events_after_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                auth = AuthContext(tenant_id="tenant_a", user_id="user_a")
                events = list(
                    orchestrator.run_chat(
                        auth=auth,
                        session_id="sess_a",
                        content="Please analyze my portfolio risk.",
                    )
                )
                task_id = events[-1].task_id
                replay = orchestrator.resume_events(
                    auth=auth,
                    task_id=task_id,
                    session_id="sess_a",
                    last_sequence=2,
                )

            self.assertTrue(replay)
            self.assertEqual(replay[0].sequence, 3)
            self.assertEqual(replay[-1].event_type, "final")

    def test_tool_events_include_step_input_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="Please analyze my portfolio risk.",
                    )
                )

                tool_events = [event for event in events if event.event_type == "tool_result"]
                self.assertTrue(tool_events)
                self.assertIn("input", tool_events[0].payload)
                self.assertIn("output", tool_events[0].payload)
                self.assertIn("agent", tool_events[0].payload)
                self.assertIn("step_id", tool_events[0].payload)
                self.assertIn("arguments", tool_events[0].payload["input"])
                self.assertIn("idempotency_key", tool_events[0].payload["input"])
                self.assertIn("status", tool_events[0].payload["output"])

    def test_market_quote_resolves_chinese_asset_name_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="\u67e5\u4e00\u4e0b\u82f9\u679c\u80a1\u4ef7",
                    )
                )

                routing = next(event for event in events if event.payload.get("stage") == "intent_routing")
                quote_event = next(
                    event
                    for event in events
                    if event.event_type == "tool_result"
                    and event.payload["output"]["tool_name"] == "get_realtime_stock"
                )

                self.assertEqual(routing.payload["output"]["intent"], "market_quote")
                self.assertEqual(quote_event.payload["input"]["arguments"], {"ticker": "AAPL", "market": "US"})
                self.assertEqual(quote_event.payload["output"]["status"], "success")

    def test_portfolio_impact_dispatches_declared_function_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="\u7f8e\u8054\u50a8\u52a0\u606f\u5bf9\u6211\u7684\u6301\u4ed3\u5f71\u54cd",
                    )
                )

                routing = next(event for event in events if event.payload.get("stage") == "intent_routing")
                tool_events = [event for event in events if event.event_type == "tool_result"]
                emitted_tools = [event.payload["output"]["tool_name"] for event in tool_events]
                emitted_agents = [event.payload["agent"] for event in tool_events]
                emitted_steps = [event.payload["step_id"] for event in tool_events]
                quant_output = next(event for event in events if event.payload.get("stage") == "quant_agent")
                quant_metadata = quant_output.payload["output"]["metadata"]

                self.assertEqual(routing.payload["output"]["intent"], "portfolio_impact")
                self.assertEqual(routing.payload["output"]["route_plan"][0]["node"], "macro_agent")
                self.assertIsNone(quant_metadata["analysis_context"]["sector"])
                self.assertEqual(quant_metadata["analysis_context"]["scenario"], "rate_hike_25bps")
                self.assertEqual(len(quant_metadata["portfolio"]["holdings"]), 3)
                self.assertEqual(
                    emitted_tools,
                    [
                        "macro_indicator",
                        "web_search",
                        "get_user_portfolio",
                        "get_historical_prices",
                        "get_historical_prices",
                        "get_historical_prices",
                        "calc_portfolio_metrics",
                        "run_scenario_analysis",
                        "analyze_impact",
                        "compliance_rule_check",
                    ],
                )
                self.assertEqual(
                    emitted_agents,
                    [
                        "macro_agent",
                        "macro_agent",
                        "quant_agent",
                        "quant_agent",
                        "quant_agent",
                        "quant_agent",
                        "quant_agent",
                        "quant_agent",
                        "quant_agent",
                        "critic",
                    ],
                )
                self.assertEqual(
                    emitted_steps,
                    [
                        "route_step_1",
                        "route_step_1",
                        "route_step_2",
                        "route_step_2",
                        "route_step_2",
                        "route_step_2",
                        "route_step_2",
                        "route_step_2",
                        "route_step_2",
                        "route_step_3",
                    ],
                )

    def test_portfolio_impact_sector_hint_filters_holdings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="\u7f8e\u8054\u50a8\u52a0\u606f\u5bf9\u6211\u7684\u94f6\u884c\u80a1\u6301\u4ed3\u5f71\u54cd",
                    )
                )

                quant_output = next(event for event in events if event.payload.get("stage") == "quant_agent")
                metadata = quant_output.payload["output"]["metadata"]
                tool_events = [event for event in events if event.event_type == "tool_result"]
                emitted_tools = [event.payload["output"]["tool_name"] for event in tool_events]

                self.assertEqual(metadata["analysis_context"]["sector"], "Financials")
                self.assertEqual(metadata["analysis_context"]["scenario"], "rate_hike_25bps")
                self.assertEqual(
                    [holding["ticker"] for holding in metadata["portfolio"]["holdings"]],
                    ["JPM", "BAC"],
                )
                self.assertEqual(emitted_tools.count("get_historical_prices"), 2)

    def test_cpi_portfolio_impact_uses_dynamic_macro_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="CPI \u901a\u80c0\u5bf9\u6211\u7684\u79d1\u6280\u80a1\u6301\u4ed3\u5f71\u54cd",
                    )
                )

                routing = next(event for event in events if event.payload.get("stage") == "intent_routing")
                macro_output = next(event for event in events if event.payload.get("stage") == "macro_agent")
                quant_output = next(event for event in events if event.payload.get("stage") == "quant_agent")

                self.assertEqual(routing.payload["output"]["intent"], "portfolio_impact")
                self.assertEqual(macro_output.payload["output"]["metadata"]["macro_request"]["indicator"], "cpi_yoy")
                self.assertEqual(quant_output.payload["output"]["metadata"]["analysis_context"]["sector"], "Technology")
                self.assertEqual(quant_output.payload["output"]["metadata"]["analysis_context"]["scenario"], "sector_stress")

    def test_preference_update_uses_memory_agent_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                events = list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="\u6700\u5927\u56de\u64a4\u4e0d\u80fd\u8d85\u8fc78%\uff0c\u4e0d\u4e70ST\u80a1",
                    )
                )

                routing = next(event for event in events if event.payload.get("stage") == "intent_routing")
                memory_output = next(event for event in events if event.payload.get("stage") == "memory_agent")

                self.assertEqual(routing.payload["output"]["intent"], "preference_update")
                self.assertEqual(routing.payload["output"]["route_plan"][0]["node"], "memory_agent")
                self.assertIsNone(routing.payload["input"]["previous_active_task"])
                self.assertEqual(routing.payload["output"]["active_task"]["intent"], "preference_update")
                self.assertEqual(routing.payload["output"]["task_stack"], [])
                self.assertIn("Maximum drawdown tolerance is 8%.", memory_output.payload["output"]["findings"])

    def test_session_owner_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                list(
                    orchestrator.run_chat(
                        auth=AuthContext(tenant_id="tenant_a", user_id="user_a"),
                        session_id="sess_a",
                        content="Please analyze my portfolio risk.",
                    )
                )

                with self.assertRaises(ForbiddenError):
                    list(
                        orchestrator.run_chat(
                            auth=AuthContext(tenant_id="tenant_a", user_id="user_b"),
                            session_id="sess_a",
                            content="Can I access this session?",
                        )
                    )

    def test_interrupt_loads_previous_running_task_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                orchestrator = build_test_orchestrator(store, FakeLLM())
                auth = AuthContext(tenant_id="tenant_a", user_id="user_a")
                store.ensure_conversation(auth.tenant_id, auth.user_id, "sess_a")
                store.create_task(auth.tenant_id, "sess_a", auth.user_id, "task_previous", "trace_previous")
                previous_state = AgentState(
                    trace_id="trace_previous",
                    tenant_id=auth.tenant_id,
                    session_id="sess_a",
                    user_id=auth.user_id,
                    request_id="req_previous",
                    task_id="task_previous",
                    messages=[Message(role="user", content="Analyze NVDA 10-K risk factors.")],
                    active_task=TaskFrame(
                        task_id="task_previous",
                        intent=IntentType.REPORT_ANALYSIS,
                        description="Analyze NVDA 10-K risk factors.",
                    ),
                    status="running",
                )
                store.save_checkpoint(auth.tenant_id, "task_previous", previous_state)

                events = list(
                    orchestrator.run_chat(
                        auth=auth,
                        session_id="sess_a",
                        content="wait, what is the USD/JPY rate?",
                    )
                )

                routing = next(event for event in events if event.payload.get("stage") == "intent_routing")
                interrupt_final = next(
                    event
                    for event in events
                    if event.event_type == "final" and event.task_id != "task_previous"
                )
                resumed_start = next(
                    event
                    for event in events
                    if event.payload.get("stage") == "resume_task"
                )
                resumed_final = next(
                    event
                    for event in events
                    if event.event_type == "final" and event.task_id == "task_previous"
                )
                latest = store.get_latest_session_state(auth.tenant_id, "sess_a", auth.user_id)

                self.assertEqual(routing.payload["input"]["previous_active_task"]["task_id"], "task_previous")
                self.assertIsNotNone(routing.payload["input"]["previous_active_task"]["checkpoint_id"])
                self.assertEqual(routing.payload["output"]["task_stack"][0]["task_id"], "task_previous")
                self.assertEqual(routing.payload["output"]["interruption"]["interrupted_task_id"], "task_previous")
                self.assertEqual(interrupt_final.payload["resumed_task"]["task_id"], "task_previous")
                self.assertIn("\u5df2\u6062\u590d\u4e4b\u524d\u6682\u505c\u7684\u4efb\u52a1", interrupt_final.payload["answer"])
                self.assertEqual(resumed_start.task_id, "task_previous")
                self.assertIsNotNone(resumed_start.payload["output"]["checkpoint_id"])
                self.assertEqual(resumed_final.payload["status"], "completed")
                self.assertIsNotNone(latest)
                self.assertEqual(latest.active_task.task_id, "task_previous")
                self.assertIsNotNone(latest.active_task.checkpoint_id)
                self.assertEqual(latest.task_stack, [])


if __name__ == "__main__":
    unittest.main()
