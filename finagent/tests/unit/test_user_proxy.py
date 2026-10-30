from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finagent.packages.agents.market import MarketAgent
from finagent.packages.agents.report import ReportAgent
from finagent.packages.agents.asset_resolver import clear_asset_alias_cache, resolve_asset_alias
from finagent.packages.agents.user_proxy import LightweightIntentClassifier, UserProxy, build_intent_classifier
from finagent.packages.common.config import Settings
from finagent.packages.common.errors import ExternalDependencyError
from finagent.packages.common.sqlite_store import SQLiteStore
from finagent.packages.orchestrator.models import AgentState, IntentType, Message, TaskFrame
from finagent.packages.tools.executor import build_default_registry
from finagent.devtools import InMemoryMarketDataProvider


def make_state() -> AgentState:
    return AgentState(
        trace_id="trace",
        tenant_id="tenant",
        session_id="session",
        user_id="user",
        request_id="request",
        task_id="task_current",
        messages=[Message(role="user", content="hello")],
    )


class UserProxyTest(unittest.TestCase):
    def test_preference_update_takes_priority_over_portfolio_risk_keywords(self) -> None:
        intent, confidence = UserProxy().classify(
            "\u6700\u5927\u56de\u64a4\u4e0d\u80fd\u8d85\u8fc78%\uff0c\u4e0d\u4e70ST\u80a1"
        )

        self.assertEqual(intent, IntentType.PREFERENCE_UPDATE)
        self.assertGreaterEqual(confidence, 0.9)

    def test_plain_portfolio_risk_still_routes_to_quant_agent(self) -> None:
        state = UserProxy().plan(make_state(), "\u8bf7\u5206\u6790\u6211\u7684\u7ec4\u5408\u56de\u64a4\u98ce\u9669")

        self.assertEqual(state.current_intent, IntentType.PORTFOLIO_RISK)
        self.assertEqual(state.route_plan[0].node, "quant_agent")
        self.assertEqual(state.active_task.intent, IntentType.PORTFOLIO_RISK)

    def test_general_risk_education_does_not_route_to_quant_agent(self) -> None:
        state = UserProxy().plan(make_state(), "\u8bf7\u7b80\u5355\u8bf4\u660e\u5206\u6563\u6295\u8d44\u7684\u98ce\u9669\u8fb9\u754c")

        self.assertEqual(state.current_intent, IntentType.GENERAL)
        self.assertEqual(state.route_plan[0].node, "router")

    def test_macro_shock_with_holdings_routes_to_portfolio_impact(self) -> None:
        state = UserProxy().plan(make_state(), "CPI \u901a\u80c0\u5bf9\u6211\u7684\u79d1\u6280\u80a1\u6301\u4ed3\u5f71\u54cd")

        self.assertEqual(state.current_intent, IntentType.PORTFOLIO_IMPACT)
        self.assertEqual(state.route_plan[0].node, "macro_agent")
        self.assertEqual(state.route_plan[1].node, "quant_agent")

    def test_macro_shock_without_holdings_routes_to_macro_agent(self) -> None:
        state = UserProxy().plan(make_state(), "\u539f\u6cb9\u4ef7\u683c\u4e0a\u6da8\u7684\u5b8f\u89c2\u5f71\u54cd\u662f\u4ec0\u4e48")

        self.assertEqual(state.current_intent, IntentType.MACRO_EVENT)
        self.assertEqual(state.route_plan[0].node, "macro_agent")

    def test_preference_update_routes_to_memory_agent(self) -> None:
        state = UserProxy().plan(make_state(), "\u6211\u7684\u98ce\u9669\u504f\u597d\u662f\u907f\u514dST")

        self.assertEqual(state.current_intent, IntentType.PREFERENCE_UPDATE)
        self.assertEqual(state.route_plan[0].node, "memory_agent")
        self.assertEqual(state.route_plan[1].node, "critic")

    def test_interrupt_signal_pushes_existing_active_task_to_stack(self) -> None:
        state = make_state()
        state.active_task = TaskFrame(
            task_id="task_previous",
            intent=IntentType.PORTFOLIO_RISK,
            description="previous portfolio analysis",
        )

        planned = UserProxy().plan(state, "\u7b49\u7b49\uff0c\u5148\u770b\u4e00\u4e0b\u65e5\u5143\u6c47\u7387")

        self.assertEqual(planned.current_intent, IntentType.MARKET_QUOTE)
        self.assertEqual(len(planned.task_stack), 1)
        self.assertEqual(planned.task_stack[0].task_id, "task_previous")
        self.assertIsNotNone(planned.interruption)
        self.assertEqual(planned.interruption.interrupted_task_id, "task_previous")

    def test_interrupt_signal_bypasses_remote_classifier(self) -> None:
        class FailingClassifier:
            model_name = "failing-classifier"

            def classify(self, text: str):
                raise AssertionError("remote classifier should not be called for explicit interruptions")

        state = make_state()
        state.active_task = TaskFrame(
            task_id="task_previous",
            intent=IntentType.REPORT_ANALYSIS,
            description="previous report analysis",
        )

        planned = UserProxy(intent_classifier=FailingClassifier()).plan(state, "\u7b49\u7b49\uff0c\u5148\u770b\u4e00\u4e0b\u65e5\u5143\u6c47\u7387")

        self.assertEqual(planned.current_intent, IntentType.MARKET_QUOTE)
        self.assertEqual(planned.route_plan[0].node, "market_agent")
        self.assertEqual(planned.task_stack[0].task_id, "task_previous")

    def test_followup_reference_inherits_active_task(self) -> None:
        state = make_state()
        state.active_task = TaskFrame(
            task_id="task_previous",
            intent=IntentType.PORTFOLIO_RISK,
            description="previous portfolio analysis",
        )

        planned = UserProxy().plan(state, "\u90a3\u8fd9\u4e2a\u98ce\u9669\u5462")

        self.assertEqual(planned.current_intent, IntentType.PORTFOLIO_RISK)
        self.assertGreaterEqual(planned.intent_confidence, 0.72)

    def test_ambiguous_followup_without_active_task_requests_clarification(self) -> None:
        planned = UserProxy().plan(make_state(), "\u90a3\u8fd9\u4e2a\u5462")

        self.assertEqual(planned.current_intent, IntentType.CLARIFICATION)
        self.assertEqual(planned.route_plan[0].node, "router")

    def test_relevance_feedback_requests_clarification(self) -> None:
        planned = UserProxy().plan(make_state(), "\u4f60\u56de\u7b54\u7684\u4e0d\u592a\u76f8\u5173")

        self.assertEqual(planned.current_intent, IntentType.CLARIFICATION)
        self.assertEqual(planned.route_plan[0].node, "router")

    def test_capability_question_does_not_inherit_previous_task(self) -> None:
        state = make_state()
        state.active_task = TaskFrame(
            task_id="task_previous",
            intent=IntentType.PORTFOLIO_RISK,
            description="previous portfolio analysis",
        )

        planned = UserProxy().plan(state, "\u4f60\u80fd\u56de\u7b54\u4ec0\u4e48")

        self.assertEqual(planned.current_intent, IntentType.GENERAL)
        self.assertGreaterEqual(planned.intent_confidence, 0.9)
        self.assertEqual(planned.route_plan[0].node, "router")

    def test_portfolio_followup_without_active_task_uses_recent_context(self) -> None:
        state = make_state()
        state.short_term_summary = "goals=Analyze portfolio risk; assets=AAPL,JPM,MSFT"
        state.messages = [
            Message(role="user", content="\u8bf7\u5206\u6790\u6211\u7684\u7ec4\u5408\u98ce\u9669"),
            Message(role="assistant", content="\u7ec4\u5408\u98ce\u9669\u62a5\u544a\u9700\u8981\u5386\u53f2\u884c\u60c5\u548c\u6301\u4ed3\u660e\u7ec6"),
        ]

        planned = UserProxy().plan(state, "\u7ee7\u7eed\u7ed3\u5408\u6211\u7684\u98ce\u9669\u504f\u597d\u5206\u6790")

        self.assertEqual(planned.current_intent, IntentType.PORTFOLIO_RISK)
        self.assertEqual(planned.route_plan[0].node, "quant_agent")

    def test_default_classifier_exposes_lightweight_classifier_layer(self) -> None:
        proxy = UserProxy()

        self.assertIsInstance(proxy.intent_classifier, LightweightIntentClassifier)
        self.assertIn("classifier", proxy.intent_classifier.model_name)

    def test_settings_can_build_rules_classifier_for_tests(self) -> None:
        classifier = build_intent_classifier(Settings(runtime_mode="test"))

        self.assertIsInstance(classifier, LightweightIntentClassifier)

    def test_http_bert_classifier_requires_gateway_url(self) -> None:
        with self.assertRaises(ExternalDependencyError):
            build_intent_classifier(Settings(runtime_mode="production", intent_classifier_backend="http_bert"))

    def test_route_plan_tools_are_registered(self) -> None:
        examples = [
            "\u7f8e\u8054\u50a8\u52a0\u606f\u5bf9\u6211\u7684\u6301\u4ed3\u5f71\u54cd",
            "NVDA 10-K risk factors",
            "\u8bf7\u5206\u6790\u6211\u7684\u7ec4\u5408\u98ce\u9669",
            "\u67e5\u4e00\u4e0b\u82f9\u679c\u80a1\u4ef7",
            "\u7f8e\u8054\u50a8\u964d\u606f\u7684\u5f71\u54cd\u662f\u4ec0\u4e48",
            "\u63a8\u8350\u4e00\u53ea\u660e\u5929\u80af\u5b9a\u6da8\u505c\u7684A\u80a1",
            "\u6211\u4e0d\u4e70ST\u80a1",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                market_data = InMemoryMarketDataProvider()
                registry_names = set(
                    build_default_registry(
                        Settings(runtime_mode="test"),
                        store,
                        market_data_provider=market_data,
                        research_retriever=market_data,
                    ).names()
                )

        for text in examples:
            with self.subTest(text=text):
                state = UserProxy().plan(make_state(), text)
                tools = {tool for step in state.route_plan for tool in step.tools}
                self.assertLessEqual(tools, registry_names)

    def test_market_quote_resolves_common_asset_names_without_demo_fallback(self) -> None:
        self.assertEqual(MarketAgent._extract_quote_target("\u67e5\u4e00\u4e0b\u82f9\u679c\u80a1\u4ef7"), ("AAPL", "US"))
        self.assertEqual(MarketAgent._extract_quote_target("\u8d35\u5dde\u8305\u53f0\u80a1\u4ef7"), ("600519.SH", "CN"))
        self.assertEqual(MarketAgent._extract_quote_target("\u817e\u8baf\u63a7\u80a1\u884c\u60c5"), ("00700.HK", "HK"))
        self.assertIsNone(MarketAgent._extract_quote_target("\u67e5\u4e00\u4e0b\u80a1\u4ef7"))
        self.assertEqual(MarketAgent._extract_quote_target("AAPL price"), ("AAPL", "US"))
        self.assertEqual(MarketAgent._extract_quote_target("Apple price"), ("AAPL", "US"))
        self.assertEqual(MarketAgent._extract_quote_target("What is MSFT price?"), ("MSFT", "US"))
        self.assertEqual(MarketAgent._extract_quote_target("$tsla quote"), ("TSLA", "US"))

    def test_asset_resolver_loads_external_alias_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            alias_file = Path(tmp) / "aliases.json"
            alias_file.write_text(
                '[{"ticker":"300750.SZ","market":"CN","canonical_name":"CATL","aliases":["宁德时代","CATL"]}]',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"FINAGENT_ASSET_ALIAS_FILE": str(alias_file)}):
                clear_asset_alias_cache()
                self.assertEqual(resolve_asset_alias("\u5b81\u5fb7\u65f6\u4ee3\u80a1\u4ef7"), ("300750.SZ", "CN"))
            clear_asset_alias_cache()

    def test_report_analysis_resolves_common_asset_names_without_demo_fallback(self) -> None:
        self.assertIsNone(ReportAgent._extract_ticker("SEC 10-K risk factors"))
        self.assertEqual(ReportAgent._extract_ticker("NVDA 10-K risk factors"), "NVDA")
        self.assertEqual(ReportAgent._extract_ticker("\u82f1\u4f1f\u8fbe 10-K risk factors"), "NVDA")
        self.assertEqual(ReportAgent._extract_cn_ticker("\u8d35\u5dde\u8305\u53f0\u5e74\u62a5"), "600519.SH")
        self.assertEqual(ReportAgent._extract_ticker("Please analyze MSFT 10-K risk factors"), "MSFT")

    def test_portfolio_impact_route_lists_all_function_calls(self) -> None:
        state = UserProxy().plan(make_state(), "\u7f8e\u8054\u50a8\u52a0\u606f\u5bf9\u6211\u7684\u6301\u4ed3\u5f71\u54cd")
        tools = [tool for step in state.route_plan for tool in step.tools]

        self.assertEqual(state.current_intent, IntentType.PORTFOLIO_IMPACT)
        self.assertIn("macro_indicator", tools)
        self.assertIn("web_search", tools)
        self.assertIn("get_user_portfolio", tools)
        self.assertIn("get_historical_prices", tools)
        self.assertIn("calc_portfolio_metrics", tools)
        self.assertIn("run_scenario_analysis", tools)
        self.assertIn("analyze_impact", tools)


if __name__ == "__main__":
    unittest.main()
