from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from finagent.packages.common.config import Settings
from finagent.packages.common.sqlite_store import SQLiteStore
from finagent.packages.orchestrator.models import ToolCall
from finagent.devtools import InMemoryMarketDataProvider
from finagent.packages.tools.executor import ToolExecutor, build_default_registry
from finagent.packages.tools.registry import ToolContext, ToolRegistry, ToolSpec
from finagent.packages.tools.schemas import TickerMarketInput


def build_test_registry(settings: Settings, store: SQLiteStore):
    market_data = InMemoryMarketDataProvider()
    return build_default_registry(
        settings,
        store,
        market_data_provider=market_data,
        research_retriever=market_data,
    )


class ToolExecutorTest(unittest.TestCase):
    def test_schema_error_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                executor = ToolExecutor(
                    build_test_registry(Settings(runtime_mode="test"), store),
                    store,
                )
                result = executor.execute(
                    ToolCall(tool_name="get_realtime_stock", arguments={"ticker": "AAPL;DROP", "market": "US"}),
                    ToolContext(tenant_id="tenant_a", user_id="user_a", task_id="task_a", trace_id="trace_a"),
                )

                self.assertEqual(result.status, "error")
                self.assertIn("工具参数校验失败", result.error or "")

    def test_invalid_provider_status_is_structured(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="bad_status_tool",
                description="Return an invalid status for executor hardening.",
                input_model=TickerMarketInput,
                handler=lambda payload, context: {"status": "partial", "sources": []},
            )
        )
        executor = ToolExecutor(registry)
        result = executor.execute(
            ToolCall(tool_name="bad_status_tool", arguments={"ticker": "AAPL", "market": "US"}),
            ToolContext(tenant_id="tenant_a", user_id="user_a", task_id="task_a", trace_id="trace_a"),
        )

        self.assertEqual(result.status, "error")
        self.assertIn("工具返回了无效状态", result.error or "")

    def test_quote_timeout_falls_back_to_delayed_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                executor = ToolExecutor(
                    build_test_registry(Settings(runtime_mode="test"), store),
                    store,
                )
                result = executor.execute(
                    ToolCall(tool_name="get_realtime_stock", arguments={"ticker": "TIMEOUT", "market": "US"}),
                    ToolContext(tenant_id="tenant_a", user_id="user_a", task_id="task_a", trace_id="trace_a"),
                )

                self.assertEqual(result.status, "success")
                self.assertEqual(result.data["freshness"], "delayed")
                self.assertEqual(result.data["fallback"]["fallback_tool"], "get_delayed_quote")
                self.assertIn("请求超时", result.data["fallback"]["reason"])

    def test_tool_call_arguments_are_repaired_before_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                executor = ToolExecutor(
                    build_test_registry(Settings(runtime_mode="test"), store),
                    store,
                )
                call = ToolCall(
                    tool_name="get_forex_rate",
                    arguments="{base: 'USD', quote: 'JPY',}",
                )
                result = executor.execute(
                    call,
                    ToolContext(tenant_id="tenant_a", user_id="user_a", task_id="task_a", trace_id="trace_a"),
                )

                self.assertEqual(call.arguments, {"base": "USD", "quote": "JPY"})
                self.assertEqual(result.status, "success")
                self.assertEqual(result.data["pair"], "USD/JPY")

    def test_documented_tools_are_registered_without_runtime_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                registry = build_test_registry(Settings(runtime_mode="test"), store)
                registry_names = set(registry.names())

                self.assertIn("macro_indicator", registry_names)
                self.assertIn("web_search", registry_names)
                self.assertIn("get_forex_rate", registry_names)
                self.assertIn("analyze_impact", registry_names)
                self.assertIn("calc_position_value", registry_names)
                self.assertNotIn("get_macro_indicator", registry_names)
                self.assertNotIn("search_research_reports", registry_names)

                executor = ToolExecutor(registry, store)
                result = executor.execute(
                    ToolCall(tool_name="get_forex_rate", arguments={"base": "USD", "quote": "JPY"}),
                    ToolContext(tenant_id="tenant_a", user_id="user_a", task_id="task_a", trace_id="trace_a"),
                )

                self.assertEqual(result.status, "success")
                self.assertEqual(result.data["pair"], "USD/JPY")

    def test_production_quote_requires_configured_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                settings = Settings(runtime_mode="production", market_data_provider="http")
                executor = ToolExecutor(
                    build_default_registry(
                        settings,
                        store,
                        research_retriever=InMemoryMarketDataProvider(),
                    ),
                    store,
                )
                result = executor.execute(
                    ToolCall(tool_name="get_realtime_stock", arguments={"ticker": "AAPL", "market": "US"}),
                    ToolContext(tenant_id="tenant_a", user_id="user_a", task_id="task_a", trace_id="trace_a"),
                )

                self.assertEqual(result.status, "error")
                self.assertIn("FINAGENT_MARKET_DATA_BASE_URL", result.error or "")

    def test_production_portfolio_reads_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                store.upsert_portfolio_snapshot(
                    "tenant_a",
                    "user_a",
                    account_id="default",
                    currency="USD",
                    positions=[
                        {
                            "ticker": "AAPL",
                            "market": "US",
                            "name": "Apple Inc.",
                            "quantity": 10,
                            "price": 200.0,
                            "cost_price": 150.0,
                            "sector": "Technology",
                        }
                    ],
                )
                settings = Settings(runtime_mode="production", market_data_provider="http")
                executor = ToolExecutor(
                    build_default_registry(
                        settings,
                        store,
                        research_retriever=InMemoryMarketDataProvider(),
                    ),
                    store,
                )
                result = executor.execute(
                    ToolCall(tool_name="get_user_portfolio", arguments={"account_scope": "default"}),
                    ToolContext(tenant_id="tenant_a", user_id="user_a", task_id="task_a", trace_id="trace_a"),
                )

                self.assertEqual(result.status, "success")
                self.assertEqual(result.data["total_value"], 2000.0)
                self.assertEqual(result.data["holdings"][0]["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main()
