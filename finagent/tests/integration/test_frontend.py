from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from finagent.apps.api.main import create_app
from finagent.scripts.run_local_preview_api import build_app
from finagent.packages.common.config import Settings
from finagent.packages.common.sqlite_store import SQLiteStore
from finagent.devtools import EmptyMarketGraphStore, InMemoryMarketDataProvider, InMemoryMemoryVectorStore


def create_test_app(settings: Settings, store: SQLiteStore):
    market_data = InMemoryMarketDataProvider()
    return create_app(
        settings,
        store,
        memory_vector_store=InMemoryMemoryVectorStore(),
        graph_store=EmptyMarketGraphStore(),
        research_retriever=market_data,
        market_data_provider=market_data,
        temporal_gateway=None,
    )


class FrontendTest(unittest.TestCase):
    def test_frontend_route_serves_console(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                app = create_test_app(Settings(runtime_mode="test"), store)
                client = TestClient(app)
                response = client.get("/")

                self.assertEqual(response.status_code, 200)
                self.assertIn("FinAgent 多轮金融对话", response.text)
                self.assertIn("多轮对话", response.text)
                self.assertIn("发送", response.text)
                self.assertIn("step-toggle", response.text)
                self.assertIn("aria-expanded", response.text)
                self.assertIn("flex-direction: column;", response.text)
                self.assertIn("flex: 0 0 auto;", response.text)
                self.assertIn("body.hidden = !nextExpanded;", response.text)
                self.assertIn("function createTraceIo", response.text)
                self.assertIn("pre.textContent = asJson(value);", response.text)
                self.assertIn("/v1/chat/stream", response.text)
                self.assertIn("/v1/chat/history", response.text)
                self.assertIn('<select id="accountScope">', response.text)
                self.assertIn("/v1/portfolio/accounts", response.text)
                self.assertIn("loadAccountScopes", response.text)

    def test_chat_history_returns_persisted_session_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                app = create_test_app(Settings(runtime_mode="test"), store)
                store.ensure_conversation("tenant_default", "user_default", "sess_history")
                store.add_message("tenant_default", "sess_history", "user_default", "msg_1", "user", "第一轮问题")
                store.add_message("tenant_default", "sess_history", "user_default", "msg_2", "assistant", "第一轮回答")
                client = TestClient(app)

                response = client.get("/v1/chat/history", params={"session_id": "sess_history"})

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["session_id"], "sess_history")
                self.assertEqual([item["role"] for item in payload["messages"]], ["user", "assistant"])
                self.assertEqual(payload["messages"][0]["content"], "第一轮问题")

    def test_portfolio_accounts_returns_current_user_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                app = create_test_app(Settings(runtime_mode="test"), store)
                store.upsert_portfolio_snapshot(
                    "tenant_default",
                    "user_default",
                    account_id="default",
                    currency="USD",
                    positions=[
                        {
                            "ticker": "AAPL",
                            "market": "US",
                            "quantity": 1,
                            "price": 100,
                            "cost_price": 90,
                        }
                    ],
                )
                store.upsert_portfolio_snapshot(
                    "tenant_default",
                    "user_default",
                    account_id="growth",
                    currency="USD",
                    positions=[
                        {
                            "ticker": "NVDA",
                            "market": "US",
                            "quantity": 1,
                            "price": 120,
                            "cost_price": 100,
                        }
                    ],
                )
                store.upsert_portfolio_snapshot(
                    "tenant_default",
                    "other_user",
                    account_id="hidden",
                    currency="USD",
                    positions=[
                        {
                            "ticker": "MSFT",
                            "market": "US",
                            "quantity": 1,
                            "price": 80,
                            "cost_price": 70,
                        }
                    ],
                )
                client = TestClient(app)

                response = client.get(
                    "/v1/portfolio/accounts",
                    headers={"X-Tenant-ID": "tenant_default", "X-User-ID": "user_default"},
                )

                self.assertEqual(response.status_code, 200)
                account_ids = [item["account_id"] for item in response.json()["items"]]
                self.assertEqual(account_ids, ["default", "growth"])

    def test_create_app_settings_are_used_by_auth_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                settings = Settings(
                    runtime_mode="test",
                    default_tenant_id="tenant_custom",
                    default_user_id="user_custom",
                )
                app = create_test_app(settings, store)
                client = TestClient(app)
                response = client.post(
                    "/v1/memory/preferences",
                    json={
                        "session_id": "prefs",
                        "content": "\u6700\u5927\u56de\u64a4\u4e0d\u80fd\u8d85\u8fc78%",
                    },
                )

                self.assertEqual(response.status_code, 200)
                memories = store.list_memories("tenant_custom", "user_custom")
                self.assertEqual(len(memories), 1)
                self.assertEqual(memories[0]["content"], "Maximum drawdown tolerance is 8%.")

    def test_default_langgraph_backend_does_not_create_temporal_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                market_data = InMemoryMarketDataProvider()
                app = create_app(
                    Settings(runtime_mode="test"),
                    store,
                    memory_vector_store=InMemoryMemoryVectorStore(),
                    graph_store=EmptyMarketGraphStore(),
                    research_retriever=market_data,
                    market_data_provider=market_data,
                )
                client = TestClient(app)
                response = client.get("/ready")
                payload = response.json()

                self.assertIsNone(app.state.temporal_gateway)
                self.assertEqual(payload["orchestrator_backend"], "langgraph")
                self.assertNotIn("temporal_reachable", payload["dependencies"])
                self.assertIn("unready_dependencies", payload)
                self.assertLessEqual(set(payload["unready_dependencies"]), set(payload["dependencies"]))

    def test_local_preview_app_closes_owned_store_on_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = build_app(Path(tmp) / "preview.sqlite3")
            with TestClient(app) as client:
                response = client.get("/health")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(app.state.store._closed)

    def test_deep_ready_runs_dependency_smoke_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = build_app(Path(tmp) / "preview.sqlite3")
            with TestClient(app) as client:
                response = client.get("/ready/deep")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "ready")
            checks = {item["name"]: item for item in payload["checks"]}
            self.assertEqual(checks["postgresql"]["status"], "ok")
            self.assertEqual(checks["milvus"]["status"], "ok")
            self.assertEqual(checks["neo4j"]["status"], "ok")
            self.assertEqual(checks["rag"]["status"], "ok")
            self.assertEqual(checks["market_data"]["status"], "ok")
            self.assertEqual(checks["llm"]["status"], "ok")

    def test_temporal_backend_requires_gateway_in_production(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                market_data = InMemoryMarketDataProvider()
                settings = Settings(
                    runtime_mode="production",
                    orchestrator_backend="temporal",
                    database_url="postgresql://example/db",
                    milvus_uri="http://milvus:19530",
                    embedding_base_url="http://embeddings:8080",
                    neo4j_uri="bolt://neo4j:7687",
                    neo4j_user="neo4j",
                    neo4j_password="secret",
                    intent_classifier_base_url="http://intent:8080",
                    temporal_target_host="temporal:7233",
                    rag_base_url="http://rag:8080",
                    market_data_base_url="http://market-data:8080",
                    sec_user_agent="FinAgent tests contact@example.com",
                    llm_gateway_base_url="http://llm-gateway:8000",
                )

                with self.assertRaisesRegex(ValueError, "Temporal 网关"):
                    create_app(
                        settings,
                        store,
                        memory_vector_store=InMemoryMemoryVectorStore(),
                        graph_store=EmptyMarketGraphStore(),
                        research_retriever=market_data,
                        market_data_provider=market_data,
                        temporal_gateway=None,
                    )

    def test_chat_stream_returns_persisted_error_event_when_llm_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                app = create_test_app(Settings(runtime_mode="test"), store)
                client = TestClient(app)
                response = client.post(
                    "/v1/chat/stream",
                    json={
                        "session_id": "sess_error",
                        "message": "Please analyze my portfolio risk.",
                        "account_scope": "default",
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertIn("event: error", response.text)
                self.assertIn("external_dependency_error", response.text)
                self.assertIn("\"stage\":\"candidate_response\"", response.text)


if __name__ == "__main__":
    unittest.main()
