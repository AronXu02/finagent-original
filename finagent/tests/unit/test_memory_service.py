from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finagent.packages.common.config import Settings
from finagent.packages.common.sqlite_store import SQLiteStore
from finagent.packages.memory.service import MemoryService
from finagent.packages.memory.vector_store import HttpEmbeddingClient
from finagent.packages.orchestrator.models import Message
from finagent.devtools import EmptyMarketGraphStore, InMemoryMemoryVectorStore


class FakeEmbeddingResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class MemoryServiceTest(unittest.TestCase):
    def test_core_principles_do_not_decay(self) -> None:
        self.assertEqual(MemoryService.calculate_weight(365, 0, True), 1.0)
        self.assertLess(MemoryService.calculate_weight(365, 0, False), 0.01)

    def test_extracts_drawdown_and_st_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                service = MemoryService(
                    store,
                    vector_store=InMemoryMemoryVectorStore(),
                    graph_store=EmptyMarketGraphStore(),
                )
                created = service.extract_and_store_preferences(
                    tenant_id="tenant_a",
                    user_id="user_a",
                    session_id="sess_a",
                    text="\u6700\u5927\u56de\u64a4\u4e0d\u80fd\u8d85\u8fc710%\uff0c\u4e0d\u4e70ST\u80a1",
                )

                self.assertEqual(len(created), 2)
                context = service.retrieve_context(
                    tenant_id="tenant_a",
                    user_id="user_a",
                    query="\u8fd9\u4e2a\u7ec4\u5408\u53ef\u4ee5\u4e70\u5417",
                )
                self.assertEqual(len(context.top_memories), 2)
                self.assertTrue(all(item.is_core for item in context.top_memories))

    def test_does_not_duplicate_identical_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                service = MemoryService(
                    store,
                    vector_store=InMemoryMemoryVectorStore(),
                    graph_store=EmptyMarketGraphStore(),
                )
                text = "\u6700\u5927\u56de\u64a4\u4e0d\u80fd\u8d85\u8fc78%\uff0c\u4e0d\u4e70ST\u80a1"

                first = service.extract_and_store_preferences(
                    tenant_id="tenant_a",
                    user_id="user_a",
                    session_id="sess_a",
                    text=text,
                )
                second = service.extract_and_store_preferences(
                    tenant_id="tenant_a",
                    user_id="user_a",
                    session_id="sess_b",
                    text=text,
                )
                context = service.retrieve_context(
                    tenant_id="tenant_a",
                    user_id="user_a",
                    query="\u7ec4\u5408\u98ce\u9669",
                )

                self.assertEqual(len(first), 2)
                self.assertEqual(second, [])
                self.assertEqual(len(context.top_memories), 2)

    def test_short_term_summary_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                service = MemoryService(
                    store,
                    max_recent_messages=2,
                    vector_store=InMemoryMemoryVectorStore(),
                    graph_store=EmptyMarketGraphStore(),
                )
                summary = service.compress_short_term(
                    [
                        Message(role="user", content="Analyze AAPL portfolio risk."),
                        Message(role="assistant", content="I will check the data."),
                        Message(role="user", content="My max drawdown tolerance is 8%."),
                        Message(role="assistant", content="Noted."),
                        Message(role="user", content="What about MSFT?"),
                    ]
                )

                self.assertIn("goals=", summary)
                self.assertIn("assets=AAPL", summary)
                self.assertIn("constraints=", summary)

    def test_default_memory_components_match_documented_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                service = MemoryService(
                    store,
                    vector_store=InMemoryMemoryVectorStore(),
                    graph_store=EmptyMarketGraphStore(),
                )

                self.assertIn("Qwen-7B", service.summarizer.model_name)
                self.assertIn("heuristic", service.reranker.model_name)

    def test_memory_service_accepts_bge_reranker(self) -> None:
        from finagent.packages.memory.service import BGEMemoryReranker

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                reranker = BGEMemoryReranker(base_url="http://localhost:8080")
                service = MemoryService(
                    store,
                    vector_store=InMemoryMemoryVectorStore(),
                    graph_store=EmptyMarketGraphStore(),
                    reranker=reranker,
                )
                self.assertIn("BGE", service.reranker.model_name)

    def test_retrieval_records_access_and_boosts_next_decay_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "test.sqlite3") as store:
                vector_store = InMemoryMemoryVectorStore()
                service = MemoryService(
                    store,
                    vector_store=vector_store,
                    graph_store=EmptyMarketGraphStore(),
                )
                memory_id = store.insert_memory(
                    "tenant_a",
                    "user_a",
                    memory_type="news",
                    content="User asked about semiconductor export controls.",
                    importance=0.5,
                    decay_weight=0.2,
                    source="session:sess_a",
                    is_core=False,
                )
                vector_store.upsert_memory(
                    tenant_id="tenant_a",
                    user_id="user_a",
                    memory_id=memory_id,
                    memory_type="news",
                    content="User asked about semiconductor export controls.",
                    importance=0.5,
                    decay_weight=0.2,
                    source="session:sess_a",
                    is_core=False,
                )
                old_timestamp = (datetime.now(UTC) - timedelta(days=100)).isoformat()
                store._conn.execute("UPDATE memory_items SET valid_from = ? WHERE memory_id = ?", (old_timestamp, memory_id))
                store._conn.commit()

                first = service.retrieve_context(tenant_id="tenant_a", user_id="user_a", query="export controls")
                second = service.retrieve_context(tenant_id="tenant_a", user_id="user_a", query="export controls")
                rows = store.list_memories("tenant_a", "user_a")

                self.assertEqual(rows[0]["access_count"], 2)
                self.assertGreater(second.top_memories[0].decay_weight, first.top_memories[0].decay_weight)

    def test_embedding_client_uses_ollama_native_endpoint(self) -> None:
        settings = Settings(
            runtime_mode="test",
            embedding_base_url="http://127.0.0.1:11434",
            embedding_model="nomic-embed-text",
        )
        with patch(
            "finagent.packages.memory.vector_store.urlopen",
            return_value=FakeEmbeddingResponse({"embedding": [1, 2, 3]}),
        ) as fake_urlopen:
            vector = HttpEmbeddingClient(settings).embed("portfolio risk")

        request = fake_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/embeddings")
        self.assertEqual(json.loads(request.data.decode("utf-8")), {"model": "nomic-embed-text", "prompt": "portfolio risk"})
        self.assertEqual(vector, [1.0, 2.0, 3.0])

    def test_embedding_client_keeps_openai_compatible_endpoint(self) -> None:
        settings = Settings(
            runtime_mode="test",
            embedding_base_url="http://embeddings:8080",
            embedding_model="text-embedding-3-small",
        )
        with patch(
            "finagent.packages.memory.vector_store.urlopen",
            return_value=FakeEmbeddingResponse({"data": [{"embedding": [0.1, 0.2]}]}),
        ) as fake_urlopen:
            vector = HttpEmbeddingClient(settings).embed("portfolio risk")

        request = fake_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://embeddings:8080/v1/embeddings")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {"model": "text-embedding-3-small", "input": "portfolio risk"},
        )
        self.assertEqual(vector, [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
