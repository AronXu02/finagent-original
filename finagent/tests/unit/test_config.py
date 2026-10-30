from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from finagent.packages.common.config import LONG_TERM_GRAPH_STORE_BACKEND, Settings
from finagent.packages.llm import OpenAICompatibleChatClient, OllamaClient, build_llm_client


class ConfigTest(unittest.TestCase):
    def test_defaults_are_production_backends(self) -> None:
        settings = Settings()

        self.assertEqual(settings.runtime_mode, "production")
        self.assertEqual(settings.database_backend, "postgresql")
        self.assertEqual(settings.vector_store_backend, "milvus")
        self.assertEqual(settings.graph_store_backend, LONG_TERM_GRAPH_STORE_BACKEND)
        self.assertEqual(settings.orchestrator_backend, "langgraph")
        self.assertEqual(settings.rag_backend, "http")
        self.assertEqual(settings.llm_provider, "ollama")

    def test_market_data_provider_is_http(self) -> None:
        settings = Settings(runtime_mode="production")
        self.assertEqual(settings.market_data_provider, "http")
        self.assertEqual(settings.database_backend, "postgresql")
        self.assertEqual(settings.market_data_sources, ("postgres_snapshot", "sec_edgar"))

    def test_local_downgrade_backends_are_rejected_by_schema(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(runtime_mode="production", database_backend="sqlite")
        with self.assertRaises(ValidationError):
            Settings(runtime_mode="production", vector_store_backend="disabled")
        with self.assertRaises(ValidationError):
            Settings(runtime_mode="production", graph_store_backend="nebula")
        with self.assertRaises(ValidationError):
            Settings(runtime_mode="production", orchestrator_backend="local")

    def test_production_requires_all_external_dependencies(self) -> None:
        settings = Settings(runtime_mode="production", database_url="postgresql://example/db")

        with self.assertRaisesRegex(ValueError, "FINAGENT_MILVUS_URI"):
            settings.validate_production_dependencies()

    def test_production_requires_market_data_and_sec_configuration(self) -> None:
        settings = Settings(
            runtime_mode="production",
            database_url="postgresql://example/db",
            milvus_uri="http://milvus:19530",
            embedding_base_url="http://embeddings:8080",
            neo4j_uri="bolt://neo4j:7687",
            neo4j_user="neo4j",
            neo4j_password="secret",
            intent_classifier_base_url="http://intent:8080",
            temporal_target_host="temporal:7233",
            rag_base_url="http://rag:8080",
        )

        with self.assertRaisesRegex(ValueError, "FINAGENT_MARKET_DATA_BASE_URL"):
            settings.validate_production_dependencies()

    def test_production_dependency_validation_passes_when_configured_for_langgraph(self) -> None:
        settings = Settings(
            runtime_mode="production",
            database_url="postgresql://example/db",
            milvus_uri="http://milvus:19530",
            embedding_base_url="http://embeddings:8080",
            neo4j_uri="bolt://neo4j:7687",
            neo4j_user="neo4j",
            neo4j_password="secret",
            intent_classifier_base_url="http://intent:8080",
            rag_base_url="http://rag:8080",
            market_data_base_url="http://market-data:8080",
            sec_user_agent="FinAgent tests contact@example.com",
            llm_gateway_base_url="http://llm-gateway:8000",
        )

        settings.validate_production_dependencies()

    def test_temporal_backend_requires_temporal_target_host(self) -> None:
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
            rag_base_url="http://rag:8080",
            market_data_base_url="http://market-data:8080",
            sec_user_agent="FinAgent tests contact@example.com",
            llm_gateway_base_url="http://llm-gateway:8000",
        )

        with self.assertRaisesRegex(ValueError, "FINAGENT_TEMPORAL_TARGET_HOST"):
            settings.validate_production_dependencies()

    def test_unknown_market_data_provider_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(runtime_mode="production", market_data_provider="memory")

    def test_default_ollama_llm_uses_qwen_4b(self) -> None:
        settings = Settings(runtime_mode="production")
        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.ollama_model, "qwen3.5:4b")
        self.assertEqual(settings.llm_gateway_model, "qwen3.5:4b")

    def test_production_requires_llm_gateway_when_openai_compatible(self) -> None:
        settings = Settings(
            runtime_mode="production",
            llm_provider="openai_compatible",
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
        )

        with self.assertRaisesRegex(ValueError, "FINAGENT_LLM_GATEWAY_BASE_URL"):
            settings.validate_production_dependencies()

    def test_unknown_llm_provider_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(runtime_mode="production", llm_provider="openai")

    def test_build_llm_client_uses_openai_compatible_gateway(self) -> None:
        settings = Settings(
            runtime_mode="production",
            llm_provider="openai_compatible",
            llm_gateway_base_url="http://llm-gateway:8000/v1",
            llm_gateway_api_key="secret",
            llm_gateway_model="qwen-test",
        )

        client = build_llm_client(settings)

        self.assertIsInstance(client, OpenAICompatibleChatClient)
        self.assertEqual(client.provider, "openai_compatible")
        self.assertEqual(client.model, "qwen-test")

    def test_build_llm_client_can_still_use_ollama(self) -> None:
        settings = Settings(runtime_mode="production", llm_provider="ollama", ollama_model="qwen-ollama")

        client = build_llm_client(settings)

        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.provider, "ollama")
        self.assertEqual(client.model, "qwen-ollama")

    def test_ollama_base_url_uses_ollama_host(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_HOST": ":8000"}, clear=False):
            settings = Settings(runtime_mode="production")

        self.assertEqual(settings.ollama_base_url, "http://127.0.0.1:8000")


if __name__ == "__main__":
    unittest.main()
