from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from finagent.apps.api.main import create_app
from finagent.packages.common.config import Settings
from finagent.packages.common.sqlite_store import SQLiteStore
from finagent.packages.llm import OpenAICompatibleChatClient, OllamaClient
from finagent.devtools import EmptyMarketGraphStore, InMemoryMarketDataProvider, InMemoryMemoryVectorStore


class LocalPreviewLLM:
    provider = "test_stub"
    model = "local-preview-test-stub"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return (
            "本地预览链路已完成记忆检索、意图路由、工具结果、候选回复生成和合规审核。"
            "当前预览使用测试 Provider；生产数据需要接入 PostgreSQL、Milvus、Neo4j、"
            "Temporal、RAG、行情网关和 LLM 后端。"
        )


def build_app(
    db_path: Path,
    *,
    settings: Settings | None = None,
    llm_client: LocalPreviewLLM | OllamaClient | OpenAICompatibleChatClient | None = None,
):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(db_path)
    market_data = InMemoryMarketDataProvider()
    return create_app(
        settings or Settings(runtime_mode="test"),
        store,
        close_store_on_shutdown=True,
        memory_vector_store=InMemoryMemoryVectorStore(),
        graph_store=EmptyMarketGraphStore(),
        research_retriever=market_data,
        market_data_provider=market_data,
        temporal_gateway=None,
        llm_client=llm_client or LocalPreviewLLM(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行本地预览版 FinAgent API。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--db", default="finagent/data/local-preview.sqlite3")
    parser.add_argument("--llm", choices=("ollama", "openai-compatible", "local"), default="ollama")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default="qwen3.5:4b")
    args = parser.parse_args()
    uvicorn.run(
        build_app(Path(args.db), settings=_build_settings(args), llm_client=_build_llm_client(args)),
        host=args.host,
        port=args.port,
        log_level="info",
    )


def _build_settings(args: argparse.Namespace) -> Settings:
    if args.llm == "ollama":
        return Settings(
            runtime_mode="test",
            llm_provider="ollama",
            ollama_base_url=args.llm_base_url or "http://127.0.0.1:11434",
            ollama_model=args.llm_model,
        )
    if args.llm == "openai-compatible":
        return Settings(
            runtime_mode="test",
            llm_provider="openai_compatible",
            llm_gateway_base_url=args.llm_base_url or "http://127.0.0.1:8000",
            llm_gateway_model=args.llm_model,
        )
    return Settings(runtime_mode="test")


def _build_llm_client(args: argparse.Namespace):
    if args.llm == "local":
        return LocalPreviewLLM()
    if args.llm == "ollama":
        return OllamaClient(
            base_url=args.llm_base_url or "http://127.0.0.1:11434",
            model=args.llm_model,
        )
    return OpenAICompatibleChatClient(
        base_url=args.llm_base_url or "http://127.0.0.1:8000",
        api_key=args.llm_api_key,
        model=args.llm_model,
    )


if __name__ == "__main__":
    main()
