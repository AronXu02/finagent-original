from __future__ import annotations

from functools import lru_cache
import os
from typing import Literal

from pydantic import BaseModel, Field


LONG_TERM_GRAPH_STORE_BACKEND = "neo4j"


def env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw_value = os.getenv(name, default)
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def default_ollama_base_url() -> str:
    configured = os.getenv("FINAGENT_OLLAMA_BASE_URL")
    if configured:
        return configured.rstrip("/")
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if host.startswith(":"):
        return f"http://127.0.0.1{host}"
    if "://" not in host:
        return f"http://{host}"
    return host.rstrip("/")


class Settings(BaseModel):
    app_name: str = "finagent"
    env: str = Field(default_factory=lambda: os.getenv("FINAGENT_ENV", "prod"))
    runtime_mode: Literal["production", "test"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_RUNTIME_MODE", "production")
    )
    database_backend: Literal["postgresql", "sqlite"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_DATABASE_BACKEND", "postgresql")
    )
    database_url: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_DATABASE_URL"))
    database_connect_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_DATABASE_CONNECT_TIMEOUT_SECONDS", "10"))
    )
    database_pool_min_size: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_DATABASE_POOL_MIN_SIZE", "1"))
    )
    database_pool_max_size: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_DATABASE_POOL_MAX_SIZE", "10"))
    )
    vector_store_backend: Literal["milvus"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_VECTOR_STORE_BACKEND", "milvus")
    )
    milvus_uri: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_MILVUS_URI"))
    milvus_token: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_MILVUS_TOKEN"))
    milvus_collection: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_MILVUS_COLLECTION", "user_memory_vectors")
    )
    embedding_base_url: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_EMBEDDING_BASE_URL"))
    embedding_api_key: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_EMBEDDING_API_KEY"))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dimension: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_EMBEDDING_DIMENSION", "1536"))
    )
    embedding_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_EMBEDDING_TIMEOUT_SECONDS", "60"))
    )
    graph_store_backend: Literal["neo4j"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_GRAPH_STORE_BACKEND", LONG_TERM_GRAPH_STORE_BACKEND)
    )
    neo4j_uri: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_NEO4J_URI"))
    neo4j_user: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_NEO4J_USER"))
    neo4j_password: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_NEO4J_PASSWORD"))
    neo4j_database: str = Field(default_factory=lambda: os.getenv("FINAGENT_NEO4J_DATABASE", "neo4j"))
    neo4j_narrative_index: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_NEO4J_NARRATIVE_INDEX", "market_narrative_fulltext")
    )
    intent_classifier_backend: Literal["http_bert", "rules"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_INTENT_CLASSIFIER_BACKEND", "http_bert")
    )
    intent_classifier_base_url: str | None = Field(
        default_factory=lambda: os.getenv("FINAGENT_INTENT_CLASSIFIER_BASE_URL")
    )
    intent_classifier_api_key: str | None = Field(
        default_factory=lambda: os.getenv("FINAGENT_INTENT_CLASSIFIER_API_KEY")
    )
    intent_classifier_model: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_INTENT_CLASSIFIER_MODEL", "finbert-intent-lite")
    )
    intent_classifier_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_INTENT_CLASSIFIER_TIMEOUT_SECONDS", "3"))
    )
    orchestrator_backend: Literal["langgraph", "temporal"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_ORCHESTRATOR_BACKEND", "langgraph")
    )
    temporal_target_host: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_TEMPORAL_TARGET_HOST"))
    temporal_namespace: str = Field(default_factory=lambda: os.getenv("FINAGENT_TEMPORAL_NAMESPACE", "default"))
    temporal_task_queue: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_TEMPORAL_TASK_QUEUE", "finagent-orchestrator")
    )
    temporal_workflow_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_TEMPORAL_WORKFLOW_TIMEOUT_SECONDS", "300"))
    )
    rag_backend: Literal["http"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_RAG_BACKEND", "http")
    )
    rag_base_url: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_RAG_BASE_URL"))
    rag_api_key: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_RAG_API_KEY"))
    rag_index_name: str = Field(default_factory=lambda: os.getenv("FINAGENT_RAG_INDEX_NAME", "research_snippets"))
    rag_rerank_base_url: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_RAG_RERANK_BASE_URL"))
    rag_rerank_api_key: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_RAG_RERANK_API_KEY"))
    rag_rerank_model: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_RAG_RERANK_MODEL", "bge-reranker-large")
    )
    memory_rerank_base_url: str | None = Field(
        default_factory=lambda: os.getenv("FINAGENT_MEMORY_RERANK_BASE_URL") or os.getenv("FINAGENT_RAG_RERANK_BASE_URL")
    )
    memory_rerank_api_key: str | None = Field(
        default_factory=lambda: os.getenv("FINAGENT_MEMORY_RERANK_API_KEY") or os.getenv("FINAGENT_RAG_RERANK_API_KEY")
    )
    memory_rerank_model: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_MEMORY_RERANK_MODEL", "bge-reranker-large")
    )
    memory_rerank_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_MEMORY_RERANK_TIMEOUT_SECONDS", "3"))
    )
    memory_summary_model: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_MEMORY_SUMMARY_MODEL", "qwen3.5:4b")
    )
    memory_summary_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_MEMORY_SUMMARY_TIMEOUT_SECONDS", "15"))
    )
    default_tenant_id: str = "tenant_default"
    default_user_id: str = "user_default"
    token_budget_warning_threshold: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_TOKEN_BUDGET_WARNING_THRESHOLD", "6000"))
    )
    max_recent_messages: int = 10
    market_data_provider: Literal["http"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_MARKET_DATA_PROVIDER", "http")
    )
    market_data_base_url: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_MARKET_DATA_BASE_URL"))
    market_data_api_key: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_MARKET_DATA_API_KEY"))
    market_data_sources: tuple[str, ...] = Field(
        default_factory=lambda: env_csv("FINAGENT_MARKET_DATA_SOURCES", "postgres_snapshot,sec_edgar")
    )
    market_data_primary_source: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_MARKET_DATA_PRIMARY_SOURCE", "postgres_snapshot")
    )
    market_data_fallback_source: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_MARKET_DATA_FALLBACK_SOURCE", "postgres_snapshot")
    )
    market_data_cache_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_MARKET_DATA_CACHE_TTL_SECONDS", "30"))
    )
    tushare_token: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_TUSHARE_TOKEN"))
    akshare_base_url: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_AKSHARE_BASE_URL"))
    alpha_vantage_api_key: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_ALPHA_VANTAGE_API_KEY"))
    polygon_api_key: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_POLYGON_API_KEY"))
    finnhub_api_key: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_FINNHUB_API_KEY"))
    yahoo_finance_base_url: str | None = Field(
        default_factory=lambda: os.getenv("FINAGENT_YAHOO_FINANCE_BASE_URL")
    )
    sec_user_agent: str | None = Field(default_factory=lambda: os.getenv("FINAGENT_SEC_USER_AGENT"))
    llm_provider: Literal["openai_compatible", "ollama"] = Field(
        default_factory=lambda: os.getenv("FINAGENT_LLM_PROVIDER", "ollama")
    )
    llm_gateway_base_url: str | None = Field(
        default_factory=lambda: os.getenv("FINAGENT_LLM_GATEWAY_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )
    llm_gateway_api_key: str | None = Field(
        default_factory=lambda: os.getenv("FINAGENT_LLM_GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    llm_gateway_model: str = Field(
        default_factory=lambda: os.getenv("FINAGENT_LLM_GATEWAY_MODEL", "qwen3.5:4b")
    )
    llm_gateway_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_LLM_GATEWAY_TIMEOUT_SECONDS", "120"))
    )
    llm_gateway_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_LLM_GATEWAY_MAX_TOKENS", "768"))
    )
    llm_gateway_temperature: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_LLM_GATEWAY_TEMPERATURE", "0.2"))
    )
    llm_gateway_top_p: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_LLM_GATEWAY_TOP_P", "0.9"))
    )
    ollama_base_url: str = Field(default_factory=default_ollama_base_url)
    ollama_model: str = Field(default_factory=lambda: os.getenv("FINAGENT_OLLAMA_MODEL", "qwen3.5:4b"))
    ollama_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_OLLAMA_TIMEOUT_SECONDS", "120"))
    )
    ollama_num_predict: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_OLLAMA_NUM_PREDICT", "768"))
    )
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_REQUEST_TIMEOUT_SECONDS", "8"))
    )
    agent_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("FINAGENT_AGENT_TIMEOUT_SECONDS", "60"))
    )
    rate_limit_max_requests: int = Field(
        default_factory=lambda: int(os.getenv("FINAGENT_RATE_LIMIT_MAX_REQUESTS", "30"))
    )

    def validate_storage_configuration(self) -> None:
        if self.database_backend == "sqlite":
            return
        if not self.database_url:
            raise ValueError("使用 PostgreSQL 数据库存储时必须配置 FINAGENT_DATABASE_URL")

    def validate_production_dependencies(self) -> None:
        if self.runtime_mode != "production":
            return
        self.validate_storage_configuration()
        missing: list[str] = []
        if not self.milvus_uri:
            missing.append("FINAGENT_MILVUS_URI")
        if not self.embedding_base_url:
            missing.append("FINAGENT_EMBEDDING_BASE_URL")
        if not self.neo4j_uri:
            missing.append("FINAGENT_NEO4J_URI")
        if not self.neo4j_user:
            missing.append("FINAGENT_NEO4J_USER")
        if not self.neo4j_password:
            missing.append("FINAGENT_NEO4J_PASSWORD")
        if self.intent_classifier_backend == "http_bert" and not self.intent_classifier_base_url:
            missing.append("FINAGENT_INTENT_CLASSIFIER_BASE_URL")
        if self.orchestrator_backend == "temporal" and not self.temporal_target_host:
            missing.append("FINAGENT_TEMPORAL_TARGET_HOST")
        if not self.rag_base_url:
            missing.append("FINAGENT_RAG_BASE_URL")
        if not self.market_data_base_url:
            missing.append("FINAGENT_MARKET_DATA_BASE_URL")
        if not self.sec_user_agent:
            missing.append("FINAGENT_SEC_USER_AGENT")
        if not self.market_data_sources:
            missing.append("FINAGENT_MARKET_DATA_SOURCES")
        if self.llm_provider == "openai_compatible" and not self.llm_gateway_base_url:
            missing.append("FINAGENT_LLM_GATEWAY_BASE_URL")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"生产运行缺少必要依赖配置：{joined}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
