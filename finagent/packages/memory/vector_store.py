from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from finagent.packages.common.config import Settings
from finagent.packages.common.errors import ExternalDependencyError
from finagent.packages.common.time import utc_now_iso


@dataclass(frozen=True)
class MemorySearchHit:
    memory_id: str
    memory_type: str
    content: str
    score: float
    importance: float
    decay_weight: float
    source: str | None
    is_core: bool
    market_tag: str | None = None
    created_at: datetime | None = None
    valid_to: datetime | None = None


class MemoryVectorStore(Protocol):
    def upsert_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        memory_type: str,
        content: str,
        importance: float,
        decay_weight: float,
        source: str | None,
        is_core: bool,
        market_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def search(self, *, tenant_id: str, user_id: str, query: str, limit: int, market_state: str | None = None) -> list[MemorySearchHit]: ...

    def delete_memory(self, *, tenant_id: str, user_id: str, memory_id: str) -> None: ...


class HttpEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.embedding_base_url:
            raise ExternalDependencyError("向量记忆需要配置 FINAGENT_EMBEDDING_BASE_URL")
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.api_key = settings.embedding_api_key
        self.model = settings.embedding_model
        self.timeout = getattr(settings, "embedding_timeout_seconds", settings.request_timeout_seconds)
        self.initial_timeout = max(self.timeout, 60.0)
        self.provider = self._infer_provider()
        self._warmed = False

    def embed(self, text: str) -> list[float]:
        timeout = self.initial_timeout if not self._warmed else self.timeout
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url, body = self._request_payload(text)
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read().decode("utf-8")
                break
            except TimeoutError as exc:
                last_error = exc
                timeout = max(timeout * 2, self.initial_timeout)
                continue
            except HTTPError as exc:
                raise ExternalDependencyError(f"Embedding 服务返回 HTTP {exc.code}") from exc
            except URLError as exc:
                raise ExternalDependencyError(f"Embedding 服务不可用：{exc.reason}") from exc
        else:
            raise ExternalDependencyError("Embedding 服务请求超时，模型可能处于冷启动状态。") from last_error
        try:
            data = json.loads(payload)
            embedding = self._parse_embedding(data)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ExternalDependencyError("Embedding 服务返回了无效响应") from exc
        if not isinstance(embedding, list) or not all(isinstance(item, int | float) for item in embedding):
            raise ExternalDependencyError("Embedding 服务返回了非数值向量")
        self._warmed = True
        return [float(item) for item in embedding]

    def _request_payload(self, text: str) -> tuple[str, dict[str, Any]]:
        if self.provider == "ollama":
            return f"{self.base_url}/api/embeddings", {"model": self.model, "prompt": text}
        return f"{self.base_url}/v1/embeddings", {"model": self.model, "input": text}

    def _parse_embedding(self, data: dict[str, Any]) -> Any:
        if self.provider == "ollama":
            if "embedding" in data:
                return data["embedding"]
            return data["embeddings"][0]
        return data["data"][0]["embedding"]

    def _infer_provider(self) -> str:
        host = urlparse(self.base_url).hostname or ""
        if host == "ollama" or host.endswith(".ollama") or urlparse(self.base_url).port == 11434:
            return "ollama"
        return "openai_compatible"


class MilvusMemoryVectorStore:
    def __init__(self, settings: Settings, embedding_client: HttpEmbeddingClient | None = None) -> None:
        if not settings.milvus_uri:
            raise ExternalDependencyError("Milvus 记忆存储需要配置 FINAGENT_MILVUS_URI")
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "Milvus 记忆存储需要安装 pymilvus。请使用 production extra 安装 finagent。"
            ) from exc
        self.collection = settings.milvus_collection
        self.embedding_client = embedding_client or HttpEmbeddingClient(settings)
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
        self.dimension = settings.embedding_dimension
        if not self.client.has_collection(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.dimension,
                metric_type="COSINE",
                primary_field_name="memory_id",
                id_type="string",
                max_length=128,
            )

    def upsert_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        memory_type: str,
        content: str,
        importance: float,
        decay_weight: float,
        source: str | None,
        is_core: bool,
        market_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vector = self.embedding_client.embed(content)
        if len(vector) != self.dimension:
            raise ExternalDependencyError(
                f"Embedding 向量维度不匹配：期望 {self.dimension}，实际 {len(vector)}"
            )
        self.client.upsert(
            collection_name=self.collection,
            data=[
                {
                    "memory_id": memory_id,
                    "vector": vector,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "memory_type": memory_type,
                    "content": content,
                    "importance": float(importance),
                    "decay_weight": float(decay_weight),
                    "source": source or "",
                    "is_core": bool(is_core),
                    "market_tag": market_tag or "",
                    "metadata_json": json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True),
                    "valid_from": utc_now_iso(),
                    "deleted_at": "",
                }
            ],
        )

    def search(self, *, tenant_id: str, user_id: str, query: str, limit: int, market_state: str | None = None) -> list[MemorySearchHit]:
        vector = self.embedding_client.embed(query)
        safe_tenant = tenant_id.replace('"', '""')
        safe_user = user_id.replace('"', '""')
        filter_parts = [
            f'tenant_id == "{safe_tenant}"',
            f'user_id == "{safe_user}"',
            'deleted_at == ""',
        ]
        if market_state:
            safe_state = market_state.replace('"', '""')
            filter_parts.append(f'(market_tag == "{safe_state}" or market_tag == "")')
        filter_expr = " and ".join(filter_parts)
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            filter=filter_expr,
            limit=limit,
            output_fields=[
                "memory_id",
                "memory_type",
                "content",
                "importance",
                "decay_weight",
                "source",
                "is_core",
                "market_tag",
                "valid_from",
            ],
        )
        hits = results[0] if results else []
        return [_hit_from_milvus(item) for item in hits]

    def delete_memory(self, *, tenant_id: str, user_id: str, memory_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            filter=f'memory_id == "{memory_id}" and tenant_id == "{tenant_id}" and user_id == "{user_id}"',
        )


def build_memory_vector_store(settings: Settings) -> MemoryVectorStore:
    return MilvusMemoryVectorStore(settings)


def _hit_from_milvus(item: Any) -> MemorySearchHit:
    entity = item.get("entity", item) if isinstance(item, dict) else getattr(item, "entity", {})
    score = item.get("distance", item.get("score", 0.0)) if isinstance(item, dict) else getattr(item, "distance", 0.0)
    if not isinstance(entity, dict):
        entity = dict(entity)
    created_at = _parse_datetime(entity.get("valid_from"))
    market_tag = entity.get("market_tag")
    return MemorySearchHit(
        memory_id=str(entity.get("memory_id")),
        memory_type=str(entity.get("memory_type", "preference")),
        content=str(entity.get("content", "")),
        score=float(score),
        importance=float(entity.get("importance", 0.5)),
        decay_weight=float(entity.get("decay_weight", score or 1.0)),
        source=entity.get("source") or None,
        is_core=bool(entity.get("is_core", False)),
        market_tag=market_tag if market_tag else None,
        created_at=created_at,
    )


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
