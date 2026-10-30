from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from finagent.packages.common.config import Settings
from finagent.packages.common.errors import ExternalDependencyError, ToolTimeoutError
from finagent.packages.orchestrator.models import SourceRef
from finagent.packages.tools.registry import ToolContext
from finagent.packages.tools.schemas import ResearchSearchInput


class ResearchRetriever(Protocol):
    def search_research_reports(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]: ...

    def web_search(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]: ...


class ConfiguredHttpResearchRetriever:
    """Production RAG gateway for paragraph-level report/news retrieval."""

    def __init__(self, settings: Settings) -> None:
        if not settings.rag_base_url:
            raise ExternalDependencyError("研报/新闻 RAG 需要配置 FINAGENT_RAG_BASE_URL")
        self.base_url = settings.rag_base_url.rstrip("/")
        self.api_key = settings.rag_api_key
        self.timeout = settings.request_timeout_seconds
        self.index_name = settings.rag_index_name
        self.rerank_base_url = settings.rag_rerank_base_url
        self.rerank_model = settings.rag_rerank_model

    def search_research_reports(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]:
        return self._post_search("/v1/research/search", payload, context, kind="研报/新闻 RAG")

    def web_search(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]:
        return self._post_search("/v1/web/search", payload, context, kind="联网搜索")

    def _post_search(
        self,
        path: str,
        payload: ResearchSearchInput,
        context: ToolContext,
        *,
        kind: str,
    ) -> dict[str, Any]:
        body = {
            **payload.model_dump(mode="json"),
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "trace_id": context.trace_id,
            "index_name": self.index_name,
            "rerank": {
                "enabled": bool(self.rerank_base_url),
                "model": self.rerank_model,
            },
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_payload = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise ToolTimeoutError(f"{kind} 请求超时") from exc
        except HTTPError as exc:
            raise ExternalDependencyError(f"{kind} 返回 HTTP {exc.code}") from exc
        except URLError as exc:
            raise ExternalDependencyError(f"{kind} 不可用：{exc.reason}") from exc
        try:
            data = json.loads(response_payload)
        except json.JSONDecodeError as exc:
            raise ExternalDependencyError(f"{kind} 返回了无效 JSON") from exc
        if not isinstance(data, dict):
            raise ExternalDependencyError(f"{kind} 返回的 JSON 不是对象")
        return _normalize_rag_response(data, default_url=f"{self.base_url}{path}")


def build_research_retriever(settings: Settings) -> ResearchRetriever:
    return ConfiguredHttpResearchRetriever(settings)


def _normalize_rag_response(data: dict[str, Any], *, default_url: str) -> dict[str, Any]:
    normalized = dict(data)
    raw_sources = normalized.get("sources") or []
    sources = [
        item if isinstance(item, SourceRef) else SourceRef.model_validate(item)
        for item in raw_sources
    ]
    if not sources:
        snippets = normalized.get("snippets") or normalized.get("chunks") or []
        sources = [
            SourceRef(
                source_type="research_report",
                title=str(item.get("title", "Research RAG snippet")),
                url=item.get("url") or default_url,
                published_at=_parse_datetime(item.get("published_at")),
                fetched_at=datetime.now(UTC),
                freshness="snapshot",
                checksum=item.get("checksum"),
            )
            for item in snippets
            if isinstance(item, dict)
        ]
    normalized["sources"] = sources
    normalized["status"] = normalized.get("status", "success")
    return normalized


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
