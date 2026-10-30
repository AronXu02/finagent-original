from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from finagent.packages.common.config import LONG_TERM_GRAPH_STORE_BACKEND, Settings
from finagent.packages.common.errors import ExternalDependencyError


@dataclass(frozen=True)
class MarketNarrative:
    narrative_id: str
    content: str
    importance: float
    score: float
    source: str | None = None
    updated_at: datetime | None = None


class MarketGraphStore(Protocol):
    def find_related_narratives(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int,
    ) -> list[MarketNarrative]: ...


class Neo4jMarketGraphStore:
    def __init__(self, settings: Settings) -> None:
        if not settings.neo4j_uri or not settings.neo4j_user or not settings.neo4j_password:
            raise ExternalDependencyError(
                "图记忆需要配置 FINAGENT_NEO4J_URI、FINAGENT_NEO4J_USER 和 FINAGENT_NEO4J_PASSWORD"
            )
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "Neo4j 图记忆需要安装 neo4j。请使用 production extra 安装 finagent。"
            ) from exc
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self.database = settings.neo4j_database
        self.index_name = settings.neo4j_narrative_index

    def find_related_narratives(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int,
    ) -> list[MarketNarrative]:
        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $search_query, {limit: $limit})
        YIELD node, score
        WHERE coalesce(node.tenant_id, 'global') IN ['global', $tenant_id]
        RETURN coalesce(node.narrative_id, elementId(node)) AS narrative_id,
               coalesce(node.content, node.summary, node.name) AS content,
               coalesce(node.importance, 0.5) AS importance,
               coalesce(node.source, node.url) AS source,
               coalesce(node.updated_at, node.created_at) AS updated_at,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:
                rows = session.run(
                    cypher,
                    index_name=self.index_name,
                    search_query=query,
                    limit=limit,
                    tenant_id=tenant_id,
                )
                return [_narrative_from_row(row.data()) for row in rows]
        except Exception as exc:
            raise ExternalDependencyError(f"Neo4j 市场叙事检索失败：{exc}") from exc


def build_market_graph_store(settings: Settings) -> MarketGraphStore:
    if settings.graph_store_backend != LONG_TERM_GRAPH_STORE_BACKEND:
        raise ExternalDependencyError("长期市场叙事记忆需要使用 Neo4j 图存储")
    return Neo4jMarketGraphStore(settings)


def _narrative_from_row(row: dict[str, Any]) -> MarketNarrative:
    return MarketNarrative(
        narrative_id=str(row["narrative_id"]),
        content=str(row.get("content") or ""),
        importance=float(row.get("importance", 0.5)),
        score=float(row.get("score", 0.0)),
        source=row.get("source"),
        updated_at=_parse_datetime(row.get("updated_at")),
    )


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
