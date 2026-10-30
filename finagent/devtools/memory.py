from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from finagent.packages.memory.graph_store import MarketNarrative
from finagent.packages.memory.vector_store import MemorySearchHit


class InMemoryMemoryVectorStore:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

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
        self.items = [item for item in self.items if item["memory_id"] != memory_id]
        self.items.append(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "memory_id": memory_id,
                "memory_type": memory_type,
                "content": content,
                "importance": importance,
                "decay_weight": decay_weight,
                "source": source,
                "is_core": is_core,
                "market_tag": market_tag or "",
                "created_at": datetime.now(UTC),
            }
        )

    def search(self, *, tenant_id: str, user_id: str, query: str, limit: int, market_state: str | None = None) -> list[MemorySearchHit]:
        rows = [
            item
            for item in self.items
            if item["tenant_id"] == tenant_id and item["user_id"] == user_id
        ]
        if market_state:
            rows = [
                item for item in rows
                if not item.get("market_tag") or item["market_tag"] == market_state
            ]
        rows.sort(key=lambda item: (item["is_core"], item["decay_weight"], item["importance"]), reverse=True)
        return [
            MemorySearchHit(
                memory_id=item["memory_id"],
                memory_type=item["memory_type"],
                content=item["content"],
                score=0.0,
                importance=item["importance"],
                decay_weight=item["decay_weight"],
                source=item["source"],
                is_core=item["is_core"],
                market_tag=item.get("market_tag") or None,
                created_at=item["created_at"],
            )
            for item in rows[:limit]
        ]

    def delete_memory(self, *, tenant_id: str, user_id: str, memory_id: str) -> None:
        self.items = [
            item
            for item in self.items
            if not (
                item["tenant_id"] == tenant_id
                and item["user_id"] == user_id
                and item["memory_id"] == memory_id
            )
        ]


class EmptyMarketGraphStore:
    def find_related_narratives(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int,
    ) -> list[MarketNarrative]:
        return []
