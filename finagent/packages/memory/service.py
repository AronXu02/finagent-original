from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from finagent.packages.common.store import Store
from finagent.packages.memory.graph_store import MarketGraphStore
from finagent.packages.memory.vector_store import MemorySearchHit, MemoryVectorStore
from finagent.packages.orchestrator.models import MemoryContext, MemoryItemView, Message


MAX_DRAWDOWN_RE = re.compile(r"(?:max(?:imum)? drawdown|drawdown|\u56de\u64a4)[^\d]*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)
NO_ST_RE = re.compile(r"(\u4e0d\u4e70|\u4e0d\u78b0|\u907f\u514d).{0,8}ST", re.IGNORECASE)
ASSET_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{1,5}|\d{5,6}(?:\.(?:SH|SZ|HK))?)(?![A-Za-z0-9])")
ALLOW_ST_RE = re.compile(r"(?:\u53ef\u4ee5|\u80fd|\u5141\u8bb8|allow|buy).{0,8}ST", re.IGNORECASE)


class ShortTermSummarizer(Protocol):
    model_name: str

    def summarize(self, messages: list[Message], *, max_length: int = 500) -> str: ...

    def extract_constraints(self, text: str) -> list[dict[str, Any]]: ...


class MemoryReranker(Protocol):
    model_name: str

    def rerank(self, query: str, items: list[MemoryItemView], *, limit: int) -> list[MemoryItemView]: ...


class Qwen7BShortTermSummarizer:
    """Regex-based fallback summarizer.  Prefer ``OllamaShortTermSummarizer``."""

    model_name = "Qwen-7B-summary-compatible"

    def summarize(self, messages: list[Message], *, max_length: int = 500) -> str:
        user_messages = [message.content for message in messages if message.role == "user"]
        goals = user_messages[-3:]
        constraints = [
            text
            for text in user_messages
            if MAX_DRAWDOWN_RE.search(text) or NO_ST_RE.search(text) or ALLOW_ST_RE.search(text)
        ][-3:]
        assets = sorted({match.group(0) for text in user_messages for match in ASSET_TOKEN_RE.finditer(text)})[:8]
        parts = []
        if goals:
            parts.append("goals=" + " || ".join(goals))
        if assets:
            parts.append("assets=" + ",".join(assets))
        if constraints:
            parts.append("constraints=" + " || ".join(constraints))
        if not parts:
            parts.append(f"compressed_messages={len(messages)}")
        return " ; ".join(parts)[:max_length]

    def extract_constraints(self, text: str) -> list[dict[str, Any]]:
        return []


class OllamaShortTermSummarizer:
    """Real LLM-based summarizer that calls Ollama to compress older messages.

    Falls back to ``Qwen7BShortTermSummarizer`` when Ollama is unreachable.
    """

    model_name = "Qwen-7B-ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "qwen3.5:4b",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = Qwen7BShortTermSummarizer()

    def summarize(self, messages: list[Message], *, max_length: int = 500) -> str:
        if not messages:
            return ""
        prompt = self._build_prompt(messages)
        try:
            raw = self._call_ollama(prompt, max_length)
        except Exception:
            return self.fallback.summarize(messages, max_length=max_length)
        summary = raw.strip()
        if not summary or len(summary) < 10:
            return self.fallback.summarize(messages, max_length=max_length)
        return summary[:max_length]

    def _build_prompt(self, messages: list[Message]) -> str:
        lines = []
        for msg in messages:
            role = {"user": "用户", "assistant": "助手", "tool": "工具", "system": "系统"}.get(msg.role, msg.role)
            truncated = msg.content[:400]
            lines.append(f"[{role}] {truncated}")
        joined = "\n".join(lines)
        return (
            "请用一段中文摘要总结以下金融对话的历史要点，只提取关键信息：用户关心哪些资产、"
            "问了哪些问题、表达了什么偏好或约束（如风险承受能力、不买ST股等）。"
            "不要编造信息，不要给出投资建议，只做客观压缩。\n\n"
            f"{joined}\n\n摘要："
        )

    def _call_ollama(self, prompt: str, max_length: int) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "prompt": prompt,
            "options": {
                "temperature": 0.1,
                "top_p": 0.95,
                "num_predict": max(32, max_length),
            },
        }
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise _SummarizerError(f"Ollama summarizer failed: {exc}") from exc
        if data.get("error"):
            raise _SummarizerError(f"Ollama summarizer error: {data['error']}")
        return (data.get("response") or "").strip()

    def extract_constraints(self, text: str) -> list[dict[str, Any]]:
        if not text.strip():
            return []
        prompt = (
            "请从以下用户文本中提取金融偏好和约束信息，只提取明确表达的限制条件。"
            "返回 JSON 格式的列表，每项包含 field (约束类型) 和 value (约束值)。"
            "约束类型可包括：max_drawdown (最大回撤百分比), restricted_assets (限制资产类型如ST/stocks/crypto), "
            "risk_preference (风险偏好：conservative/moderate/aggressive), "
            "investment_horizon (投资期限), sector_preference (行业偏好), "
            "liquidity_requirement (流动性要求), benchmark (业绩基准), "
            "position_limit (单只持仓上限百分比)。"
            "不要编造信息，如果没有明确的约束则返回空列表 []。"
            "只输出 JSON 数组本身，不要包含任何其他文本、解释或 markdown 标记。\n\n"
            f"用户文本：{text[:800]}\n\n输出："
        )
        try:
            raw = self._call_ollama(prompt, 500)
            cleaned = raw.strip().lstrip("```json").rstrip("```").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict) and item.get("field")]
        except Exception:
            return []


def build_short_term_summarizer(
    *,
    base_url: str | None,
    model: str = "qwen3.5:4b",
    timeout_seconds: float = 15.0,
) -> ShortTermSummarizer:
    if base_url:
        return OllamaShortTermSummarizer(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    return Qwen7BShortTermSummarizer()


class BGEMemoryReranker:
    """Real BGE reranker that calls an HTTP rerank API endpoint.

    Falls back to ``HeuristicMemoryReranker`` when the HTTP request fails,
    so that the system remains operable during outages.
    """

    model_name = "BGE-reranker"

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "bge-reranker-large",
        api_key: str | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.fallback = HeuristicMemoryReranker()

    def rerank(self, query: str, items: list[MemoryItemView], *, limit: int) -> list[MemoryItemView]:
        if not items:
            return []
        documents = [item.content for item in items]
        try:
            reranked = self._call_rerank_api(query, documents, limit)
        except Exception:
            return self.fallback.rerank(query, items, limit=limit)
        ranked: list[MemoryItemView] = []
        seen = set()
        for entry in reranked:
            index = entry.get("index", -1)
            if index < 0 or index >= len(items) or index in seen:
                continue
            seen.add(index)
            ranked.append(items[index])
        missing = [items[i] for i in range(len(items)) if i not in seen]
        margin = limit - len(ranked)
        if margin > 0 and missing:
            deficit = self.fallback.rerank(query, missing, limit=margin)
            ranked.extend(deficit[:margin])
        return ranked[:limit]

    def _call_rerank_api(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        body = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/v1/rerank",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise _RerankError("BGE 重排序请求超时") from exc
        except HTTPError as exc:
            raise _RerankError(f"BGE 重排序返回 HTTP {exc.code}") from exc
        except URLError as exc:
            raise _RerankError(f"BGE 重排序不可用：{exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise _RerankError("BGE 重排序返回了无效 JSON") from exc
        if not isinstance(payload, dict):
            raise _RerankError("BGE 重排序返回的 JSON 不是对象")
        results = payload.get("results") or payload.get("data") or []
        if not isinstance(results, list):
            raise _RerankError(f"BGE 重排序返回的 results 格式异常：{type(results).__name__}")
        return results


class HeuristicMemoryReranker:
    """Lightweight fallback reranker using term overlap and static field sorting."""

    model_name = "heuristic-memory-reranker"

    def rerank(self, query: str, items: list[MemoryItemView], *, limit: int) -> list[MemoryItemView]:
        query_terms = _terms(query)
        scored = sorted(
            items,
            key=lambda item: (
                item.is_core,
                _term_overlap(query_terms, item.content),
                item.decay_weight,
                item.importance,
                item.created_at,
            ),
            reverse=True,
        )
        return scored[:limit]


class _RerankError(Exception):
    pass


class _SummarizerError(Exception):
    pass


def build_memory_reranker(
    *,
    base_url: str | None,
    model: str = "bge-reranker-large",
    api_key: str | None = None,
    timeout_seconds: float = 3.0,
) -> MemoryReranker:
    if base_url:
        return BGEMemoryReranker(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    return HeuristicMemoryReranker()


class MemoryService:
    def __init__(
        self,
        store: Store,
        max_recent_messages: int = 10,
        vector_store: MemoryVectorStore | None = None,
        graph_store: MarketGraphStore | None = None,
        summarizer: ShortTermSummarizer | None = None,
        reranker: MemoryReranker | None = None,
    ) -> None:
        if vector_store is None:
            raise ValueError("记忆服务需要配置可用的向量存储")
        if graph_store is None:
            raise ValueError("记忆服务需要配置可用的图存储")
        self.store = store
        self.max_recent_messages = max_recent_messages
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.summarizer = summarizer or Qwen7BShortTermSummarizer()
        self.reranker = reranker or HeuristicMemoryReranker()

    def compress_short_term(self, messages: list[Message]) -> str:
        if len(messages) <= self.max_recent_messages:
            return ""
        older = messages[: -self.max_recent_messages]
        if len(older) > 30:
            older = older[:30]
        return self.summarizer.summarize(older, max_length=500)

    def extract_and_store_preferences(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        text: str,
    ) -> list[MemoryItemView]:
        created: list[MemoryItemView] = []
        regex_fields: set[str] = set()
        drawdown = MAX_DRAWDOWN_RE.search(text)
        if drawdown:
            value = float(drawdown.group(1))
            item = self._store_preference_memory(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                content=f"Maximum drawdown tolerance is {value:g}%.",
                metadata={"field": "max_drawdown", "value": value},
            )
            if item is not None:
                created.append(item)
                regex_fields.add("max_drawdown")
        if NO_ST_RE.search(text):
            item = self._store_preference_memory(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                content="User avoids ST-designated stocks.",
                metadata={"field": "restricted_assets", "value": "ST"},
            )
            if item is not None:
                created.append(item)
                regex_fields.add("restricted_assets")
        if len(created) < 2:
            llm_constraints = self.summarizer.extract_constraints(text)
            for constraint in llm_constraints:
                field = constraint.get("field", "")
                if not field or field in regex_fields:
                    continue
                value = constraint.get("value", "")
                if not value:
                    continue
                content = f"User preference: {field}={value}."
                if self._active_memory_exists(tenant_id, user_id, content):
                    continue
                item = self._store_preference_memory(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    content=content,
                    metadata=constraint,
                )
                if item is not None:
                    created.append(item)
        return created

    @staticmethod
    def detect_market_state(text: str, short_term_summary: str | None = None) -> str:
        combined = f"{short_term_summary or ''} {text}".lower()
        if any(token in combined for token in ("rate hike", "rate_hike", "加息", "fomc", "tightening", "紧缩")):
            return "rate_hiking"
        if any(token in combined for token in ("rate cut", "rate_cut", "降息", "easing", "宽松", "qe")):
            return "rate_cutting"
        if any(token in combined for token in ("vix spike", "panic", "crash", "selloff", "暴跌", "崩盘", "恐慌")):
            return "high_volatility"
        if any(token in combined for token in ("inflation", "cpi", "ppi", "通胀", "物价")):
            return "inflation_concern"
        if any(token in combined for token in ("recession", "yield curve", "inverted", "衰退", "经济下行", "硬着陆")):
            return "recession_fear"
        if any(token in combined for token in ("bull market", "rally", "record high", "牛市", "新高", "突破")):
            return "bull_market"
        if any(token in combined for token in ("bear market", "correction", "downtrend", "熊市", "回调", "破位")):
            return "bear_market"
        if any(token in combined for token in ("unemployment", "jobless", "nonfarm", "就业", "失业", "非农")):
            return "labor_market_shift"
        return ""

    def retrieve_context(self, *, tenant_id: str, user_id: str, query: str, limit: int = 5, market_state: str | None = None) -> MemoryContext:
        hits = self.vector_store.search(tenant_id=tenant_id, user_id=user_id, query=query, limit=limit, market_state=market_state)
        memory_rows = self._memory_rows_by_id(tenant_id, user_id)
        scored = self._views_from_hits(hits, memory_rows)
        scored.extend(self._graph_narratives(tenant_id=tenant_id, query=query, limit=max(1, limit // 2)))
        reranked = self.reranker.rerank(query, scored, limit=limit)
        self.store.record_memory_access(
            tenant_id,
            user_id,
            [item.memory_id for item in reranked if not item.memory_id.startswith("graph:")],
        )
        return MemoryContext(top_memories=reranked, conflicts=self._detect_conflicts(scored))

    def delete_memory(self, tenant_id: str, user_id: str, memory_id: str) -> None:
        self.store.delete_memory(tenant_id, user_id, memory_id)
        self.vector_store.delete_memory(tenant_id=tenant_id, user_id=user_id, memory_id=memory_id)

    @staticmethod
    def calculate_weight(time_delta_days: int, access_count: int, is_core: bool) -> float:
        if is_core:
            return 1.0
        decay = math.exp(-0.05 * time_delta_days)
        return min(decay * (1 + math.log1p(access_count)), 1.0)

    @staticmethod
    def _detect_conflicts(items: list[MemoryItemView]) -> list[dict[str, Any]]:
        drawdowns = [item for item in items if "drawdown tolerance" in item.content.lower()]
        conflicts = []
        if len({item.content for item in drawdowns}) > 1:
            conflicts.append(
                {
                    "type": "constraint_conflict",
                    "field": "max_drawdown",
                    "memory_ids": [item.memory_id for item in drawdowns],
                }
            )
        restricted = [item for item in items if "avoids st" in item.content.lower()]
        allowed = [item for item in items if "allows st" in item.content.lower()]
        if restricted and allowed:
            conflicts.append(
                {
                    "type": "constraint_conflict",
                    "field": "restricted_assets",
                    "memory_ids": [item.memory_id for item in [*restricted, *allowed]],
                }
            )
        return conflicts

    def _active_memory_exists(self, tenant_id: str, user_id: str, content: str) -> bool:
        existing = {
            row["content"]
            for row in self.store.list_memories(tenant_id, user_id)
        }
        return content in existing

    def _store_preference_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        content: str,
        metadata: dict[str, Any],
        market_tag: str | None = None,
    ) -> MemoryItemView | None:
        if self._active_memory_exists(tenant_id, user_id, content):
            return None
        source = f"session:{session_id}"
        memory_id = self.store.insert_memory(
            tenant_id,
            user_id,
            memory_type="constraint",
            content=content,
            importance=0.95,
            decay_weight=1.0,
            source=source,
            is_core=True,
            market_tag=market_tag,
            metadata=metadata,
        )
        try:
            self._index_memory(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=memory_id,
                memory_type="constraint",
                content=content,
                importance=0.95,
                decay_weight=1.0,
                source=source,
                is_core=True,
                market_tag=market_tag,
                metadata=metadata,
            )
        except Exception:
            self.store.delete_memory(tenant_id, user_id, memory_id)
            raise
        return MemoryItemView(
            memory_id=memory_id,
            type="constraint",
            content=content,
            importance=0.95,
            decay_weight=1.0,
            source=source,
            is_core=True,
        )

    def _index_memory(
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
        self.vector_store.upsert_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            decay_weight=decay_weight,
            source=source,
            is_core=is_core,
            market_tag=market_tag,
            metadata=metadata,
        )

    def _memory_rows_by_id(self, tenant_id: str, user_id: str) -> dict[str, dict[str, Any]]:
        return {
            row["memory_id"]: row
            for row in self.store.list_memories(tenant_id, user_id)
        }

    def _views_from_hits(
        self,
        hits: list[MemorySearchHit],
        memory_rows: dict[str, dict[str, Any]],
    ) -> list[MemoryItemView]:
        now = datetime.now(timezone.utc)
        views = []
        for hit in hits:
            row = memory_rows.get(hit.memory_id)
            created_at = _hit_created_at(hit, row)
            is_core = bool((row or {}).get("is_core", hit.is_core))
            decay_weight = self.calculate_weight(_age_days(created_at, now), _access_count(row), is_core)
            views.append(
                MemoryItemView(
                    memory_id=hit.memory_id,
                    type=hit.memory_type,
                    content=hit.content,
                    importance=hit.importance,
                    decay_weight=round(min(decay_weight + max(hit.score, 0.0), 1.0), 6),
                    source=hit.source,
                    is_core=is_core,
                    market_tag=hit.market_tag,
                    created_at=created_at,
                    valid_to=hit.valid_to,
                )
            )
        return views

    def _graph_narratives(self, *, tenant_id: str, query: str, limit: int) -> list[MemoryItemView]:
        narratives = self.graph_store.find_related_narratives(tenant_id=tenant_id, query=query, limit=limit)
        return [
            MemoryItemView(
                memory_id=f"graph:{item.narrative_id}",
                type="market_narrative",
                content=item.content,
                importance=item.importance,
                decay_weight=round(min(item.score, 1.0), 6),
                source=item.source,
                is_core=False,
                created_at=item.updated_at or datetime.now().astimezone(),
            )
            for item in narratives
            if item.content
        ]


def _terms(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_.%]+", text) if len(token) >= 2}


def _term_overlap(query_terms: set[str], content: str) -> int:
    if not query_terms:
        return 0
    return len(query_terms & _terms(content))


def _hit_created_at(hit: MemorySearchHit, row: dict[str, Any] | None) -> datetime:
    return _parse_datetime((row or {}).get("valid_from")) or hit.created_at or datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_days(created_at: datetime, now: datetime) -> int:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max((now - created_at).days, 0)


def _access_count(row: dict[str, Any] | None) -> int:
    return int((row or {}).get("access_count") or 0)
