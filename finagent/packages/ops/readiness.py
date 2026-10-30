from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from finagent.packages.common.context import AuthContext
from finagent.packages.orchestrator.models import ToolCall
from finagent.packages.orchestrator.pipeline import FinAgentOrchestrator
from finagent.packages.tools.registry import ToolContext


@dataclass(frozen=True)
class DependencyCheck:
    name: str
    status: str
    detail: str
    latency_ms: int = 0

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
        }


def run_deep_readiness_checks(
    *,
    orchestrator: FinAgentOrchestrator,
    auth: AuthContext,
) -> dict[str, Any]:
    context = ToolContext(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        task_id="readiness_probe",
        trace_id="readiness_probe",
        account_scope="default",
    )
    checks = [
        _check("postgresql", lambda: _check_store(orchestrator, auth)),
        _check("milvus", lambda: _check_vector_store(orchestrator, auth)),
        _check("neo4j", lambda: _check_graph_store(orchestrator, auth)),
        _check("rag", lambda: _check_rag(orchestrator, context)),
        _check("market_data", lambda: _check_market_data(orchestrator, context)),
        _check("llm", lambda: _check_llm(orchestrator)),
    ]
    payload = [item.model_dump() for item in checks]
    failed = [item.name for item in checks if item.status == "error"]
    skipped = [item.name for item in checks if item.status == "skipped"]
    return {
        "status": "ready" if not failed else "not_ready",
        "checks": payload,
        "failed_dependencies": failed,
        "skipped_dependencies": skipped,
    }


def _check(name: str, callback: Callable[[], str]) -> DependencyCheck:
    import time

    started = time.perf_counter()
    try:
        detail = callback()
    except NotImplementedError as exc:
        status = "skipped"
        detail = str(exc)
    except Exception as exc:
        status = "error"
        detail = str(exc)
    else:
        status = "ok"
    latency_ms = int((time.perf_counter() - started) * 1000)
    return DependencyCheck(name=name, status=status, detail=detail, latency_ms=latency_ms)


def _check_store(orchestrator: FinAgentOrchestrator, auth: AuthContext) -> str:
    orchestrator.store.list_memories(auth.tenant_id, auth.user_id)
    return "relational store read succeeded"


def _check_vector_store(orchestrator: FinAgentOrchestrator, auth: AuthContext) -> str:
    hits = orchestrator.memory.vector_store.search(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        query="finagent readiness probe",
        limit=1,
    )
    return f"vector search succeeded; hits={len(hits)}"


def _check_graph_store(orchestrator: FinAgentOrchestrator, auth: AuthContext) -> str:
    hits = orchestrator.memory.graph_store.find_related_narratives(
        tenant_id=auth.tenant_id,
        query="finagent readiness probe",
        limit=1,
    )
    return f"graph narrative query succeeded; hits={len(hits)}"


def _check_rag(orchestrator: FinAgentOrchestrator, context: ToolContext) -> str:
    result = orchestrator.executor.execute(
        ToolCall(tool_name="web_search", arguments={"query": "finagent readiness probe", "top_k": 1}),
        context,
    )
    if result.status != "success":
        raise RuntimeError(result.error or "RAG readiness tool returned error")
    return "RAG web_search succeeded"


def _check_market_data(orchestrator: FinAgentOrchestrator, context: ToolContext) -> str:
    result = orchestrator.executor.execute(
        ToolCall(
            tool_name="macro_indicator",
            arguments={"indicator": "fomc_rate_decision", "region": "US", "latest": True},
        ),
        context,
    )
    if result.status != "success":
        raise RuntimeError(result.error or "market data readiness tool returned error")
    return "market data macro_indicator succeeded"


def _check_llm(orchestrator: FinAgentOrchestrator) -> str:
    llm_client = orchestrator.candidate_builder.llm_client
    if llm_client is None:
        raise NotImplementedError("LLM client is not configured for this runtime")
    llm_client.generate(
        system_prompt="You are a readiness probe. Return a short plain response.",
        user_prompt="Return ok.",
    )
    return "LLM generation succeeded"
