from __future__ import annotations

import json

from finagent.packages.common.config import get_settings
from finagent.packages.common.context import AuthContext
from finagent.packages.common.store import build_store
from finagent.packages.memory.graph_store import build_market_graph_store
from finagent.packages.memory.vector_store import build_memory_vector_store
from finagent.packages.ops.readiness import run_deep_readiness_checks
from finagent.packages.orchestrator.pipeline import FinAgentOrchestrator
from finagent.packages.rag.gateway import build_research_retriever


def main() -> int:
    settings = get_settings()
    settings.validate_production_dependencies()
    store = build_store(settings)
    try:
        orchestrator = FinAgentOrchestrator(
            store,
            settings=settings,
            memory_vector_store=build_memory_vector_store(settings),
            graph_store=build_market_graph_store(settings),
            research_retriever=build_research_retriever(settings),
        )
        result = run_deep_readiness_checks(
            orchestrator=orchestrator,
            auth=AuthContext(
                tenant_id=settings.default_tenant_id,
                user_id=settings.default_user_id,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ready" else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
