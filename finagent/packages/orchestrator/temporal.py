from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from finagent.packages.agents.builders import CandidateResponse
from finagent.packages.common.config import Settings
from finagent.packages.common.context import AuthContext
from finagent.packages.common.errors import ExternalDependencyError
from finagent.packages.common.ids import new_id
from finagent.packages.common.store import Store, build_store
from finagent.packages.memory.graph_store import build_market_graph_store
from finagent.packages.memory.vector_store import build_memory_vector_store
from finagent.packages.orchestrator.models import AgentState, Message, SourceRef, StreamEvent
from finagent.packages.orchestrator.pipeline import ChatGraphState, FinAgentOrchestrator
from finagent.packages.rag.gateway import build_research_retriever
from finagent.packages.tools.registry import ToolContext


TemporalSnapshot = dict[str, Any]


@dataclass
class TemporalRuntime:
    settings: Settings
    store: Store
    orchestrator: FinAgentOrchestrator
    auth: AuthContext
    state: AgentState
    tool_context: ToolContext


class TemporalChatGateway:
    def __init__(self, settings: Settings) -> None:
        if not settings.temporal_target_host:
            raise ExternalDependencyError("Temporal 编排需要配置 FINAGENT_TEMPORAL_TARGET_HOST")
        self.settings = settings

    def run_chat(
        self,
        *,
        auth: AuthContext,
        session_id: str,
        content: str,
        account_scope: str | None = None,
    ) -> list[StreamEvent]:
        return asyncio.run(
            self._run_chat(
                auth=auth,
                session_id=session_id,
                content=content,
                account_scope=account_scope,
            )
        )

    async def _run_chat(
        self,
        *,
        auth: AuthContext,
        session_id: str,
        content: str,
        account_scope: str | None,
    ) -> list[StreamEvent]:
        try:
            from temporalio.client import Client
        except ImportError as exc:
            raise RuntimeError(
                "Temporal 编排需要安装 temporalio。请使用 production extra 安装 finagent。"
            ) from exc
        client = await Client.connect(
            self.settings.temporal_target_host,
            namespace=self.settings.temporal_namespace,
        )
        payload = {
            "settings": self.settings.model_dump(mode="json"),
            "auth": auth.model_dump(mode="json"),
            "session_id": session_id,
            "content": content,
            "account_scope": account_scope,
        }
        result = await client.execute_workflow(
            FinAgentChatWorkflow.run,
            payload,
            id=new_id("wf_chat"),
            task_queue=self.settings.temporal_task_queue,
            execution_timeout=timedelta(seconds=self.settings.temporal_workflow_timeout_seconds),
        )
        return [StreamEvent.model_validate(item) for item in result]


def build_temporal_chat_gateway(settings: Settings) -> TemporalChatGateway:
    return TemporalChatGateway(settings)


try:
    from temporalio import activity, workflow
except ImportError:
    activity = None
    workflow = None


def _activity_defn(fn):
    return activity.defn(fn) if activity is not None else fn


def _activity_timeout(snapshot: TemporalSnapshot) -> timedelta:
    settings = snapshot["settings"]
    return timedelta(seconds=settings.get("temporal_workflow_timeout_seconds", 300))


@_activity_defn
async def prepare_chat_activity(payload: dict[str, Any]) -> TemporalSnapshot:
    return _prepare_chat_activity_sync(payload)


@_activity_defn
async def memory_routing_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _memory_routing_activity_sync(snapshot)


@_activity_defn
async def macro_agent_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _agent_activity_sync(snapshot, "macro_agent")


@_activity_defn
async def quant_agent_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _agent_activity_sync(snapshot, "quant_agent")


@_activity_defn
async def report_agent_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _agent_activity_sync(snapshot, "report_agent")


@_activity_defn
async def market_agent_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _agent_activity_sync(snapshot, "market_agent")


@_activity_defn
async def memory_agent_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _agent_activity_sync(snapshot, "memory_agent")


@_activity_defn
async def router_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _agent_activity_sync(snapshot, "router")


@_activity_defn
async def candidate_response_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _candidate_response_activity_sync(snapshot)


@_activity_defn
async def critic_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _critic_activity_sync(snapshot)


@_activity_defn
async def final_response_activity(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    return _final_response_activity_sync(snapshot)


TEMPORAL_AGENT_ACTIVITIES: dict[str, Callable[[TemporalSnapshot], Any]] = {
    "macro_agent": macro_agent_activity,
    "quant_agent": quant_agent_activity,
    "report_agent": report_agent_activity,
    "market_agent": market_agent_activity,
    "memory_agent": memory_agent_activity,
    "router": router_activity,
}

TEMPORAL_ACTIVITIES = [
    prepare_chat_activity,
    memory_routing_activity,
    macro_agent_activity,
    quant_agent_activity,
    report_agent_activity,
    market_agent_activity,
    memory_agent_activity,
    router_activity,
    candidate_response_activity,
    critic_activity,
    final_response_activity,
]


if workflow is not None:

    @workflow.defn
    class FinAgentChatWorkflow:
        @workflow.run
        async def run(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
            snapshot = await _execute_activity(prepare_chat_activity, payload)
            snapshot = await _execute_activity(memory_routing_activity, snapshot)
            while not snapshot.get("done"):
                node = _temporal_next_node(snapshot)
                if node == "candidate_response":
                    snapshot = await _execute_activity(candidate_response_activity, snapshot)
                elif node == "critic":
                    snapshot = await _execute_activity(critic_activity, snapshot)
                elif node == "final_builder":
                    snapshot = await _execute_activity(final_response_activity, snapshot)
                    break
                else:
                    snapshot = await _execute_activity(TEMPORAL_AGENT_ACTIVITIES.get(node, router_activity), snapshot)
            return snapshot["events"]

else:

    class FinAgentChatWorkflow:  # type: ignore[no-redef]
        pass


async def _execute_activity(fn, payload):
    return await workflow.execute_activity(
        fn,
        payload,
        start_to_close_timeout=_activity_timeout(payload),
    )


def _prepare_chat_activity_sync(payload: dict[str, Any]) -> TemporalSnapshot:
    settings = Settings.model_validate(payload["settings"])
    settings.validate_production_dependencies()
    auth = AuthContext.model_validate(payload["auth"])
    store = build_store(settings)
    try:
        orchestrator = _build_orchestrator(settings, store)
        trace_id = new_id("trace")
        request_id = new_id("req")
        task_id = new_id("task")
        session_id = payload["session_id"]
        content = payload["content"]
        account_scope = payload.get("account_scope")
        store.ensure_conversation(auth.tenant_id, auth.user_id, session_id)
        user_message = Message(role="user", content=content)
        store.add_message(
            auth.tenant_id,
            session_id,
            auth.user_id,
            user_message.message_id,
            user_message.role,
            user_message.content,
            user_message.token_count,
        )
        store.create_task(auth.tenant_id, session_id, auth.user_id, task_id, trace_id)
        recent_messages = [
            Message(
                role=row["role"],
                content=row["content"],
                message_id=row["message_id"],
                token_count=row["token_count"],
            )
            for row in store.get_recent_messages(auth.tenant_id, session_id, auth.user_id, 20)
        ]
        state = AgentState(
            trace_id=trace_id,
            tenant_id=auth.tenant_id,
            session_id=session_id,
            user_id=auth.user_id,
            request_id=request_id,
            task_id=task_id,
            messages=recent_messages,
        )
        previous_state = store.get_latest_session_state(auth.tenant_id, session_id, auth.user_id)
        if previous_state and orchestrator._should_restore_previous_active_task(previous_state):
            state.active_task = previous_state.active_task
            state.task_stack = previous_state.task_stack
        received = orchestrator._emit(
            state,
            "status",
            {
                "stage": "received",
                "input": {
                    "session_id": session_id,
                    "message": content,
                    "account_scope": account_scope,
                },
                "output": {
                    "message": "request accepted",
                    "trace_id": trace_id,
                    "request_id": request_id,
                    "task_id": task_id,
                },
            },
        )
        return {
            "settings": settings.model_dump(mode="json"),
            "auth": auth.model_dump(mode="json"),
            "content": content,
            "account_scope": account_scope,
            "state": state.model_dump(mode="json"),
            "previous_state": previous_state.model_dump(mode="json") if previous_state else None,
            "route_index": 0,
            "known_tool_ids": [],
            "events": [received.model_dump(mode="json")],
        }
    finally:
        store.close()


def _memory_routing_activity_sync(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    runtime = _open_runtime(snapshot)
    try:
        graph_state = _graph_state(snapshot, runtime)
        graph_state.update(runtime.orchestrator._graph_memory_retrieval(graph_state))
        _append_event(snapshot, runtime.orchestrator._emit_memory_event(runtime.state, snapshot["content"], graph_state.get("created_memories", [])))
        graph_state.update(runtime.orchestrator._graph_intent_routing(graph_state))
        snapshot["route_index"] = graph_state.get("route_index", 0)
        _append_event(
            snapshot,
            runtime.orchestrator._emit_routing_event(
                runtime.state,
                snapshot["content"],
                graph_state.get("previous_active_task"),
            ),
        )
        snapshot["state"] = runtime.state.model_dump(mode="json")
        return snapshot
    finally:
        runtime.store.close()


def _agent_activity_sync(snapshot: TemporalSnapshot, node_name: str) -> TemporalSnapshot:
    runtime = _open_runtime(snapshot)
    try:
        graph_state = _graph_state(snapshot, runtime)
        node_update = _run_graph_agent_node(runtime.orchestrator, node_name, graph_state)
        graph_state.update(node_update)
        output = graph_state["agent_output"]
        known_tool_ids = set(snapshot.get("known_tool_ids", []))
        _append_event(snapshot, runtime.orchestrator._emit_agent_event(runtime.state, output))
        for event in runtime.orchestrator._emit_new_tool_events(runtime.state, known_tool_ids, output):
            _append_event(snapshot, event)
        snapshot["known_tool_ids"] = sorted(known_tool_ids)
        snapshot["route_index"] = graph_state.get("route_index", snapshot.get("route_index", 0))
        snapshot["state"] = runtime.state.model_dump(mode="json")
        return snapshot
    finally:
        runtime.store.close()


def _candidate_response_activity_sync(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    runtime = _open_runtime(snapshot)
    try:
        graph_state = _graph_state(snapshot, runtime)
        graph_state.update(runtime.orchestrator._graph_candidate_response(graph_state))
        snapshot["state"] = runtime.state.model_dump(mode="json")
        if "error_payload" in graph_state:
            _append_event(snapshot, runtime.orchestrator._emit(runtime.state, "error", graph_state["error_payload"]))
            snapshot["done"] = True
            return snapshot
        candidate = graph_state["candidate"]
        snapshot["candidate"] = _candidate_to_json(candidate)
        _append_event(snapshot, runtime.orchestrator._emit_candidate_event(runtime.state, candidate))
        return snapshot
    finally:
        runtime.store.close()


def _critic_activity_sync(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    runtime = _open_runtime(snapshot)
    try:
        graph_state = _graph_state(snapshot, runtime)
        graph_state["candidate"] = _candidate_from_json(snapshot["candidate"])
        graph_state.update(runtime.orchestrator._graph_critic_review(graph_state))
        critic_output = graph_state["critic_output"]
        known_tool_ids = set(snapshot.get("known_tool_ids", []))
        for event in runtime.orchestrator._emit_new_tool_events(runtime.state, known_tool_ids, critic_output):
            _append_event(snapshot, event)
        snapshot["known_tool_ids"] = sorted(known_tool_ids)
        _append_event(
            snapshot,
            runtime.orchestrator._emit_risk_event(
                runtime.state,
                graph_state["candidate"],
                graph_state["compliance_result"],
            ),
        )
        snapshot["route_index"] = graph_state.get("route_index", snapshot.get("route_index", 0))
        snapshot["state"] = runtime.state.model_dump(mode="json")
        return snapshot
    finally:
        runtime.store.close()


def _final_response_activity_sync(snapshot: TemporalSnapshot) -> TemporalSnapshot:
    runtime = _open_runtime(snapshot)
    try:
        graph_state = _graph_state(snapshot, runtime)
        graph_state["candidate"] = _candidate_from_json(snapshot["candidate"])
        graph_state.update(runtime.orchestrator._graph_final_response(graph_state))
        resumed_task = graph_state.get("resumed_task")
        _append_event(snapshot, runtime.orchestrator._emit_final_event(runtime.state, resumed_task))
        snapshot["state"] = runtime.state.model_dump(mode="json")
        previous_state_payload = snapshot.get("previous_state")
        if resumed_task and previous_state_payload:
            previous_state = AgentState.model_validate(previous_state_payload)
            for event in runtime.orchestrator._run_resumed_task_events(
                previous_state=previous_state,
                auth=runtime.auth,
                account_scope=snapshot.get("account_scope"),
            ):
                _append_event(snapshot, event)
        snapshot["done"] = True
        return snapshot
    finally:
        runtime.store.close()


def _run_graph_agent_node(orchestrator: FinAgentOrchestrator, node_name: str, graph_state: ChatGraphState) -> ChatGraphState:
    node_methods = {
        "macro_agent": orchestrator._graph_macro_agent,
        "quant_agent": orchestrator._graph_quant_agent,
        "report_agent": orchestrator._graph_report_agent,
        "market_agent": orchestrator._graph_market_agent,
        "memory_agent": orchestrator._graph_memory_agent,
        "router": orchestrator._graph_router,
    }
    return node_methods.get(node_name, orchestrator._graph_router)(graph_state)


def _open_runtime(snapshot: TemporalSnapshot) -> TemporalRuntime:
    settings = Settings.model_validate(snapshot["settings"])
    settings.validate_production_dependencies()
    auth = AuthContext.model_validate(snapshot["auth"])
    state = AgentState.model_validate(snapshot["state"])
    store = build_store(settings)
    orchestrator = _build_orchestrator(settings, store)
    tool_context = ToolContext(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        task_id=state.task_id,
        trace_id=state.trace_id,
        account_scope=snapshot.get("account_scope"),
    )
    return TemporalRuntime(
        settings=settings,
        store=store,
        orchestrator=orchestrator,
        auth=auth,
        state=state,
        tool_context=tool_context,
    )


def _build_orchestrator(settings: Settings, store: Store) -> FinAgentOrchestrator:
    return FinAgentOrchestrator(
        store,
        settings=settings,
        memory_vector_store=build_memory_vector_store(settings),
        graph_store=build_market_graph_store(settings),
        research_retriever=build_research_retriever(settings),
    )


def _graph_state(snapshot: TemporalSnapshot, runtime: TemporalRuntime) -> ChatGraphState:
    graph_state: ChatGraphState = {
        "state": runtime.state,
        "content": snapshot["content"],
        "tool_context": runtime.tool_context,
        "route_index": snapshot.get("route_index", 0),
    }
    if snapshot.get("candidate"):
        graph_state["candidate"] = _candidate_from_json(snapshot["candidate"])
    return graph_state


def _temporal_next_node(snapshot: TemporalSnapshot) -> str:
    state = snapshot["state"]
    route_plan = state.get("route_plan", [])
    route_index = snapshot.get("route_index", 0)
    if route_index >= len(route_plan):
        return "final_builder"
    node = route_plan[route_index].get("node", "router")
    if node == "critic" and "candidate" not in snapshot:
        return "candidate_response"
    if node in {
        "macro_agent",
        "quant_agent",
        "report_agent",
        "market_agent",
        "memory_agent",
        "router",
        "critic",
        "final_builder",
    }:
        return node
    return "router"


def _append_event(snapshot: TemporalSnapshot, event: StreamEvent) -> None:
    snapshot.setdefault("events", []).append(event.model_dump(mode="json"))


def _candidate_to_json(candidate: CandidateResponse) -> dict[str, Any]:
    return {
        "answer": candidate.answer,
        "sources": [source.model_dump(mode="json") for source in candidate.sources],
        "llm_provider": candidate.llm_provider,
        "llm_model": candidate.llm_model,
        "prompt_tokens": candidate.prompt_tokens,
        "completion_tokens": candidate.completion_tokens,
    }


def _candidate_from_json(payload: dict[str, Any]) -> CandidateResponse:
    return CandidateResponse(
        answer=payload["answer"],
        sources=[SourceRef.model_validate(item) for item in payload.get("sources", [])],
        llm_provider=payload.get("llm_provider", "ollama"),
        llm_model=payload.get("llm_model"),
        prompt_tokens=payload.get("prompt_tokens", 0),
        completion_tokens=payload.get("completion_tokens", 0),
    )
