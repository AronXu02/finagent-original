from __future__ import annotations

from collections.abc import Iterator
import re
import threading
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE_SAVER = True
except ImportError:
    SqliteSaver = None  # type: ignore[assignment]
    _HAS_SQLITE_SAVER = False

from finagent.packages.agents.builders import CandidateResponse
from finagent.packages.agents.builders import AnswerLLM, CandidateResponseBuilder, FinalResponseBuilder
from finagent.packages.agents.base import execute_tool
from finagent.packages.common.errors import FinAgentError
from finagent.packages.agents.macro import MacroAgent
from finagent.packages.agents.market import MarketAgent
from finagent.packages.agents.quant import QuantAgent
from finagent.packages.agents.report import ReportAgent
from finagent.packages.agents.user_proxy import UserProxy, build_intent_classifier
from finagent.packages.common.config import Settings, get_settings
from finagent.packages.common.context import AuthContext
from finagent.packages.common.ids import new_id
from finagent.packages.common.store import Store
from finagent.packages.llm import build_llm_client
from finagent.packages.market_data.adapters import MarketDataProvider
from finagent.packages.memory.graph_store import MarketGraphStore
from finagent.packages.memory.service import MemoryService, build_memory_reranker, build_short_term_summarizer
from finagent.packages.memory.vector_store import MemoryVectorStore
from finagent.packages.orchestrator.models import (
    AgentOutput,
    AgentState,
    FinalResponse,
    Message,
    RiskReview,
    StreamEvent,
)
from finagent.packages.rag.gateway import ResearchRetriever
from finagent.packages.risk.rules import RiskService
from finagent.packages.tools.executor import ToolExecutor, build_default_registry
from finagent.packages.tools.registry import ToolContext


class ChatGraphState(TypedDict, total=False):
    state: AgentState
    content: str
    tool_context: ToolContext
    risk_level: str
    created_memories: list[Any]
    previous_active_task: dict[str, Any] | None
    route_index: int
    agent_output: AgentOutput
    candidate: CandidateResponse
    error_payload: dict[str, Any]
    compliance_result: Any
    critic_output: AgentOutput
    resumed_task: dict[str, Any] | None


ROUTE_EXECUTION_NODES = {
    "macro_agent",
    "quant_agent",
    "report_agent",
    "market_agent",
    "memory_agent",
    "router",
    "critic",
    "final_builder",
}


AGENT_EVENT_NODES = {
    "macro_agent",
    "quant_agent",
    "report_agent",
    "market_agent",
    "memory_agent",
    "router",
}


class FinAgentOrchestrator:
    _thread_local = threading.local()

    def __init__(
        self,
        store: Store,
        settings: Settings | None = None,
        market_data_provider: MarketDataProvider | None = None,
        memory_vector_store: MemoryVectorStore | None = None,
        graph_store: MarketGraphStore | None = None,
        research_retriever: ResearchRetriever | None = None,
        llm_client: AnswerLLM | None = None,
    ) -> None:
        if memory_vector_store is None:
            raise ValueError("FinAgentOrchestrator 需要配置记忆向量存储")
        if graph_store is None:
            raise ValueError("FinAgentOrchestrator 需要配置市场图存储")
        if research_retriever is None:
            raise ValueError("FinAgentOrchestrator 需要配置研报/新闻检索器")
        self.settings = settings or get_settings()
        self.store = store
        self.memory = MemoryService(
            store,
            max_recent_messages=self.settings.max_recent_messages,
            vector_store=memory_vector_store,
            graph_store=graph_store,
            summarizer=build_short_term_summarizer(
                base_url=self.settings.ollama_base_url if self.settings.llm_provider == "ollama" else None,
                model=self.settings.memory_summary_model,
                timeout_seconds=self.settings.memory_summary_timeout_seconds,
            ),
            reranker=build_memory_reranker(
                base_url=self.settings.memory_rerank_base_url,
                model=self.settings.memory_rerank_model,
                api_key=self.settings.memory_rerank_api_key,
                timeout_seconds=self.settings.memory_rerank_timeout_seconds,
            ),
        )
        self.user_proxy = UserProxy(intent_classifier=build_intent_classifier(self.settings))
        self.executor = ToolExecutor(
            build_default_registry(
                self.settings,
                store,
                market_data_provider=market_data_provider,
                research_retriever=research_retriever,
            ),
            store=store,
        )
        self.macro_agent = MacroAgent()
        self.quant_agent = QuantAgent()
        self.report_agent = ReportAgent()
        self.market_agent = MarketAgent()
        self.candidate_builder = CandidateResponseBuilder(llm_client=llm_client or build_llm_client(self.settings))
        self.final_builder = FinalResponseBuilder()
        self.risk = RiskService()
        self.langgraph_checkpointer = self._build_checkpointer()
        self.chat_graph = self._build_chat_graph()

    def _build_checkpointer(self):
        if _HAS_SQLITE_SAVER and self.settings.database_backend == "sqlite" and self.settings.database_url:
            from pathlib import Path
            db_path = Path(self.settings.database_url)
            checkpoints_path = db_path.parent / "langgraph_checkpoints.sqlite3"
            return SqliteSaver.from_conn_string(str(checkpoints_path))
        return MemorySaver()

    @staticmethod
    def _chunk_answer_for_stream(answer: str, chunk_size: int = 8) -> list[str]:
        if not answer:
            return []
        chunks = re.split(r"(?<=[。！？\n.,;:!?])", answer)
        result: list[str] = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                if chunk:
                    result.append(chunk)
            else:
                for i in range(0, len(chunk), chunk_size):
                    result.append(chunk[i:i + chunk_size])
        return [c for c in result if c]

    def run_chat(
        self,
        *,
        auth: AuthContext,
        session_id: str,
        content: str,
        account_scope: str | None = None,
        risk_level: str | None = None,
    ) -> Iterator[StreamEvent]:
        self._thread_local.checkpoint_id = ""
        trace_id = new_id("trace")
        request_id = new_id("req")
        task_id = new_id("task")
        self.store.ensure_conversation(auth.tenant_id, auth.user_id, session_id)
        user_message = Message(role="user", content=content)
        self.store.add_message(
            auth.tenant_id,
            session_id,
            auth.user_id,
            user_message.message_id,
            user_message.role,
            user_message.content,
            user_message.token_count,
        )
        self.store.create_task(auth.tenant_id, session_id, auth.user_id, task_id, trace_id)

        effective_risk_level = risk_level or self.store.get_user_risk_level(auth.tenant_id, auth.user_id)

        tool_progress: list[dict[str, Any]] = []

        def progress_callback(tool_name: str, phase: str) -> None:
            tool_progress.append({"tool_name": tool_name, "phase": phase})

        recent_messages = [
            Message(
                role=row["role"],
                content=row["content"],
                message_id=row["message_id"],
                token_count=row["token_count"],
            )
            for row in self.store.get_recent_messages(auth.tenant_id, session_id, auth.user_id, self.settings.max_recent_messages)
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
        previous_state = self.store.get_latest_session_state(auth.tenant_id, session_id, auth.user_id)
        if previous_state and self._should_restore_previous_active_task(previous_state):
            state.active_task = previous_state.active_task
            state.task_stack = previous_state.task_stack
        yield self._emit(
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

        tool_context = ToolContext(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            task_id=task_id,
            trace_id=trace_id,
            account_scope=account_scope,
            risk_level=effective_risk_level,
        )
        known_tool_ids: set[str] = set()
        graph_state: ChatGraphState = {
            "state": state,
            "content": content,
            "tool_context": tool_context,
            "risk_level": effective_risk_level,
        }
        graph_config = {"configurable": {"thread_id": task_id}}
        ToolExecutor.set_progress_callback(progress_callback)
        try:
            for update in self.chat_graph.stream(graph_state, stream_mode="updates", config=graph_config):
                node_name, node_update = next(iter(update.items()))
                if node_update:
                    graph_state.update(node_update)
                state = graph_state["state"]
                if node_name == "memory_retrieval":
                    yield self._emit_memory_event(state, content, graph_state.get("created_memories", []))
                elif node_name == "intent_routing":
                    yield self._emit_routing_event(state, content, graph_state.get("previous_active_task"), checkpoint=True)
                elif node_name in AGENT_EVENT_NODES:
                    output = graph_state["agent_output"]
                    yield self._emit_agent_event(state, output)
                    for entry in tool_progress:
                        yield self._emit(
                            state, "status",
                            {
                                "stage": "tool_progress",
                                "input": {"tool": entry["tool_name"]},
                                "output": {"phase": entry["phase"], "tool": entry["tool_name"]},
                            },
                        )
                    tool_progress.clear()
                    for event in self._emit_new_tool_events(state, known_tool_ids, output):
                        yield event
                elif node_name == "candidate_response":
                    if "error_payload" in graph_state:
                        yield self._emit(state, "error", graph_state["error_payload"], checkpoint=True)
                        return
                    candidate = graph_state["candidate"]
                    for token_chunk in self._chunk_answer_for_stream(candidate.answer):
                        yield self._emit(state, "token", {"token": token_chunk})
                    yield self._emit_candidate_event(state, candidate)
                elif node_name == "critic":
                    critic_output = graph_state["critic_output"]
                    for event in self._emit_new_tool_events(state, known_tool_ids, critic_output):
                        yield event
                    yield self._emit_risk_event(state, graph_state["candidate"], graph_state["compliance_result"], checkpoint=True)
                elif node_name == "final_builder":
                    yield self._emit_final_event(state, graph_state.get("resumed_task"), checkpoint=True)
                    if graph_state.get("resumed_task") and previous_state is not None:
                        yield from self._run_resumed_task_events(
                            previous_state=previous_state,
                            auth=auth,
                            account_scope=account_scope,
                        )
        finally:
            ToolExecutor.set_progress_callback(None)

    def resume_events(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        session_id: str,
        last_sequence: int = 0,
        last_event_id: str | None = None,
    ) -> list[StreamEvent]:
        self.store.validate_task_owner(auth.tenant_id, auth.user_id, task_id, session_id)
        return self.store.get_events_after(
            auth.tenant_id,
            task_id,
            last_sequence=last_sequence,
            last_event_id=last_event_id,
        )

    def _run_resumed_task_events(
        self,
        *,
        previous_state: AgentState,
        auth: AuthContext,
        account_scope: str | None,
    ) -> Iterator[StreamEvent]:
        if previous_state.active_task is None:
            return
        self._thread_local.checkpoint_id = ""
        content = previous_state.active_task.description
        previous_state.interruption = None
        previous_state.task_stack = []
        previous_state.status = "running"
        tool_context = ToolContext(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            task_id=previous_state.task_id,
            trace_id=previous_state.trace_id,
            account_scope=account_scope,
        )
        yield self._emit(
            previous_state,
            "status",
            {
                "stage": "resume_task",
                "input": {
                    "resumed_task": previous_state.active_task.model_dump(mode="json"),
                },
                "output": {
                    "message": "resumed interrupted task",
                    "task_id": previous_state.task_id,
                    "checkpoint_id": previous_state.active_task.checkpoint_id,
                },
            },
        )
        known_tool_ids = set(previous_state.tool_results)
        graph_state: ChatGraphState = {
            "state": previous_state,
            "content": content,
            "tool_context": tool_context,
        }
        graph_config = {"configurable": {"thread_id": previous_state.task_id}}
        for update in self.chat_graph.stream(graph_state, stream_mode="updates", config=graph_config):
            node_name, node_update = next(iter(update.items()))
            if node_update:
                graph_state.update(node_update)
            state = graph_state["state"]
            if node_name == "memory_retrieval":
                yield self._emit_memory_event(state, content, graph_state.get("created_memories", []))
            elif node_name == "intent_routing":
                yield self._emit_routing_event(state, content, None, checkpoint=True)
            elif node_name in AGENT_EVENT_NODES:
                output = graph_state["agent_output"]
                yield self._emit_agent_event(state, output)
                for event in self._emit_new_tool_events(state, known_tool_ids, output):
                    yield event
            elif node_name == "candidate_response":
                if "error_payload" in graph_state:
                    yield self._emit(state, "error", graph_state["error_payload"], checkpoint=True)
                    return
                candidate = graph_state["candidate"]
                for token_chunk in self._chunk_answer_for_stream(candidate.answer):
                    yield self._emit(state, "token", {"token": token_chunk})
                yield self._emit_candidate_event(state, candidate)
            elif node_name == "critic":
                critic_output = graph_state["critic_output"]
                for event in self._emit_new_tool_events(state, known_tool_ids, critic_output):
                    yield event
                yield self._emit_risk_event(state, graph_state["candidate"], graph_state["compliance_result"], checkpoint=True)
            elif node_name == "final_builder":
                yield self._emit_final_event(state, graph_state.get("resumed_task"), checkpoint=True)

    def _build_chat_graph(self):
        graph = StateGraph(ChatGraphState)
        graph.add_node("memory_retrieval", self._graph_memory_retrieval)
        graph.add_node("intent_routing", self._graph_intent_routing)
        graph.add_node("macro_agent", self._graph_macro_agent)
        graph.add_node("quant_agent", self._graph_quant_agent)
        graph.add_node("report_agent", self._graph_report_agent)
        graph.add_node("market_agent", self._graph_market_agent)
        graph.add_node("memory_agent", self._graph_memory_agent)
        graph.add_node("router", self._graph_router)
        graph.add_node("candidate_response", self._graph_candidate_response)
        graph.add_node("critic", self._graph_critic_review)
        graph.add_node("final_builder", self._graph_final_response)
        graph.add_edge(START, "memory_retrieval")
        graph.add_edge("memory_retrieval", "intent_routing")
        route_targets = {
            "macro_agent": "macro_agent",
            "quant_agent": "quant_agent",
            "report_agent": "report_agent",
            "market_agent": "market_agent",
            "memory_agent": "memory_agent",
            "router": "router",
            "candidate_response": "candidate_response",
            "critic": "critic",
            "final_builder": "final_builder",
        }
        graph.add_conditional_edges("intent_routing", self._next_route_node, route_targets)
        for node in AGENT_EVENT_NODES:
            graph.add_conditional_edges(node, self._next_route_node, route_targets)
        graph.add_conditional_edges(
            "candidate_response",
            self._candidate_route,
            {"error": END, "critic": "critic"},
        )
        graph.add_conditional_edges("critic", self._next_route_node, route_targets)
        graph.add_edge("final_builder", END)
        return graph.compile(checkpointer=self.langgraph_checkpointer)

    def _graph_memory_retrieval(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        content = graph_state["content"]
        state.short_term_summary = self.memory.compress_short_term(state.messages)
        created_memories = self.memory.extract_and_store_preferences(
            tenant_id=state.tenant_id,
            user_id=state.user_id,
            session_id=state.session_id,
            text=content,
        )
        market_state = self.memory.detect_market_state(content, state.short_term_summary)
        state.memory_context = self.memory.retrieve_context(
            tenant_id=state.tenant_id,
            user_id=state.user_id,
            query=content,
            market_state=market_state,
        )
        return {"state": state, "created_memories": created_memories}

    def _graph_intent_routing(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        previous_active_task = state.active_task.model_dump(mode="json") if state.active_task else None
        self.user_proxy.plan(state, graph_state["content"])
        return {"state": state, "previous_active_task": previous_active_task, "route_index": 0}

    @staticmethod
    def _run_with_timeout(func, timeout_seconds: float = 60.0):
        result_container: list[Any] = []
        error_container: list[Exception | None] = [None]

        def target():
            try:
                result_container.append(func())
            except Exception as exc:
                error_container[0] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            raise TimeoutError(f"Agent execution timed out after {timeout_seconds}s")
        if error_container[0] is not None:
            raise error_container[0]
        return result_container[0] if result_container else None

    def _graph_macro_agent(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        output = self._run_with_timeout(
            lambda: self.macro_agent.run(state, self.executor, graph_state["tool_context"], graph_state["content"]),
            timeout_seconds=self.settings.agent_timeout_seconds,
        )
        return self._complete_route_step(graph_state, "macro_agent", output)

    def _graph_quant_agent(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        output = self._run_with_timeout(
            lambda: self.quant_agent.run(
                state,
                self.executor,
                graph_state["tool_context"],
                user_input=graph_state["content"],
            ),
            timeout_seconds=self.settings.agent_timeout_seconds,
        )
        return self._complete_route_step(graph_state, "quant_agent", output)

    def _graph_report_agent(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        output = self._run_with_timeout(
            lambda: self.report_agent.run(state, self.executor, graph_state["tool_context"], graph_state["content"]),
            timeout_seconds=self.settings.agent_timeout_seconds,
        )
        return self._complete_route_step(graph_state, "report_agent", output)

    def _graph_market_agent(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        output = self._run_with_timeout(
            lambda: self.market_agent.run(state, self.executor, graph_state["tool_context"], graph_state["content"]),
            timeout_seconds=self.settings.agent_timeout_seconds,
        )
        return self._complete_route_step(graph_state, "market_agent", output)

    def _graph_memory_agent(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        findings = [item.content for item in state.memory_context.top_memories]
        if not findings:
            findings = ["本条消息未检测到可支持的偏好记忆模式。"]
        output = AgentOutput(
            agent_name="memory_agent",
            summary="用户偏好状态已写入并完成记忆检索。",
            findings=findings,
            confidence=state.intent_confidence,
            metadata={"memory_context": state.memory_context.model_dump(mode="json")},
        )
        state.agent_outputs[output.agent_name] = output
        return self._complete_route_step(graph_state, "memory_agent", output)

    def _graph_router(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        if state.current_intent and state.current_intent.value == "high_risk_request":
            output = AgentOutput(
                agent_name="router",
                summary="生成投资建议前已识别高风险请求。",
                findings=["该请求需要先经过风控审核并拒绝，因此未调用推荐类工具。"],
                confidence=0.99,
            )
        elif state.current_intent and state.current_intent.value == "clarification":
            output = AgentOutput(
                agent_name="router",
                summary="调用工具前需要进一步澄清。",
                findings=[
                    "用户消息引用了历史上下文或未明确资产，但当前没有可可靠继承的活跃任务。",
                    "在调用行情、财报或组合工具前，应先提出一个简短追问。",
                ],
                confidence=state.intent_confidence,
            )
        else:
            output = AgentOutput(
                agent_name="router",
                summary="通用金融知识问题已路由为无需外部工具调用。",
                findings=[
                    "概念解释不需要外部行情数据。",
                    "回答应保持教育说明口径，避免具体资产推荐或交易执行指令。",
                ],
                confidence=0.55,
            )
        state.agent_outputs[output.agent_name] = output
        return self._complete_route_step(graph_state, "router", output)

    def _graph_candidate_response(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        try:
            candidate = self.candidate_builder.build(state, graph_state["content"])
        except FinAgentError as exc:
            state.status = "failed"
            self.store.update_task(state.tenant_id, state.task_id, status=state.status)
            return {
                "state": state,
                "error_payload": {
                    "stage": "candidate_response",
                    "input": {
                        "intent": state.current_intent.value if state.current_intent else None,
                    },
                    "output": {
                        "code": exc.code,
                        "message": exc.message,
                        "status": state.status,
                    },
                    "status": state.status,
                },
            }
        except Exception as exc:
            state.status = "failed"
            self.store.update_task(state.tenant_id, state.task_id, status=state.status)
            return {
                "state": state,
                "error_payload": {
                    "stage": "candidate_response",
                    "input": {},
                    "output": {
                        "code": "llm_generation_error",
                        "message": f"LLM 生成候选回复失败：{exc}",
                        "status": state.status,
                    },
                    "status": state.status,
                },
            }
        state.token_usage.prompt_tokens += candidate.prompt_tokens
        state.token_usage.completion_tokens += candidate.completion_tokens
        state.token_budget = state.token_usage.prompt_tokens + state.token_usage.completion_tokens
        return {"state": state, "candidate": candidate}

    def _candidate_route(self, graph_state: ChatGraphState) -> str:
        return "error" if "error_payload" in graph_state else "critic"

    def _next_route_node(self, graph_state: ChatGraphState) -> str:
        state = graph_state["state"]
        route_index = graph_state.get("route_index", 0)
        if route_index >= len(state.route_plan):
            return "final_builder"
        node = state.route_plan[route_index].node
        if node == "critic" and "candidate" not in graph_state:
            return "candidate_response"
        if node in ROUTE_EXECUTION_NODES:
            return node
        return "router"

    def _complete_route_step(
        self,
        graph_state: ChatGraphState,
        node_name: str,
        output: AgentOutput,
    ) -> ChatGraphState:
        return {
            "state": graph_state["state"],
            "route_index": self._advance_route_index(graph_state, node_name),
            "agent_output": output,
        }

    @staticmethod
    def _advance_route_index(graph_state: ChatGraphState, node_name: str) -> int:
        state = graph_state["state"]
        route_index = graph_state.get("route_index", 0)
        if route_index < len(state.route_plan) and state.route_plan[route_index].node == node_name:
            return route_index + 1
        for index in range(route_index, len(state.route_plan)):
            if state.route_plan[index].node == node_name:
                return index + 1
        return route_index

    def _graph_critic_review(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        candidate = graph_state["candidate"]
        compliance_result = execute_tool(
            state=state,
            executor=self.executor,
            context=graph_state["tool_context"],
            tool_name="compliance_rule_check",
            arguments={
                "text": candidate.answer,
                "user_input": graph_state["content"],
                "source_count": len(candidate.sources),
                "user_risk_level": graph_state.get("risk_level", "balanced"),
            },
        )
        critic_output = AgentOutput(
            agent_name="critic",
            summary="Compliance rule check completed.",
            findings=[f"Compliance verdict: {compliance_result.data.get('verdict', compliance_result.status)}"],
            tool_call_ids=[compliance_result.tool_call_id],
            sources=candidate.sources,
            confidence=1.0,
            metadata={
                "tool_status": compliance_result.status,
                "tool_error": compliance_result.error,
            },
        )
        state.agent_outputs[critic_output.agent_name] = critic_output
        review = self._risk_review_from_tool_result(compliance_result) or self.risk.review(
            candidate_text=candidate.answer,
            user_input=graph_state["content"],
            sources=candidate.sources,
        )
        state.critic_report = review
        self.store.insert_risk_review(
            state.tenant_id,
            state.task_id,
            review.review_id,
            review.verdict,
            [flag.model_dump(mode="json") for flag in review.flags],
            review.rewrite_suggestion,
        )
        return {
            "state": state,
            "route_index": self._advance_route_index(graph_state, "critic"),
            "compliance_result": compliance_result,
            "critic_output": critic_output,
        }

    def _graph_final_response(self, graph_state: ChatGraphState) -> ChatGraphState:
        state = graph_state["state"]
        candidate = graph_state["candidate"]
        review = state.critic_report
        if review is None:
            raise RuntimeError("生成最终答复前必须先完成风控审核")
        if review.verdict == "reject":
            final = FinalResponse(
                answer=self.risk.safe_response(user_input=graph_state["content"], sources=candidate.sources),
                sources=candidate.sources,
                risk_disclaimer="仅供决策参考，不保证收益；请核对数据时效、成本税费与自身适配性。",
            )
            state.status = "risk_rejected"
        elif review.verdict == "clarify":
            final = FinalResponse(
                answer="还需要补充风险承受能力、投资期限和关注资产范围，我再给出更贴近你的分析。",
                sources=candidate.sources,
                risk_disclaimer="当前尚未形成投资结论。",
            )
            state.status = "needs_clarification"
        elif review.verdict == "rewrite":
            final = FinalResponse(
                answer=self.risk.rewrite_response(review=review, candidate_text=candidate.answer, user_input=graph_state["content"]),
                sources=candidate.sources,
                risk_disclaimer="仅供决策参考，不保证收益；请核对数据时效、成本税费与自身适配性。",
            )
            state.status = "completed"
        else:
            final = self.final_builder.build(candidate)
            state.status = "completed"
        resumed_task = self._resume_interrupted_task(state)
        if resumed_task is not None:
            final.answer = self._append_resume_notice(final.answer, resumed_task)
        state.final_response = final
        assistant_message = Message(role="assistant", content=final.answer)
        self.store.add_message(
            state.tenant_id,
            state.session_id,
            state.user_id,
            assistant_message.message_id,
            assistant_message.role,
            assistant_message.content,
            assistant_message.token_count,
        )
        self.store.update_task(state.tenant_id, state.task_id, status=state.status)
        if resumed_task is not None:
            self.store.update_task(
                state.tenant_id,
                resumed_task.task_id,
                status="running",
                active_node=resumed_task.intent.value,
                checkpoint_id=resumed_task.checkpoint_id,
            )
        return {
            "state": state,
            "resumed_task": resumed_task.model_dump(mode="json") if resumed_task else None,
        }

    def _emit_memory_event(self, state: AgentState, content: str, created_memories: list[Any], *, checkpoint: bool = False) -> StreamEvent:
        return self._emit(
            state,
            "status",
            {
                "stage": "memory_retrieval",
                "input": {
                    "query": content,
                    "message_count": len(state.messages),
                },
                "output": {
                    "short_term_summary": state.short_term_summary,
                    "memory_count": len(state.memory_context.top_memories),
                    "created_memories": [item.model_dump(mode="json") for item in created_memories],
                    "top_memories": [item.model_dump(mode="json") for item in state.memory_context.top_memories],
                    "conflicts": state.memory_context.conflicts,
                },
            },
            checkpoint=checkpoint,
        )

    def _emit_routing_event(
        self,
        state: AgentState,
        content: str,
        previous_active_task: dict[str, Any] | None,
        *,
        checkpoint: bool = False,
    ) -> StreamEvent:
        return self._emit(
            state,
            "status",
            {
                "stage": "intent_routing",
                "input": {
                    "message": content,
                    "previous_active_task": previous_active_task,
                },
                "output": {
                    "intent": state.current_intent.value if state.current_intent else None,
                    "confidence": state.intent_confidence,
                    "route_plan": [step.model_dump(mode="json") for step in state.route_plan],
                    "active_task": state.active_task.model_dump(mode="json") if state.active_task else None,
                    "task_stack": [task.model_dump(mode="json") for task in state.task_stack],
                    "interruption": state.interruption.model_dump(mode="json") if state.interruption else None,
                },
            },
            checkpoint=checkpoint,
        )

    def _emit_agent_event(self, state: AgentState, output: AgentOutput, *, checkpoint: bool = False) -> StreamEvent:
        return self._emit(
            state,
            "status",
            {
                "stage": output.agent_name,
                "input": {
                    "intent": state.current_intent.value if state.current_intent else None,
                    "memory_context": state.memory_context.model_dump(mode="json"),
                    "route_plan": [step.model_dump(mode="json") for step in state.route_plan],
                },
                "output": output.model_dump(mode="json"),
            },
            checkpoint=checkpoint,
        )

    def _emit_candidate_event(self, state: AgentState, candidate: CandidateResponse, *, checkpoint: bool = False) -> StreamEvent:
        return self._emit(
            state,
            "status",
            {
                "stage": "candidate_response",
                "input": {
                    "agent_outputs": {
                        name: output.model_dump(mode="json")
                        for name, output in state.agent_outputs.items()
                    }
                },
                "output": {
                    "answer": candidate.answer,
                    "source_count": len(candidate.sources),
                    "sources": [source.model_dump(mode="json") for source in candidate.sources],
                    "llm_provider": candidate.llm_provider,
                    "llm_model": candidate.llm_model,
                    "token_usage": state.token_usage.model_dump(mode="json"),
                    "token_budget": state.token_budget,
                    "token_budget_warning": self._token_budget_warning(state),
                },
            },
            checkpoint=checkpoint,
        )

    def _emit_risk_event(self, state: AgentState, candidate: CandidateResponse, compliance_result: Any, *, checkpoint: bool = False) -> StreamEvent:
        review = state.critic_report
        if review is None:
            raise RuntimeError("critic review must exist before risk event")
        return self._emit(
            state,
            "risk_review",
            {
                "verdict": review.verdict,
                "input": {
                    "candidate_answer": candidate.answer,
                    "sources": [source.model_dump(mode="json") for source in candidate.sources],
                    "compliance_tool_call_id": compliance_result.tool_call_id,
                },
                "output": {
                    "verdict": review.verdict,
                    "flags": [flag.model_dump(mode="json") for flag in review.flags],
                    "rewrite_suggestion": review.rewrite_suggestion,
                    "self_reflection": review.self_reflection,
                    "review_id": review.review_id,
                    "compliance_tool": compliance_result.model_dump(mode="json"),
                },
            },
            checkpoint=checkpoint,
        )

    def _emit_final_event(self, state: AgentState, resumed_task: dict[str, Any] | None = None, *, checkpoint: bool = False) -> StreamEvent:
        final = state.final_response
        review = state.critic_report
        if final is None or review is None:
            raise RuntimeError("发送最终事件前必须存在最终答复和风控审核结果")
        output = {
            "answer": final.answer,
            "sources": [source.model_dump(mode="json") for source in final.sources],
            "risk_disclaimer": final.risk_disclaimer,
            "status": state.status,
            "token_usage": state.token_usage.model_dump(mode="json"),
            "token_budget": state.token_budget,
            "token_budget_warning": self._token_budget_warning(state),
        }
        if resumed_task is not None:
            output["resumed_task"] = resumed_task
        return self._emit(
            state,
            "final",
            {
                "input": {
                    "risk_verdict": review.verdict,
                    "risk_flags": [flag.model_dump(mode="json") for flag in review.flags],
                },
                "output": output,
                "answer": final.answer,
                "sources": [source.model_dump(mode="json") for source in final.sources],
                "risk_disclaimer": final.risk_disclaimer,
                "status": state.status,
                "resumed_task": resumed_task,
                "token_usage": state.token_usage.model_dump(mode="json"),
                "token_budget": state.token_budget,
                "token_budget_warning": self._token_budget_warning(state),
            },
            checkpoint=checkpoint,
        )

    def _emit_new_tool_events(
        self,
        state: AgentState,
        known_tool_ids: set[str],
        agent_output: AgentOutput,
    ) -> Iterator[StreamEvent]:
        calls_by_id = {call.tool_call_id: call for call in state.pending_tools}
        step_id = self._route_step_id(state, agent_output.agent_name)
        for tool_call_id in agent_output.tool_call_ids:
            if tool_call_id in known_tool_ids:
                continue
            result = state.tool_results.get(tool_call_id)
            if result is None:
                continue
            known_tool_ids.add(tool_call_id)
            call = calls_by_id.get(tool_call_id)
            yield self._emit(
                state,
                "tool_result",
                {
                    "stage": "tool_result",
                    "agent": agent_output.agent_name,
                    "step_id": step_id,
                    "input": {
                        "tool_call_id": tool_call_id,
                        "tool": result.tool_name,
                        "agent": agent_output.agent_name,
                        "step_id": step_id,
                        "arguments": call.arguments if call else {},
                        "idempotency_key": call.idempotency_key if call else None,
                    },
                    "output": result.model_dump(mode="json"),
                    "tool_call_id": tool_call_id,
                    "tool": result.tool_name,
                    "status": result.status,
                    "source_ts": result.source_ts.isoformat() if result.source_ts else None,
                    "error": result.error,
                },
            )

    def _route_step_id(self, state: AgentState, agent_name: str) -> str:
        for index, step in enumerate(state.route_plan, start=1):
            if step.node == agent_name:
                return f"route_step_{index}"
        return f"route_step:{agent_name}"

    @staticmethod
    def _risk_review_from_tool_result(compliance_result: Any) -> RiskReview | None:
        if getattr(compliance_result, "status", None) != "success":
            return None
        data = getattr(compliance_result, "data", None)
        if not isinstance(data, dict) or "verdict" not in data:
            return None
        try:
            return RiskReview.model_validate(
                {
                    "verdict": data["verdict"],
                    "flags": data.get("flags", []),
                    "rewrite_suggestion": data.get("rewrite_suggestion"),
                    "self_reflection": data.get("self_reflection", []),
                    "reviewed_text_hash": data.get("reviewed_text_hash"),
                }
            )
        except Exception:
            return None

    @staticmethod
    def _should_restore_previous_active_task(previous_state: AgentState) -> bool:
        if previous_state.status in {"running", "waiting_tool"}:
            return previous_state.active_task is not None
        if previous_state.status == "completed" and previous_state.active_task is not None:
            return previous_state.active_task.task_id != previous_state.task_id
        if previous_state.status in {"risk_rejected", "needs_clarification", "failed"}:
            return previous_state.active_task is not None and bool(previous_state.task_stack)
        return False

    @staticmethod
    def _resume_interrupted_task(state: AgentState):
        if state.interruption is None or not state.task_stack:
            return None
        resumed_task = state.task_stack.pop()
        state.active_task = resumed_task
        state.interruption = None
        return resumed_task

    @staticmethod
    def _append_resume_notice(answer: str, resumed_task: Any) -> str:
        description = getattr(resumed_task, "description", "") or getattr(resumed_task, "task_id", "previous task")
        return f"{answer}\n\n已恢复之前暂停的任务：{description}"

    def _token_budget_warning(self, state: AgentState) -> dict[str, Any]:
        threshold = self.settings.token_budget_warning_threshold
        return {
            "used": state.token_budget,
            "threshold": threshold,
            "warning": threshold > 0 and state.token_budget >= threshold,
        }

    def _emit(self, state: AgentState, event_type: str, payload: dict, *, checkpoint: bool = False) -> StreamEvent:
        if checkpoint:
            self._thread_local.checkpoint_id = self.store.save_checkpoint(state.tenant_id, state.task_id, state)
        checkpoint_id = getattr(self._thread_local, "checkpoint_id", "")
        if not checkpoint_id:
            checkpoint_id = state.active_task.checkpoint_id if state.active_task else ""
            if checkpoint_id:
                self._thread_local.checkpoint_id = checkpoint_id
        return self.store.append_event(state.tenant_id, state.task_id, checkpoint_id, event_type, payload)
