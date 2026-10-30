from __future__ import annotations

from typing import Any

from finagent.packages.orchestrator.models import AgentState, ToolCall, ToolError, ToolResult
from finagent.packages.tools.executor import ToolExecutor
from finagent.packages.tools.registry import ToolContext


def execute_tool(
    *,
    state: AgentState,
    executor: ToolExecutor,
    context: ToolContext,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolResult:
    call = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        idempotency_key=f"{state.task_id}:{tool_name}:{len(state.pending_tools) + 1}",
    )
    state.pending_tools.append(call)
    result = executor.execute(call, context)
    state.tool_results[call.tool_call_id] = result
    if result.status == "error":
        state.tool_errors.append(
            ToolError(
                tool_call_id=call.tool_call_id,
                tool_name=tool_name,
                error_code="tool_error",
                message=result.error or "工具执行失败",
                retryable=False,
            )
        )
    return result
