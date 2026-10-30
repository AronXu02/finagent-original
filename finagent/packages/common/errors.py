from __future__ import annotations


class FinAgentError(Exception):
    status_code = 500
    code = "finagent_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ForbiddenError(FinAgentError):
    status_code = 403
    code = "forbidden"


class NotFoundError(FinAgentError):
    status_code = 404
    code = "not_found"


class ToolExecutionError(FinAgentError):
    status_code = 502
    code = "tool_execution_error"


class ToolTimeoutError(ToolExecutionError):
    status_code = 504
    code = "tool_timeout"


class ExternalDependencyError(ToolExecutionError):
    status_code = 503
    code = "external_dependency_error"
