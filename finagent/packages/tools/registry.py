from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from finagent.packages.common.errors import NotFoundError


ToolHandler = Callable[[BaseModel, "ToolContext"], dict[str, Any]]


@dataclass(frozen=True)
class ToolContext:
    tenant_id: str
    user_id: str
    task_id: str
    trace_id: str
    account_scope: str | None = None
    risk_level: str = "balanced"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    timeout_seconds: float = 3.0
    max_retries: int = 1
    fallback_tool: str | None = None
    permissions: tuple[str, ...] = ("user",)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise NotFoundError(f"工具未注册：{name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)
