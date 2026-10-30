from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AuthContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    user_id: str
    roles: tuple[str, ...] = ("user",)

