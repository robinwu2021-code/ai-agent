"""
tenant/manager.py — 多租户隔离管理器

功能：
  - TenantContext   租户上下文（组织级配置）
  - TenantManager   租户注册与隔离
  - 数据隔离：每个租户有独立的记忆命名空间、Skill 白名单、System Prompt
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class TenantContext:
    tenant_id:      str
    name:           str
    system_prompt:  str  = ""              # 租户专属的 System Prompt 前缀
    allowed_skills: list[str] = field(default_factory=list)   # 空=全部允许
    allowed_mcps:   list[str] = field(default_factory=list)
    max_steps:      int  = 20
    token_budget:   int  = 12_000
    daily_cost_limit: float = 10.0
    metadata:       dict = field(default_factory=dict)

    def memory_namespace(self, user_id: str) -> str:
        """生成租户隔离的记忆命名空间。"""
        return f"{self.tenant_id}::{user_id}"


class TenantManager:
    """租户注册与请求上下文绑定。"""

    def __init__(self) -> None:
        self._tenants: dict[str, TenantContext] = {}
        # 默认租户（单租户场景）
        self._tenants["default"] = TenantContext(
            tenant_id="default", name="Default",
            system_prompt="",
        )

    def register(self, ctx: TenantContext) -> None:
        self._tenants[ctx.tenant_id] = ctx
        log.info("tenant.registered", tenant_id=ctx.tenant_id, name=ctx.name)

    def get(self, tenant_id: str) -> TenantContext | None:
        return self._tenants.get(tenant_id)

    def get_or_default(self, tenant_id: str | None) -> TenantContext:
        if tenant_id and tenant_id in self._tenants:
            return self._tenants[tenant_id]
        return self._tenants["default"]

    def resolve_user_id(self, tenant_id: str, raw_user_id: str) -> str:
        """将用户 ID 带上租户前缀，实现数据隔离。"""
        if tenant_id == "default":
            return raw_user_id
        return f"{tenant_id}::{raw_user_id}"

    def filter_skills(
        self, tenant_id: str, descriptors: list[Any]
    ) -> list[Any]:
        ctx = self.get_or_default(tenant_id)
        if not ctx.allowed_skills:
            return descriptors
        return [d for d in descriptors if d.name in ctx.allowed_skills]

    def list_tenants(self) -> list[dict]:
        return [
            {"tenant_id": t.tenant_id, "name": t.name}
            for t in self._tenants.values()
        ]
