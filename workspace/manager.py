"""
workspace/manager.py — 工作区业务逻辑管理器

职责：
  - 工作区 / 项目 / 成员 的 CRUD 及权限校验
  - 解析运行时 WorkspaceContext（含继承关系的配置合并）
  - 为 AgentContainer 提供 get_context() 入口
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog

from workspace.models import (
    MemoryScope,
    Project,
    ProjectMember,
    ProjectRole,
    Workspace,
    WorkspaceContext,
    WorkspaceMember,
    WorkspaceMemoryEntry,
    WorkspaceRole,
)
from workspace.store import WorkspaceStore

log = structlog.get_logger(__name__)


class PermissionError(Exception):
    """权限不足。"""


class WorkspaceManager:
    """
    工作区管理器，持有一个 WorkspaceStore 实例。
    所有写操作均做权限校验。
    """

    def __init__(self, store: WorkspaceStore) -> None:
        self._store = store

    # ══════════════════════════════════════════════════════════════
    # 工作区 CRUD
    # ══════════════════════════════════════════════════════════════

    async def create_workspace(
        self,
        name:           str,
        creator_id:     str,
        description:    str = "",
        allowed_skills: list[str] | None = None,
        system_prompt:  str = "",
        token_budget:   int = 16_000,
        max_steps:      int = 20,
        metadata:       dict[str, Any] | None = None,
    ) -> Workspace:
        ws = Workspace(
            workspace_id=f"ws_{uuid.uuid4().hex[:8]}",
            name=name,
            description=description,
            allowed_skills=allowed_skills or [],
            system_prompt=system_prompt,
            token_budget=token_budget,
            max_steps=max_steps,
            metadata=metadata or {},
            members=[
                WorkspaceMember(
                    user_id=creator_id,
                    role=WorkspaceRole.ADMIN,
                )
            ],
        )
        await self._store.save_workspace(ws)
        log.info("workspace.created", workspace_id=ws.workspace_id, name=name, creator=creator_id)
        return ws

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        return await self._store.get_workspace(workspace_id)

    async def list_workspaces(self, user_id: str | None = None) -> list[Workspace]:
        return await self._store.list_workspaces(user_id)

    async def update_workspace(
        self,
        workspace_id: str,
        operator_id:  str,
        **updates: Any,
    ) -> Workspace:
        ws = await self._require_workspace(workspace_id)
        self._require_role(ws, operator_id, WorkspaceRole.ADMIN)
        allowed_fields = {
            "name", "description", "allowed_skills", "system_prompt",
            "token_budget", "max_steps", "metadata",
        }
        for k, v in updates.items():
            if k in allowed_fields:
                setattr(ws, k, v)
        ws.updated_at = datetime.utcnow()
        await self._store.save_workspace(ws)
        return ws

    async def delete_workspace(self, workspace_id: str, operator_id: str) -> None:
        ws = await self._require_workspace(workspace_id)
        self._require_role(ws, operator_id, WorkspaceRole.ADMIN)
        await self._store.delete_workspace(workspace_id)
        log.info("workspace.deleted", workspace_id=workspace_id)

    # ══════════════════════════════════════════════════════════════
    # 成员管理
    # ══════════════════════════════════════════════════════════════

    async def add_member(
        self,
        workspace_id: str,
        operator_id:  str,
        user_id:      str,
        role:         WorkspaceRole = WorkspaceRole.MEMBER,
    ) -> None:
        ws = await self._require_workspace(workspace_id)
        self._require_role(ws, operator_id, WorkspaceRole.ADMIN)
        member = WorkspaceMember(user_id=user_id, role=role)
        await self._store.add_workspace_member(workspace_id, member)
        log.info("workspace.member_added", workspace_id=workspace_id, user_id=user_id, role=role)

    async def remove_member(
        self, workspace_id: str, operator_id: str, user_id: str
    ) -> None:
        ws = await self._require_workspace(workspace_id)
        self._require_role(ws, operator_id, WorkspaceRole.ADMIN)
        if user_id == operator_id:
            raise ValueError("不能移除自己")
        await self._store.remove_workspace_member(workspace_id, user_id)

    async def update_member_role(
        self,
        workspace_id: str,
        operator_id:  str,
        user_id:      str,
        new_role:     WorkspaceRole,
    ) -> None:
        ws = await self._require_workspace(workspace_id)
        self._require_role(ws, operator_id, WorkspaceRole.ADMIN)
        await self._store.update_workspace_member_role(workspace_id, user_id, new_role)

    # ══════════════════════════════════════════════════════════════
    # 项目 CRUD
    # ══════════════════════════════════════════════════════════════

    async def create_project(
        self,
        workspace_id:   str,
        creator_id:     str,
        name:           str,
        description:    str = "",
        allowed_skills: list[str] | None = None,
        system_prompt:  str = "",
        token_budget:   int = 12_000,
        max_steps:      int = 20,
        metadata:       dict[str, Any] | None = None,
    ) -> Project:
        ws = await self._require_workspace(workspace_id)
        self._require_workspace_member(ws, creator_id)

        proj = Project(
            project_id=f"proj_{uuid.uuid4().hex[:8]}",
            workspace_id=workspace_id,
            name=name,
            description=description,
            allowed_skills=allowed_skills or [],
            system_prompt=system_prompt,
            token_budget=token_budget,
            max_steps=max_steps,
            metadata=metadata or {},
            members=[
                ProjectMember(user_id=creator_id, role=ProjectRole.ADMIN)
            ],
        )
        await self._store.save_project(proj)
        log.info("project.created", project_id=proj.project_id, name=name, workspace_id=workspace_id)
        return proj

    async def get_project(self, project_id: str) -> Project | None:
        return await self._store.get_project(project_id)

    async def list_projects(self, workspace_id: str, user_id: str | None = None) -> list[Project]:
        projects = await self._store.list_projects(workspace_id)
        if user_id:
            projects = [p for p in projects if p.has_member(user_id)]
        return projects

    async def update_project(
        self,
        project_id:  str,
        operator_id: str,
        **updates:   Any,
    ) -> Project:
        proj = await self._require_project(project_id)
        self._require_project_admin(proj, operator_id)
        allowed = {
            "name", "description", "allowed_skills", "system_prompt",
            "token_budget", "max_steps", "metadata",
        }
        for k, v in updates.items():
            if k in allowed:
                setattr(proj, k, v)
        proj.updated_at = datetime.utcnow()
        await self._store.save_project(proj)
        return proj

    async def delete_project(self, project_id: str, operator_id: str) -> None:
        proj = await self._require_project(project_id)
        self._require_project_admin(proj, operator_id)
        await self._store.delete_project(project_id)

    async def add_project_member(
        self,
        project_id:   str,
        workspace_id: str,
        operator_id:  str,
        user_id:      str,
        role:         ProjectRole = ProjectRole.MEMBER,
    ) -> None:
        """新成员必须先是工作区成员。"""
        ws = await self._require_workspace(workspace_id)
        self._require_workspace_member(ws, user_id)   # 检查被加入者
        proj = await self._require_project(project_id)
        self._require_project_admin(proj, operator_id)
        await self._store.add_project_member(
            project_id, ProjectMember(user_id=user_id, role=role)
        )

    async def remove_project_member(
        self, project_id: str, operator_id: str, user_id: str
    ) -> None:
        proj = await self._require_project(project_id)
        self._require_project_admin(proj, operator_id)
        await self._store.remove_project_member(project_id, user_id)

    # ══════════════════════════════════════════════════════════════
    # 运行时上下文解析
    # ══════════════════════════════════════════════════════════════

    async def get_context(
        self,
        user_id:      str,
        workspace_id: str | None = None,
        project_id:   str | None = None,
    ) -> WorkspaceContext | None:
        """
        解析 Agent 运行时所需的完整上下文，含配置继承。

        继承规则（Project 覆盖 Workspace）：
          - system_prompt:  project != "" → 用 project；否则用 workspace
          - allowed_skills: project != [] → 用 project；否则用 workspace
          - token_budget:   取 min(project, workspace)（防止超限）
          - max_steps:      取 min(project, workspace）
        """
        if not workspace_id:
            return None

        ws = await self._store.get_workspace(workspace_id)
        if not ws:
            return None

        member = ws.get_member(user_id)
        if not member:
            log.warning("workspace.context.not_member",
                        user_id=user_id, workspace_id=workspace_id)
            return None

        proj: Project | None = None
        if project_id:
            proj = ws.get_project(project_id) or await self._store.get_project(project_id)
            if proj and not proj.has_member(user_id):
                log.warning("workspace.context.not_project_member",
                            user_id=user_id, project_id=project_id)
                proj = None   # 无项目权限则退到 workspace 级别

        # 配置合并
        if proj:
            system_prompt  = proj.system_prompt  or ws.system_prompt
            allowed_skills = proj.allowed_skills or ws.allowed_skills
            token_budget   = min(proj.token_budget, ws.token_budget)
            max_steps      = min(proj.max_steps,    ws.max_steps)
        else:
            system_prompt  = ws.system_prompt
            allowed_skills = ws.allowed_skills
            token_budget   = ws.token_budget
            max_steps      = ws.max_steps

        return WorkspaceContext(
            workspace=ws,
            project=proj,
            user_id=user_id,
            user_role=member.role,
            system_prompt=system_prompt,
            allowed_skills=allowed_skills,
            token_budget=token_budget,
            max_steps=max_steps,
        )

    # ══════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════

    async def _require_workspace(self, workspace_id: str) -> Workspace:
        ws = await self._store.get_workspace(workspace_id)
        if not ws:
            raise ValueError(f"工作区不存在: {workspace_id}")
        return ws

    async def _require_project(self, project_id: str) -> Project:
        proj = await self._store.get_project(project_id)
        if not proj:
            raise ValueError(f"项目不存在: {project_id}")
        return proj

    def _require_role(
        self, ws: Workspace, user_id: str, min_role: WorkspaceRole
    ) -> None:
        member = ws.get_member(user_id)
        if not member:
            raise PermissionError(f"用户 {user_id} 不是工作区成员")
        role_order = {
            WorkspaceRole.VIEWER: 0,
            WorkspaceRole.MEMBER: 1,
            WorkspaceRole.ADMIN:  2,
        }
        if role_order[member.role] < role_order[min_role]:
            raise PermissionError(
                f"需要 {min_role.value} 权限，当前为 {member.role.value}"
            )

    def _require_workspace_member(self, ws: Workspace, user_id: str) -> None:
        if not ws.has_member(user_id):
            raise PermissionError(f"用户 {user_id} 不是工作区成员")

    def _require_project_admin(self, proj: Project, user_id: str) -> None:
        m = proj.get_member(user_id)
        if not m or m.role != ProjectRole.ADMIN:
            raise PermissionError(
                f"需要项目 ADMIN 权限"
            )
