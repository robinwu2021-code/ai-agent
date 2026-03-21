"""
workspace/models.py — 工作区核心数据模型

概念层次：
    Workspace（工作区）
    └── Project（项目，工作区成员的子集）
        └── MemoryEntry（记忆条目，支持跨项目共享）

记忆范围（MemoryScope）：
    PERSONAL  — owner_id = user_id，仅本人可见
    PROJECT   — owner_id = project_id，项目成员可见
    WORKSPACE — owner_id = workspace_id，全工作区可见

跨项目共享：
    MemoryShare 表将某条 entry 授权给其他 project，
    读取时会作为第 4 层参与混合召回。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────────────────────
# 角色枚举
# ──────────────────────────────────────────────────────────────

class WorkspaceRole(str, Enum):
    ADMIN  = "admin"   # 管理工作区、项目、成员
    MEMBER = "member"  # 读写项目记忆、发起对话
    VIEWER = "viewer"  # 只读

class ProjectRole(str, Enum):
    ADMIN  = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

class MemoryScope(str, Enum):
    PERSONAL  = "personal"   # 个人私有
    PROJECT   = "project"    # 项目内共享
    WORKSPACE = "workspace"  # 全工作区共享


# ──────────────────────────────────────────────────────────────
# 成员
# ──────────────────────────────────────────────────────────────

@dataclass
class WorkspaceMember:
    user_id:   str
    role:      WorkspaceRole
    joined_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProjectMember:
    user_id:   str
    role:      ProjectRole
    joined_at: datetime = field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────
# 项目
# ──────────────────────────────────────────────────────────────

@dataclass
class Project:
    project_id:     str
    workspace_id:   str
    name:           str
    description:    str                  = ""
    members:        list[ProjectMember]  = field(default_factory=list)
    # [] = 继承工作区配置
    allowed_skills: list[str]            = field(default_factory=list)
    system_prompt:  str                  = ""
    token_budget:   int                  = 12_000
    max_steps:      int                  = 20
    metadata:       dict[str, Any]       = field(default_factory=dict)
    created_at:     datetime             = field(default_factory=datetime.utcnow)
    updated_at:     datetime             = field(default_factory=datetime.utcnow)

    def has_member(self, user_id: str) -> bool:
        return any(m.user_id == user_id for m in self.members)

    def get_member(self, user_id: str) -> ProjectMember | None:
        return next((m for m in self.members if m.user_id == user_id), None)


# ──────────────────────────────────────────────────────────────
# 工作区
# ──────────────────────────────────────────────────────────────

@dataclass
class Workspace:
    workspace_id:   str
    name:           str
    description:    str                    = ""
    members:        list[WorkspaceMember]  = field(default_factory=list)
    projects:       list[Project]          = field(default_factory=list)
    allowed_skills: list[str]              = field(default_factory=list)
    system_prompt:  str                    = ""
    token_budget:   int                    = 16_000
    max_steps:      int                    = 20
    metadata:       dict[str, Any]         = field(default_factory=dict)
    created_at:     datetime               = field(default_factory=datetime.utcnow)
    updated_at:     datetime               = field(default_factory=datetime.utcnow)

    # ── 成员操作 ──────────────────────────────────────────────
    def has_member(self, user_id: str) -> bool:
        return any(m.user_id == user_id for m in self.members)

    def get_member(self, user_id: str) -> WorkspaceMember | None:
        return next((m for m in self.members if m.user_id == user_id), None)

    def get_project(self, project_id: str) -> Project | None:
        return next((p for p in self.projects if p.project_id == project_id), None)

    # ── 记忆命名空间 key ──────────────────────────────────────
    def ws_memory_key(self) -> str:
        """工作区记忆的 owner_id。"""
        return f"ws::{self.workspace_id}"

    @staticmethod
    def proj_memory_key(workspace_id: str, project_id: str) -> str:
        """项目记忆的 owner_id。"""
        return f"proj::{workspace_id}::{project_id}"


# ──────────────────────────────────────────────────────────────
# 记忆条目（扩展版）
# ──────────────────────────────────────────────────────────────

@dataclass
class WorkspaceMemoryEntry:
    """
    数据库层面的记忆条目。

    写入规则：
      scope=PERSONAL   → owner_id = user_id
      scope=PROJECT    → owner_id = "proj::{ws_id}::{proj_id}"
      scope=WORKSPACE  → owner_id = "ws::{ws_id}"

    读取时同 user_id 过滤 owner_id，以实现范围隔离。
    """
    entry_id:     str            = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    scope:        MemoryScope    = MemoryScope.PERSONAL
    owner_id:     str            = ""    # user_id | proj::... | ws::...
    workspace_id: str | None     = None  # 统一索引列，便于工作区级查询
    project_id:   str | None     = None  # 原始项目 ID（便于审计/共享）
    author_id:    str            = ""    # 实际写入此条记忆的用户
    memory_type:  str            = "semantic"   # semantic | episodic | profile
    text:         str            = ""
    importance:   float          = 0.5
    tags:         list[str]      = field(default_factory=list)
    metadata:     dict[str, Any] = field(default_factory=dict)
    access_count: int            = 0
    created_at:   datetime       = field(default_factory=datetime.utcnow)
    accessed_at:  datetime       = field(default_factory=datetime.utcnow)
    expires_at:   datetime | None = None


# ──────────────────────────────────────────────────────────────
# 跨项目共享记录
# ──────────────────────────────────────────────────────────────

@dataclass
class MemoryShare:
    """
    将一条记忆授权给另一个项目可见。

    使用场景：
      - 项目 A 的"竞品分析报告"共享给项目 B
      - 用户主动分享个人洞察到项目
      - 工作区管理员将核心知识下发给所有项目

    permission="read"  → 目标项目只能读取
    permission="write" → 目标项目可以追加/更新（保留原始 entry_id）
    """
    entry_id:             str
    shared_to_project_id: str
    shared_by_user_id:    str
    shared_at:            datetime = field(default_factory=datetime.utcnow)
    permission:           str      = "read"   # read | write
    note:                 str      = ""       # 分享说明


# ──────────────────────────────────────────────────────────────
# 带分数的记忆条目（检索结果）
# ──────────────────────────────────────────────────────────────

@dataclass
class RankedEntry:
    """
    多层混合检索的返回单元。

    score = relevance_score × layer_weight × importance
      relevance_score: n-gram 文本相关度，0-1
      layer_weight:    personal=1.0, project=0.9, shared=0.85, workspace=0.7
      importance:      写入时标注，0-1

    context_label 用于注入 prompt 时附加来源标注，例如：
      "[个人记忆]"、"[项目记忆]"、"[共享来自:营销项目]"、"[工作区知识]"
    """
    entry:         WorkspaceMemoryEntry
    score:         float
    layer:         str    # personal | project | shared | workspace
    context_label: str


# ──────────────────────────────────────────────────────────────
# 运行时上下文（Agent 调用时的完整 workspace 状态）
# ──────────────────────────────────────────────────────────────

@dataclass
class WorkspaceContext:
    workspace:   Workspace
    project:     Project | None
    user_id:     str
    user_role:   WorkspaceRole
    # 解析后的运行时配置（已合并继承关系）
    system_prompt:  str
    allowed_skills: list[str]   # [] = 全部
    token_budget:   int
    max_steps:      int
