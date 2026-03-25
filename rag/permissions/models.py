"""
rag/permissions/models.py — 目录权限数据模型

目录类型：
  PUBLIC   — 公共目录，按角色控制读/写权限
  PRIVATE  — 私有目录，仅所有者和显式授权用户可见

权限角色（从低到高）：
  VIEWER  → 只读
  EDITOR  → 读 + 上传/修改
  ADMIN   → 完整控制（包含删除和权限管理）

访问规则：
  1. workspace ADMIN 可访问所有目录
  2. 公共目录：已授权角色 >= VIEWER 可读，>= EDITOR 可写
  3. 私有目录：仅 created_by 和显式被授权的用户可访问
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DirectoryType(str, Enum):
    PUBLIC  = "public"
    PRIVATE = "private"


class PermissionRole(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN  = "admin"

    def level(self) -> int:
        return {"viewer": 1, "editor": 2, "admin": 3}[self.value]

    def __ge__(self, other: "PermissionRole") -> bool:
        return self.level() >= other.level()

    def __gt__(self, other: "PermissionRole") -> bool:
        return self.level() > other.level()


class PermissionLevel(str, Enum):
    """请求所需的最低权限级别。"""
    READ  = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class Directory:
    id: str                                    # UUID
    workspace_id: str
    parent_id: str | None = None               # None = 根目录
    name: str = ""
    path: str = ""                             # 物理路径，如 /公司文档/产品手册
    dir_type: DirectoryType = DirectoryType.PUBLIC
    created_by: str = ""
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Permission:
    id: str
    workspace_id: str
    directory_id: str                          # 授权目标目录
    subject_type: str = "user"                 # user | group | role
    subject_id: str = ""                       # user_id / group_id / role_name
    role: PermissionRole = PermissionRole.VIEWER
    granted_by: str = ""
    granted_at: float = 0.0
    expires_at: float | None = None            # None = 永不过期
