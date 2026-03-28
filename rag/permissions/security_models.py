"""
rag/permissions/security_models.py — 多维度安全权限数据模型

四层安全级别：
  PUBLIC       公开      — 任何人（含匿名）可访问
  INTERNAL     内部      — 认证用户 + 指定组织单元（部门/项目）
  CONFIDENTIAL 机密      — 显式点名授权用户（永久有效，手动撤销）
  PERSONAL     个人      — 仅所有者，新上传文档物理隔离到独立 collection

组织单元层级继承规则：
  - 子部门成员可读父部门文档（向上继承访问权）
  - 父部门 ADMIN 可读所有子部门文档（向下管理权）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── 枚举定义 ──────────────────────────────────────────────────────────────────

class SecurityLevel(str, Enum):
    PUBLIC       = "PUBLIC"
    INTERNAL     = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    PERSONAL     = "PERSONAL"

    def rank(self) -> int:
        return {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2, "PERSONAL": 3}[self.value]

    def __lt__(self, other: "SecurityLevel") -> bool:
        return self.rank() < other.rank()

    def __le__(self, other: "SecurityLevel") -> bool:
        return self.rank() <= other.rank()


class DocOperation(str, Enum):
    READ              = "READ"    # 可检索 / 问答
    WRITE             = "WRITE"   # 可上传 / 更新内容
    DELETE            = "DELETE"  # 可删除文档
    MANAGE_PERMISSION = "MANAGE"  # 可变更本文档权限（所有者 / 被授予）


class SystemRole(str, Enum):
    SYSTEM_ADMIN = "SYSTEM_ADMIN"  # 全局最高权限（可见所有，不含 PERSONAL）
    ORG_ADMIN    = "ORG_ADMIN"     # 所在组织单元的管理员
    MEMBER       = "MEMBER"        # 普通成员
    GUEST        = "GUEST"         # 仅可访问 PUBLIC


class OrgUnitType(str, Enum):
    DEPARTMENT = "DEPARTMENT"   # 部门（支持上下级层级）
    PROJECT    = "PROJECT"      # 项目（通常扁平，也可有子项目）


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class SecUser:
    """系统用户。"""
    user_id:    str
    name:       str
    email:      str = ""
    role:       SystemRole = SystemRole.MEMBER
    created_at: float = 0.0


@dataclass
class OrgUnit:
    """组织单元（部门 / 项目），支持树形层级。"""
    unit_id:    str
    name:       str
    unit_type:  OrgUnitType = OrgUnitType.DEPARTMENT
    parent_id:  str | None = None     # None = 根节点
    created_by: str = ""
    created_at: float = 0.0


@dataclass
class OrgMembership:
    """用户与组织单元的归属关系。"""
    user_id:   str
    unit_id:   str
    is_admin:  bool = False           # 组织单元内部管理员
    joined_at: float = 0.0


@dataclass
class DocPermission:
    """文档安全级别记录（每文档一条）。"""
    doc_id:         str
    kb_id:          str
    security_level: SecurityLevel
    owner_id:       str
    # INTERNAL 时生效：允许访问的组织单元 ID 列表
    # 空列表 = 全员内部可见
    org_unit_ids:   list[str] = field(default_factory=list)
    created_at:     float = 0.0
    updated_at:     float = 0.0
    changed_by:     str = ""


@dataclass
class DocGrant:
    """
    针对 CONFIDENTIAL 文档的显式点名授权。
    无到期时间，手动撤销生效。
    """
    grant_id:   str
    doc_id:     str
    kb_id:      str
    user_id:    str
    operation:  DocOperation        # 授予的操作类型
    granted_by: str
    created_at: float = 0.0


@dataclass
class PermAuditRecord:
    """权限变更审计日志（不可删除，永久保留）。"""
    log_id:      str
    doc_id:      str
    kb_id:       str
    action:      str                # SET_LEVEL|GRANT|REVOKE|TRANSFER_OWNER|CHANGE_ORG_UNITS
    old_value:   dict = field(default_factory=dict)
    new_value:   dict = field(default_factory=dict)
    operator_id: str = ""
    reason:      str = ""
    created_at:  float = 0.0


@dataclass
class RetrievalContext:
    """
    检索时附带的权限上下文，
    由 SecurityManager.build_retrieval_context() 生成，缓存到请求级别。
    """
    user_id:             str
    kb_id:               str
    accessible_levels:   list[SecurityLevel]   # 该用户可访问的安全级别
    accessible_unit_ids: set[str]              # INTERNAL 可访问的组织单元（含继承）
    confidential_doc_ids: set[str]             # CONFIDENTIAL 中已被显式授权的文档 ID
    personal_collection: str                   # 该用户的个人 collection 名称
    is_system_admin:     bool = False
