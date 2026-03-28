"""
rag/permissions — 权限管理模块

包含两套权限系统：
  1. 目录权限（原有）：Directory / PermissionManager — 目录树级别的 READ/WRITE 控制
  2. 多维度安全权限（新增）：SecurityLevel / SecurityManager — 文档级别的四层安全控制
     PUBLIC / INTERNAL / CONFIDENTIAL / PERSONAL
"""
from rag.permissions.models import (
    Directory, DirectoryType, Permission, PermissionLevel, PermissionRole,
)
from rag.permissions.manager import PermissionManager

# ── 多维度安全权限（Phase 1 新增）──────────────────────────────────────────
from rag.permissions.security_models import (
    SecurityLevel,
    DocOperation,
    SystemRole,
    OrgUnitType,
    SecUser,
    OrgUnit,
    OrgMembership,
    DocPermission,
    DocGrant,
    PermAuditRecord,
    RetrievalContext,
)
from rag.permissions.security_store import SecurityStore
from rag.permissions.security_manager import SecurityManager

# ── 全局 SecurityManager 单例（由 server.py startup 初始化）────────────────
_global_security_manager: SecurityManager | None = None


def get_security_manager() -> SecurityManager:
    """
    获取全局 SecurityManager 实例。
    若未初始化则抛出 RuntimeError。
    """
    if _global_security_manager is None:
        raise RuntimeError(
            "SecurityManager 未初始化。请先调用 init_security_manager()。"
        )
    return _global_security_manager


def init_security_manager(db_path: str, personal_collection_prefix: str = "kb_personal") -> SecurityManager:
    """
    初始化全局 SecurityManager 实例（同步，适合 server.py 启动时调用）。
    SecurityStore 会在首次 _get_conn() 调用时同步建表，无需显式 await initialize()。
    """
    global _global_security_manager
    store = SecurityStore(db_path=db_path)
    # 触发同步建表（SecurityStore._get_conn() 在首次访问时创建 schema）
    store._get_conn()
    _global_security_manager = SecurityManager(
        store=store,
        personal_collection_prefix=personal_collection_prefix,
    )
    return _global_security_manager


__all__ = [
    # 原有目录权限
    "Directory", "DirectoryType", "Permission", "PermissionLevel", "PermissionRole",
    "PermissionManager",
    # 多维度安全权限
    "SecurityLevel", "DocOperation", "SystemRole", "OrgUnitType",
    "SecUser", "OrgUnit", "OrgMembership", "DocPermission", "DocGrant",
    "PermAuditRecord", "RetrievalContext",
    "SecurityStore", "SecurityManager",
    # 全局单例管理
    "get_security_manager", "init_security_manager",
]
