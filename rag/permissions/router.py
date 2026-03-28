"""
rag/permissions/router.py — 权限管理 REST API

挂载路径（server.py 注册）：
  /api/permissions/...    用户 / 组织单元管理
  /api/kb/{kb_id}/...     文档权限 / 授权 / 审计

所有写操作均从 Header X-User-Id 或 query 参数 operator_id 获取操作者。
所有文档查询端点在返回前执行二次权限过滤。
"""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from rag.permissions.security_models import (
    DocOperation, OrgUnitType, SecurityLevel, SystemRole,
)
from rag.permissions.security_manager import SecurityManager

log = structlog.get_logger(__name__)

router = APIRouter(tags=["permissions"])

# ── 依赖注入（生产环境替换为真实的鉴权中间件）────────────────────────────────


def _get_manager() -> SecurityManager:
    """FastAPI 依赖注入：获取全局 SecurityManager 实例。"""
    from rag.permissions import _global_security_manager
    if _global_security_manager is None:
        raise HTTPException(503, "权限管理模块未初始化")
    return _global_security_manager


def _current_user(x_user_id: str = Header("anonymous")) -> str:
    return x_user_id


# ── Pydantic 请求/响应模型 ────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    user_id: str
    name: str
    email: str = ""
    role: str = "MEMBER"


class UpdateUserRoleRequest(BaseModel):
    role: str
    operator_id: str


class CreateOrgUnitRequest(BaseModel):
    name: str
    unit_type: str = "DEPARTMENT"
    parent_id: str | None = None


class AddMemberRequest(BaseModel):
    user_id: str
    is_admin: bool = False


class SetDocPermissionRequest(BaseModel):
    security_level: str
    owner_id: str
    org_unit_ids: list[str] = []
    operator_id: str = ""
    reason: str = ""


class ChangeSecurityLevelRequest(BaseModel):
    new_level: str
    new_org_unit_ids: list[str] | None = None
    operator_id: str
    reason: str = ""


class ChangeOrgUnitsRequest(BaseModel):
    org_unit_ids: list[str]
    operator_id: str
    reason: str = ""


class TransferOwnerRequest(BaseModel):
    new_owner_id: str
    operator_id: str
    reason: str = ""


class GrantRequest(BaseModel):
    user_id: str
    operation: str = "READ"
    granted_by: str


class RevokeRequest(BaseModel):
    user_id: str
    operation: str | None = None
    operator_id: str
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# 用户管理
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/api/permissions/users", summary="创建用户")
async def create_user(
    body: CreateUserRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        role = SystemRole(body.role)
    except ValueError:
        raise HTTPException(400, f"无效角色：{body.role}")
    user = await mgr.create_user(body.user_id, body.name, body.email, role)
    return {"user_id": user.user_id, "name": user.name, "role": user.role.value}


@router.get("/api/permissions/users/{user_id}", summary="查询用户")
async def get_user(
    user_id: str,
    mgr: SecurityManager = Depends(_get_manager),
):
    user = await mgr.get_user(user_id)
    if not user:
        raise HTTPException(404, "用户不存在")
    return {"user_id": user.user_id, "name": user.name,
            "email": user.email, "role": user.role.value}


@router.put("/api/permissions/users/{user_id}/role", summary="变更用户角色")
async def update_user_role(
    user_id: str,
    body: UpdateUserRoleRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        role = SystemRole(body.role)
    except ValueError:
        raise HTTPException(400, f"无效角色：{body.role}")
    await mgr.set_user_role(user_id, role, operator_id=body.operator_id)
    return {"ok": True}


@router.get("/api/permissions/users", summary="用户列表")
async def list_users(mgr: SecurityManager = Depends(_get_manager)):
    users = await mgr._store.list_users()
    return [{"user_id": u.user_id, "name": u.name, "role": u.role.value} for u in users]


# ═══════════════════════════════════════════════════════════════════════════════
# 组织单元管理
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/api/permissions/org-units", summary="创建组织单元")
async def create_org_unit(
    body: CreateOrgUnitRequest,
    operator: str = Depends(_current_user),
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        unit_type = OrgUnitType(body.unit_type)
    except ValueError:
        raise HTTPException(400, f"无效类型：{body.unit_type}")
    unit = await mgr.create_org_unit(
        body.name, unit_type, body.parent_id, created_by=operator
    )
    return {"unit_id": unit.unit_id, "name": unit.name,
            "type": unit.unit_type.value, "parent_id": unit.parent_id}


@router.get("/api/permissions/org-units", summary="组织单元列表")
async def list_org_units(
    unit_type: str | None = Query(None),
    mgr: SecurityManager = Depends(_get_manager),
):
    ut = OrgUnitType(unit_type) if unit_type else None
    units = await mgr.list_org_units(ut)
    return [{"unit_id": u.unit_id, "name": u.name,
             "type": u.unit_type.value, "parent_id": u.parent_id} for u in units]


@router.get("/api/permissions/org-units/{unit_id}/members", summary="组织单元成员列表")
async def list_unit_members(
    unit_id: str,
    mgr: SecurityManager = Depends(_get_manager),
):
    members = await mgr._store.get_unit_members(unit_id)
    return [{"user_id": m.user_id, "is_admin": m.is_admin} for m in members]


@router.post("/api/permissions/org-units/{unit_id}/members", summary="添加成员")
async def add_member(
    unit_id: str,
    body: AddMemberRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    await mgr.add_member(unit_id, body.user_id, body.is_admin)
    return {"ok": True}


@router.delete("/api/permissions/org-units/{unit_id}/members/{user_id}",
               summary="移除成员")
async def remove_member(
    unit_id: str,
    user_id: str,
    mgr: SecurityManager = Depends(_get_manager),
):
    await mgr.remove_member(unit_id, user_id)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════════════
# 文档权限管理
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/kb/{kb_id}/documents/{doc_id}/permissions",
            summary="查看文档权限")
async def get_doc_permission(
    kb_id: str,
    doc_id: str,
    caller: str = Depends(_current_user),
    mgr: SecurityManager = Depends(_get_manager),
):
    # 需要有读权限才能查看权限详情
    if not await mgr.can_access(caller, doc_id, kb_id, DocOperation.READ):
        raise HTTPException(403, "无权查看此文档权限信息")
    perm = await mgr._store.get_doc_permission(doc_id, kb_id)
    if not perm:
        raise HTTPException(404, "未找到文档权限记录")
    grants = await mgr.list_grants(doc_id, kb_id)
    return {
        "doc_id": perm.doc_id,
        "kb_id": perm.kb_id,
        "security_level": perm.security_level.value,
        "owner_id": perm.owner_id,
        "org_unit_ids": perm.org_unit_ids,
        "updated_at": perm.updated_at,
        "changed_by": perm.changed_by,
        "grants": [
            {"user_id": g.user_id, "operation": g.operation.value,
             "granted_by": g.granted_by, "created_at": g.created_at}
            for g in grants
        ],
    }


@router.post("/api/kb/{kb_id}/documents/{doc_id}/permissions",
             summary="设置文档权限（初始化）")
async def set_doc_permission(
    kb_id: str,
    doc_id: str,
    body: SetDocPermissionRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        level = SecurityLevel(body.security_level)
    except ValueError:
        raise HTTPException(400, f"无效安全级别：{body.security_level}")
    perm = await mgr.set_document_permission(
        doc_id=doc_id, kb_id=kb_id,
        security_level=level, owner_id=body.owner_id,
        org_unit_ids=body.org_unit_ids,
        operator_id=body.operator_id, reason=body.reason,
    )
    return {"ok": True, "security_level": perm.security_level.value}


@router.put("/api/kb/{kb_id}/documents/{doc_id}/security-level",
            summary="变更文档安全级别")
async def change_security_level(
    kb_id: str,
    doc_id: str,
    body: ChangeSecurityLevelRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        new_level = SecurityLevel(body.new_level)
    except ValueError:
        raise HTTPException(400, f"无效安全级别：{body.new_level}")
    try:
        await mgr.change_security_level(
            doc_id=doc_id, kb_id=kb_id,
            new_level=new_level, operator_id=body.operator_id,
            new_org_unit_ids=body.new_org_unit_ids,
            reason=body.reason,
        )
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True, "new_level": new_level.value}


@router.put("/api/kb/{kb_id}/documents/{doc_id}/org-units",
            summary="变更内部文档的组织单元")
async def change_org_units(
    kb_id: str,
    doc_id: str,
    body: ChangeOrgUnitsRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        await mgr.change_org_units(
            doc_id=doc_id, kb_id=kb_id,
            org_unit_ids=body.org_unit_ids,
            operator_id=body.operator_id, reason=body.reason,
        )
    except (PermissionError, ValueError) as e:
        raise HTTPException(403, str(e))
    return {"ok": True}


@router.put("/api/kb/{kb_id}/documents/{doc_id}/owner",
            summary="转移文档所有权（SYSTEM_ADMIN）")
async def transfer_owner(
    kb_id: str,
    doc_id: str,
    body: TransferOwnerRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        await mgr.transfer_ownership(
            doc_id=doc_id, kb_id=kb_id,
            new_owner_id=body.new_owner_id,
            operator_id=body.operator_id, reason=body.reason,
        )
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════════════
# 显式授权（CONFIDENTIAL 点名授权）
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/api/kb/{kb_id}/documents/{doc_id}/grants",
             summary="添加显式授权")
async def add_grant(
    kb_id: str,
    doc_id: str,
    body: GrantRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    try:
        op = DocOperation(body.operation)
    except ValueError:
        raise HTTPException(400, f"无效操作类型：{body.operation}")
    try:
        g = await mgr.grant(
            doc_id=doc_id, kb_id=kb_id,
            user_id=body.user_id, operation=op,
            granted_by=body.granted_by,
        )
    except (ValueError, PermissionError) as e:
        raise HTTPException(400, str(e))
    return {"grant_id": g.grant_id, "user_id": g.user_id,
            "operation": g.operation.value}


@router.delete("/api/kb/{kb_id}/documents/{doc_id}/grants",
               summary="撤销显式授权")
async def revoke_grant(
    kb_id: str,
    doc_id: str,
    body: RevokeRequest,
    mgr: SecurityManager = Depends(_get_manager),
):
    op = None
    if body.operation:
        try:
            op = DocOperation(body.operation)
        except ValueError:
            raise HTTPException(400, f"无效操作类型：{body.operation}")
    try:
        await mgr.revoke(
            doc_id=doc_id, kb_id=kb_id,
            user_id=body.user_id, operator_id=body.operator_id,
            operation=op, reason=body.reason,
        )
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"ok": True}


@router.get("/api/kb/{kb_id}/documents/{doc_id}/grants",
            summary="查看授权列表")
async def list_grants(
    kb_id: str,
    doc_id: str,
    caller: str = Depends(_current_user),
    mgr: SecurityManager = Depends(_get_manager),
):
    if not await mgr.can_access(caller, doc_id, kb_id, DocOperation.MANAGE_PERMISSION):
        raise HTTPException(403, "无权查看授权列表")
    grants = await mgr.list_grants(doc_id, kb_id)
    return [{"user_id": g.user_id, "operation": g.operation.value,
             "granted_by": g.granted_by} for g in grants]


# ═══════════════════════════════════════════════════════════════════════════════
# 审计日志
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/kb/{kb_id}/permission-audit", summary="权限变更审计日志")
async def get_audit_log(
    kb_id: str,
    doc_id: str | None = Query(None),
    operator_id: str | None = Query(None),
    limit: int = Query(50, le=200),
    caller: str = Depends(_current_user),
    mgr: SecurityManager = Depends(_get_manager),
):
    # 只有 SYSTEM_ADMIN 可查看全库审计日志
    user = await mgr.get_user(caller)
    if doc_id:
        # 查看单文档审计：需要 MANAGE 权限
        if not await mgr.can_access(caller, doc_id, kb_id, DocOperation.MANAGE_PERMISSION):
            raise HTTPException(403, "无权查看此文档审计日志")
    else:
        if not (user and user.role == SystemRole.SYSTEM_ADMIN):
            raise HTTPException(403, "全库审计日志需要 SYSTEM_ADMIN 权限")

    records = await mgr.get_audit_log(
        kb_id=kb_id, doc_id=doc_id,
        operator_id=operator_id, limit=limit,
    )
    return [
        {
            "log_id": r.log_id, "doc_id": r.doc_id, "action": r.action,
            "old_value": r.old_value, "new_value": r.new_value,
            "operator_id": r.operator_id, "reason": r.reason,
            "created_at": r.created_at,
        }
        for r in records
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 用户视角：我能访问的文档
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/kb/{kb_id}/my-accessible-docs",
            summary="我可以访问的文档列表（含安全级别）")
async def my_accessible_docs(
    kb_id: str,
    caller: str = Depends(_current_user),
    mgr: SecurityManager = Depends(_get_manager),
):
    all_perms = await mgr._store.list_doc_permissions_by_kb(kb_id)
    user = await mgr._store.get_user(caller)
    result = []
    for perm in all_perms:
        if await mgr._check_perm(user, perm, DocOperation.READ):
            result.append({
                "doc_id": perm.doc_id,
                "security_level": perm.security_level.value,
                "owner_id": perm.owner_id,
                "org_unit_ids": perm.org_unit_ids,
            })
    return {"total": len(result), "docs": result}
