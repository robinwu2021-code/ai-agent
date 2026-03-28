"""
rag/permissions/security_manager.py — 多维度安全权限核心逻辑

职责：
  1. 权限检查（can_access / check_operation）
  2. 组织单元层级继承计算
  3. 检索过滤条件构建（build_retrieval_context）
  4. 检索结果二次过滤（filter_chunks）
  5. 文档权限设置 / 安全级别变更
  6. 显式授权管理（grant / revoke）
  7. 审计日志

层级继承规则：
  成员规则：用户在 A 部门 → 可读 A 的所有父部门文档（向上继承读权）
  管理员规则：用户是 A 部门 ADMIN → 可读 A 的所有子部门文档（向下管理权）
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from rag.permissions.security_models import (
    DocGrant, DocOperation, DocPermission, OrgMembership, OrgUnit,
    OrgUnitType, PermAuditRecord, RetrievalContext, SecUser,
    SecurityLevel, SystemRole,
)
from rag.permissions.security_store import SecurityStore

log = structlog.get_logger(__name__)

# 安全级别降级需要 SYSTEM_ADMIN（防止信息意外扩散）
_DOWNGRADE_REQUIRES_ADMIN = {
    (SecurityLevel.INTERNAL,     SecurityLevel.PUBLIC),
    (SecurityLevel.CONFIDENTIAL, SecurityLevel.PUBLIC),
    (SecurityLevel.CONFIDENTIAL, SecurityLevel.INTERNAL),
}


class SecurityManager:
    """多维度权限管理器。"""

    def __init__(
        self,
        store: SecurityStore,
        personal_collection_prefix: str = "kb_personal",
    ) -> None:
        self._store = store
        self._prefix = personal_collection_prefix

    @classmethod
    def create(
        cls,
        db_path: str = "./data/workspace.db",
        personal_collection_prefix: str = "kb_personal",
    ) -> "SecurityManager":
        store = SecurityStore(db_path)
        return cls(store, personal_collection_prefix)

    async def initialize(self) -> None:
        await self._store.initialize()

    # ── 用户管理 ──────────────────────────────────────────────────────────────

    async def create_user(
        self, user_id: str, name: str,
        email: str = "", role: SystemRole = SystemRole.MEMBER,
    ) -> SecUser:
        user = SecUser(user_id=user_id, name=name, email=email,
                       role=role, created_at=time.time())
        await self._store.upsert_user(user)
        log.info("security.user_created", user_id=user_id, role=role.value)
        return user

    async def get_user(self, user_id: str) -> SecUser | None:
        return await self._store.get_user(user_id)

    async def set_user_role(self, user_id: str, role: SystemRole,
                            operator_id: str = "") -> None:
        await self._store.update_user_role(user_id, role)
        log.info("security.user_role_changed",
                 user_id=user_id, role=role.value, by=operator_id)

    # ── 组织单元管理 ──────────────────────────────────────────────────────────

    async def create_org_unit(
        self, name: str,
        unit_type: OrgUnitType = OrgUnitType.DEPARTMENT,
        parent_id: str | None = None,
        created_by: str = "",
    ) -> OrgUnit:
        unit = OrgUnit(
            unit_id=uuid.uuid4().hex, name=name,
            unit_type=unit_type, parent_id=parent_id,
            created_by=created_by, created_at=time.time(),
        )
        await self._store.upsert_org_unit(unit)
        log.info("security.org_unit_created",
                 unit_id=unit.unit_id, name=name, type=unit_type.value)
        return unit

    async def add_member(
        self, unit_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        await self._store.add_member(user_id, unit_id, is_admin)
        log.info("security.member_added",
                 unit_id=unit_id, user_id=user_id, is_admin=is_admin)

    async def remove_member(self, unit_id: str, user_id: str) -> None:
        await self._store.remove_member(user_id, unit_id)

    async def list_org_units(
        self, unit_type: OrgUnitType | None = None
    ) -> list[OrgUnit]:
        return await self._store.list_org_units(unit_type)

    # ── 层级继承：计算用户可访问的组织单元集合 ────────────────────────────────

    async def _effective_unit_ids(self, user_id: str) -> set[str]:
        """
        计算用户可访问的所有组织单元 ID（含继承）。

        规则：
          ① 用户直属的所有单元
          ② 每个直属单元的所有祖先（子部门成员可读父部门文档）
          ③ 如果用户是某单元的 ADMIN：该单元的所有后代（父部门管理员可读子部门文档）
        """
        memberships = await self._store.get_user_memberships(user_id)
        effective: set[str] = set()

        for m in memberships:
            effective.add(m.unit_id)
            # ① → ② 向上继承（读权）
            ancestors = await self._store.get_ancestors(m.unit_id)
            effective.update(ancestors)
            # ① → ③ 向下管理权（仅 ADMIN）
            if m.is_admin:
                descendants = await self._store.get_descendants(m.unit_id)
                effective.update(descendants)

        return effective

    # ── 权限检查 ──────────────────────────────────────────────────────────────

    async def can_access(
        self,
        user_id: str,
        doc_id: str,
        kb_id: str,
        operation: DocOperation = DocOperation.READ,
    ) -> bool:
        """
        检查用户是否可对指定文档执行操作。
        这是最细粒度的权限检查，用于单文档操作和输出过滤。
        """
        perm = await self._store.get_doc_permission(doc_id, kb_id)
        if perm is None:
            # 无权限记录：默认 PUBLIC（兼容旧文档）
            return True

        user = await self._store.get_user(user_id)
        return await self._check_perm(user, perm, operation)

    async def _check_perm(
        self,
        user: SecUser | None,
        perm: DocPermission,
        operation: DocOperation,
    ) -> bool:
        level = perm.security_level

        # ── PUBLIC：任何人可读 ─────────────────────────────────────────────
        if level == SecurityLevel.PUBLIC:
            if operation == DocOperation.READ:
                return True
            # 写/删/管理：需要认证用户
            if user is None:
                return False
            if user.role == SystemRole.SYSTEM_ADMIN:
                return True
            return (user.user_id == perm.owner_id or
                    await self._has_explicit_grant(user.user_id, perm.doc_id, perm.kb_id, operation))

        # 以下都需要认证用户
        if user is None:
            return False

        # ── SYSTEM_ADMIN：可访问所有（除 PERSONAL 文档只读）─────────────────
        if user.role == SystemRole.SYSTEM_ADMIN:
            if level == SecurityLevel.PERSONAL:
                # 系统管理员可审计 PERSONAL，但不能通过检索访问
                return operation == DocOperation.READ and user.user_id == perm.owner_id
            return True

        # GUEST 只能访问 PUBLIC（已在上面处理）
        if user.role == SystemRole.GUEST:
            return False

        # ── PERSONAL：仅所有者 ────────────────────────────────────────────
        if level == SecurityLevel.PERSONAL:
            return user.user_id == perm.owner_id

        # ── CONFIDENTIAL：所有者 + 显式授权 ──────────────────────────────
        if level == SecurityLevel.CONFIDENTIAL:
            if user.user_id == perm.owner_id:
                return True
            return await self._has_explicit_grant(user.user_id, perm.doc_id, perm.kb_id, operation)

        # ── INTERNAL：所有者 + 组织单元成员（含层级继承）────────────────────
        if level == SecurityLevel.INTERNAL:
            if user.user_id == perm.owner_id:
                return True
            # 无指定组织单元 = 全体内部成员可读
            if operation == DocOperation.READ and not perm.org_unit_ids:
                return True
            effective = await self._effective_unit_ids(user.user_id)
            if effective & set(perm.org_unit_ids):
                return True
            # 写/删/管理：需要显式授权
            if operation != DocOperation.READ:
                return await self._has_explicit_grant(
                    user.user_id, perm.doc_id, perm.kb_id, operation
                )
            return False

        return False

    async def _has_explicit_grant(
        self, user_id: str, doc_id: str, kb_id: str, operation: DocOperation
    ) -> bool:
        grants = await self._store.get_grants(doc_id, kb_id)
        return any(g.user_id == user_id and g.operation == operation for g in grants)

    # ── 检索上下文（批量检索前调用一次，避免重复查询）────────────────────────

    async def build_retrieval_context(
        self, user_id: str, kb_id: str
    ) -> RetrievalContext:
        """
        构建检索权限上下文（每次 query 调用一次）。
        返回结果可传入 filter_chunks() 做二次过滤。
        """
        user = await self._store.get_user(user_id)

        if user is None:
            # 匿名用户：仅 PUBLIC
            return RetrievalContext(
                user_id=user_id, kb_id=kb_id,
                accessible_levels=[SecurityLevel.PUBLIC],
                accessible_unit_ids=set(),
                confidential_doc_ids=set(),
                personal_collection="",
                is_system_admin=False,
            )

        if user.role == SystemRole.GUEST:
            return RetrievalContext(
                user_id=user_id, kb_id=kb_id,
                accessible_levels=[SecurityLevel.PUBLIC],
                accessible_unit_ids=set(),
                confidential_doc_ids=set(),
                personal_collection="",
                is_system_admin=False,
            )

        is_admin = (user.role == SystemRole.SYSTEM_ADMIN)

        # INTERNAL 可访问的组织单元
        unit_ids = await self._effective_unit_ids(user_id) if not is_admin else set()

        # CONFIDENTIAL 已被授权的文档 ID
        conf_doc_ids = await self._store.get_user_confidential_doc_ids(user_id, kb_id)

        accessible = [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL]
        if conf_doc_ids or is_admin:
            accessible.append(SecurityLevel.CONFIDENTIAL)

        return RetrievalContext(
            user_id=user_id,
            kb_id=kb_id,
            accessible_levels=accessible,
            accessible_unit_ids=unit_ids,
            confidential_doc_ids=set(conf_doc_ids),
            personal_collection=self._personal_collection(user_id, kb_id),
            is_system_admin=is_admin,
        )

    def _personal_collection(self, user_id: str, kb_id: str) -> str:
        safe_uid = user_id.replace("/", "_").replace(" ", "_")[:32]
        return f"{self._prefix}_{safe_uid}_{kb_id}"

    def build_vector_filter(self, ctx: RetrievalContext) -> dict:
        """
        生成向量检索预过滤条件（传给 Milvus / Qdrant）。

        策略：宽松预过滤（level 级别），组织单元匹配在应用层做。
        返回的 dict 供 vector_store 适配器解析。
        """
        levels = [lv.value for lv in ctx.accessible_levels]
        conditions: list[dict] = []

        # PUBLIC
        if SecurityLevel.PUBLIC.value in levels:
            conditions.append({"security_level": "PUBLIC"})

        # INTERNAL（先粗过滤，应用层再精细匹配 org_units）
        if SecurityLevel.INTERNAL.value in levels:
            conditions.append({"security_level": "INTERNAL"})

        # CONFIDENTIAL（仅已授权的 doc_id）
        if ctx.confidential_doc_ids:
            conditions.append({
                "security_level": "CONFIDENTIAL",
                "doc_id_in": list(ctx.confidential_doc_ids),
            })
        elif ctx.is_system_admin:
            conditions.append({"security_level": "CONFIDENTIAL"})

        # PERSONAL：由独立 collection 处理，不在共享 collection 过滤
        return {"or": conditions} if conditions else {"security_level": "PUBLIC"}

    # ── 输出二次过滤（检索结果返回前调用）────────────────────────────────────

    @staticmethod
    def _chunk_attr(chunk: Any, key: str, default: str = "") -> str:
        """兼容 dict 和 dataclass 两种 chunk 格式。"""
        if isinstance(chunk, dict):
            return chunk.get(key, default)
        return getattr(chunk, key, default) or default

    async def filter_chunks(
        self,
        chunks: list,
        ctx: RetrievalContext,
    ) -> list:
        """
        对检索结果逐条验证权限，过滤掉无权访问的 chunk。

        chunks 可以是 dict 列表，也可以是带 doc_id / kb_id 属性的 dataclass 列表
        （如 RetrievedChunk）。
        """
        if not chunks:
            return []

        result = []
        for chunk in chunks:
            doc_id = self._chunk_attr(chunk, "doc_id")
            kb_id  = self._chunk_attr(chunk, "kb_id") or ctx.kb_id
            if not doc_id:
                result.append(chunk)   # 无 doc_id 的 chunk 直接放行
                continue

            perm = await self._store.get_doc_permission(doc_id, kb_id)
            if perm is None:
                result.append(chunk)   # 无权限记录 → 视为 PUBLIC
                continue

            level = perm.security_level

            # PUBLIC
            if level == SecurityLevel.PUBLIC:
                result.append(chunk); continue

            # PERSONAL：仅所有者
            if level == SecurityLevel.PERSONAL:
                if perm.owner_id == ctx.user_id:
                    result.append(chunk)
                continue

            # CONFIDENTIAL：已授权
            if level == SecurityLevel.CONFIDENTIAL:
                if (ctx.is_system_admin or
                        perm.owner_id == ctx.user_id or
                        doc_id in ctx.confidential_doc_ids):
                    result.append(chunk)
                continue

            # INTERNAL：组织单元匹配（应用层精细过滤）
            if level == SecurityLevel.INTERNAL:
                if ctx.is_system_admin or perm.owner_id == ctx.user_id:
                    result.append(chunk); continue
                if not perm.org_unit_ids:
                    result.append(chunk); continue
                if ctx.accessible_unit_ids & set(perm.org_unit_ids):
                    result.append(chunk); continue

        filtered_count = len(chunks) - len(result)
        if filtered_count:
            log.debug("security.chunks_filtered",
                      total=len(chunks), passed=len(result),
                      filtered=filtered_count, user=ctx.user_id)
        return result

    # ── 文档权限设置 ──────────────────────────────────────────────────────────

    async def set_document_permission(
        self,
        doc_id: str,
        kb_id: str,
        security_level: SecurityLevel,
        owner_id: str,
        org_unit_ids: list[str] | None = None,
        operator_id: str = "",
        reason: str = "",
    ) -> DocPermission:
        """初始化或覆盖文档权限记录（通常在上传时调用）。"""
        now = time.time()
        old = await self._store.get_doc_permission(doc_id, kb_id)
        perm = DocPermission(
            doc_id=doc_id, kb_id=kb_id,
            security_level=security_level,
            owner_id=owner_id,
            org_unit_ids=org_unit_ids or [],
            created_at=old.created_at if old else now,
            updated_at=now,
            changed_by=operator_id,
        )
        await self._store.upsert_doc_permission(perm)
        await self._audit(doc_id, kb_id, "SET_LEVEL",
                          old_value={"level": old.security_level.value} if old else {},
                          new_value={"level": security_level.value,
                                     "org_units": perm.org_unit_ids},
                          operator_id=operator_id, reason=reason)
        log.info("security.doc_permission_set",
                 doc_id=doc_id, level=security_level.value, kb=kb_id)
        return perm

    async def change_security_level(
        self,
        doc_id: str,
        kb_id: str,
        new_level: SecurityLevel,
        operator_id: str,
        new_org_unit_ids: list[str] | None = None,
        reason: str = "",
    ) -> None:
        """
        变更文档安全级别。

        降级规则（提高可见性 = 安全风险）：
          INTERNAL→PUBLIC, CONFIDENTIAL→PUBLIC/INTERNAL 需要 SYSTEM_ADMIN
          其余变更：文档所有者或 SYSTEM_ADMIN

        PERSONAL 物理隔离说明：
          新上传 PERSONAL 文档自动写入个人 collection（由 KBManager 处理）。
          此处 change_security_level 仅更新元数据，不迁移向量（逻辑隔离）。
          若需完整物理迁移，请调用 KBManager.migrate_document_collection()。
        """
        perm = await self._store.get_doc_permission(doc_id, kb_id)
        if perm is None:
            raise ValueError(f"文档 {doc_id} 在 {kb_id} 中无权限记录")

        operator = await self._store.get_user(operator_id)

        # 权限校验
        is_sys_admin = operator and operator.role == SystemRole.SYSTEM_ADMIN
        is_owner     = perm.owner_id == operator_id
        transition   = (perm.security_level, new_level)

        if transition in _DOWNGRADE_REQUIRES_ADMIN and not is_sys_admin:
            raise PermissionError(
                f"安全级别降级（{perm.security_level.value}→{new_level.value}）"
                "需要 SYSTEM_ADMIN 权限"
            )
        if not is_owner and not is_sys_admin:
            raise PermissionError("只有文档所有者或系统管理员可变更安全级别")

        old_level = perm.security_level
        perm.security_level = new_level
        perm.org_unit_ids   = new_org_unit_ids if new_org_unit_ids is not None else perm.org_unit_ids
        perm.updated_at     = time.time()
        perm.changed_by     = operator_id
        await self._store.upsert_doc_permission(perm)
        await self._audit(doc_id, kb_id, "SET_LEVEL",
                          old_value={"level": old_level.value},
                          new_value={"level": new_level.value,
                                     "org_units": perm.org_unit_ids},
                          operator_id=operator_id, reason=reason)
        log.info("security.level_changed",
                 doc_id=doc_id, old=old_level.value, new=new_level.value, by=operator_id)

    async def change_org_units(
        self,
        doc_id: str,
        kb_id: str,
        org_unit_ids: list[str],
        operator_id: str,
        reason: str = "",
    ) -> None:
        """变更 INTERNAL 文档的组织单元列表。"""
        perm = await self._store.get_doc_permission(doc_id, kb_id)
        if perm is None:
            raise ValueError(f"文档 {doc_id} 在 {kb_id} 中无权限记录")
        if perm.security_level != SecurityLevel.INTERNAL:
            raise ValueError("只有 INTERNAL 文档才有组织单元设置")
        operator = await self._store.get_user(operator_id)
        is_sys_admin = operator and operator.role == SystemRole.SYSTEM_ADMIN
        if perm.owner_id != operator_id and not is_sys_admin:
            raise PermissionError("只有文档所有者或系统管理员可变更组织单元")

        old_units = perm.org_unit_ids
        perm.org_unit_ids = org_unit_ids
        perm.updated_at   = time.time()
        perm.changed_by   = operator_id
        await self._store.upsert_doc_permission(perm)
        await self._audit(doc_id, kb_id, "CHANGE_ORG_UNITS",
                          old_value={"org_units": old_units},
                          new_value={"org_units": org_unit_ids},
                          operator_id=operator_id, reason=reason)

    async def transfer_ownership(
        self,
        doc_id: str,
        kb_id: str,
        new_owner_id: str,
        operator_id: str,
        reason: str = "",
    ) -> None:
        """转移文档所有权（仅 SYSTEM_ADMIN）。"""
        operator = await self._store.get_user(operator_id)
        if not (operator and operator.role == SystemRole.SYSTEM_ADMIN):
            raise PermissionError("所有权转移需要 SYSTEM_ADMIN 权限")
        perm = await self._store.get_doc_permission(doc_id, kb_id)
        if perm is None:
            raise ValueError(f"文档 {doc_id} 无权限记录")
        old_owner = perm.owner_id
        perm.owner_id   = new_owner_id
        perm.updated_at = time.time()
        perm.changed_by = operator_id
        await self._store.upsert_doc_permission(perm)
        await self._audit(doc_id, kb_id, "TRANSFER_OWNER",
                          old_value={"owner": old_owner},
                          new_value={"owner": new_owner_id},
                          operator_id=operator_id, reason=reason)

    # ── 显式授权（CONFIDENTIAL 点名授权）────────────────────────────────────

    async def grant(
        self,
        doc_id: str,
        kb_id: str,
        user_id: str,
        operation: DocOperation,
        granted_by: str,
    ) -> DocGrant:
        """向指定用户授予操作权限（无到期，手动撤销）。"""
        perm = await self._store.get_doc_permission(doc_id, kb_id)
        if perm and perm.security_level == SecurityLevel.PUBLIC:
            log.warning("security.grant_on_public_doc",
                        doc_id=doc_id, hint="PUBLIC 文档无需显式授权")

        grantee = await self._store.get_user(user_id)
        grantor = await self._store.get_user(granted_by)
        if grantee is None:
            raise ValueError(f"用户 {user_id} 不存在")

        grant_doc = DocGrant(
            grant_id=uuid.uuid4().hex,
            doc_id=doc_id, kb_id=kb_id,
            user_id=user_id, operation=operation,
            granted_by=granted_by, created_at=time.time(),
        )
        await self._store.add_grant(grant_doc)
        await self._audit(doc_id, kb_id, "GRANT",
                          new_value={"user": user_id, "operation": operation.value},
                          operator_id=granted_by)
        log.info("security.grant_added",
                 doc_id=doc_id, user=user_id, op=operation.value, by=granted_by)
        return grant_doc

    async def revoke(
        self,
        doc_id: str,
        kb_id: str,
        user_id: str,
        operator_id: str,
        operation: DocOperation | None = None,
        reason: str = "",
    ) -> None:
        """撤销用户授权（可指定单个操作，或撤销该用户的全部授权）。"""
        await self._store.remove_grant(doc_id, kb_id, user_id, operation)
        await self._audit(doc_id, kb_id, "REVOKE",
                          old_value={"user": user_id,
                                     "operation": operation.value if operation else "ALL"},
                          operator_id=operator_id, reason=reason)
        log.info("security.grant_revoked",
                 doc_id=doc_id, user=user_id, op=str(operation), by=operator_id)

    async def list_grants(self, doc_id: str, kb_id: str) -> list[DocGrant]:
        return await self._store.get_grants(doc_id, kb_id)

    # ── 审计查询 ──────────────────────────────────────────────────────────────

    async def get_audit_log(
        self,
        kb_id: str,
        doc_id: str | None = None,
        operator_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[PermAuditRecord]:
        return await self._store.query_audit(
            kb_id=kb_id, doc_id=doc_id,
            operator_id=operator_id, since=since, limit=limit,
        )

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    async def _audit(
        self, doc_id: str, kb_id: str, action: str,
        old_value: dict | None = None, new_value: dict | None = None,
        operator_id: str = "", reason: str = "",
    ) -> None:
        record = PermAuditRecord(
            log_id=uuid.uuid4().hex,
            doc_id=doc_id, kb_id=kb_id, action=action,
            old_value=old_value or {}, new_value=new_value or {},
            operator_id=operator_id, reason=reason,
            created_at=time.time(),
        )
        try:
            await self._store.append_audit(record)
        except Exception as exc:
            log.warning("security.audit_write_failed", error=str(exc))
