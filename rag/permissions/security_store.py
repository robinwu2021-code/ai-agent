"""
rag/permissions/security_store.py — 多维度权限 SQLite 持久化层

与现有 workspace.db 共用同一数据库文件，新建独立表（前缀 sec_）。
不修改现有表，向后兼容。

表结构：
  sec_users            用户表
  sec_org_units        组织单元（部门/项目），树形结构
  sec_org_memberships  用户-组织单元归属（多对多）
  sec_doc_permissions  文档安全级别记录
  sec_doc_grants       CONFIDENTIAL 显式授权
  sec_audit_log        权限变更审计日志
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from rag.permissions.security_models import (
    DocGrant, DocOperation, DocPermission, OrgMembership, OrgUnit,
    OrgUnitType, PermAuditRecord, SecUser, SecurityLevel, SystemRole,
)

log = structlog.get_logger(__name__)

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sec_users (
    user_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT DEFAULT '',
    role        TEXT DEFAULT 'MEMBER',
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sec_org_units (
    unit_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    unit_type   TEXT NOT NULL DEFAULT 'DEPARTMENT',
    parent_id   TEXT REFERENCES sec_org_units(unit_id),
    created_by  TEXT DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_org_parent ON sec_org_units(parent_id);

CREATE TABLE IF NOT EXISTS sec_org_memberships (
    user_id     TEXT NOT NULL,
    unit_id     TEXT NOT NULL REFERENCES sec_org_units(unit_id),
    is_admin    INTEGER DEFAULT 0,
    joined_at   REAL NOT NULL,
    PRIMARY KEY (user_id, unit_id)
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON sec_org_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_unit ON sec_org_memberships(unit_id);

CREATE TABLE IF NOT EXISTS sec_doc_permissions (
    doc_id          TEXT NOT NULL,
    kb_id           TEXT NOT NULL,
    security_level  TEXT NOT NULL,
    owner_id        TEXT NOT NULL,
    org_unit_ids    TEXT DEFAULT '[]',  -- JSON list
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    changed_by      TEXT DEFAULT '',
    PRIMARY KEY (doc_id, kb_id)
);
CREATE INDEX IF NOT EXISTS idx_docperm_kb    ON sec_doc_permissions(kb_id);
CREATE INDEX IF NOT EXISTS idx_docperm_owner ON sec_doc_permissions(owner_id);
CREATE INDEX IF NOT EXISTS idx_docperm_level ON sec_doc_permissions(security_level);

CREATE TABLE IF NOT EXISTS sec_doc_grants (
    grant_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    kb_id       TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    operation   TEXT NOT NULL,
    granted_by  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    UNIQUE (doc_id, kb_id, user_id, operation)
);
CREATE INDEX IF NOT EXISTS idx_grant_doc  ON sec_doc_grants(doc_id, kb_id);
CREATE INDEX IF NOT EXISTS idx_grant_user ON sec_doc_grants(user_id);

CREATE TABLE IF NOT EXISTS sec_audit_log (
    log_id      TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    kb_id       TEXT NOT NULL,
    action      TEXT NOT NULL,
    old_value   TEXT DEFAULT '{}',
    new_value   TEXT DEFAULT '{}',
    operator_id TEXT DEFAULT '',
    reason      TEXT DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_doc  ON sec_audit_log(doc_id, kb_id);
CREATE INDEX IF NOT EXISTS idx_audit_op   ON sec_audit_log(operator_id);
CREATE INDEX IF NOT EXISTS idx_audit_time ON sec_audit_log(created_at);
"""


class SecurityStore:
    """多维度权限 SQLite 存储层。"""

    def __init__(self, db_path: str = "./data/workspace.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ── 连接 ──────────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_DDL)
            self._conn.commit()
        return self._conn

    def _q(self, sql: str, args: tuple = ()) -> list[dict]:
        return [dict(r) for r in self._get_conn().execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> dict | None:
        r = self._get_conn().execute(sql, args).fetchone()
        return dict(r) if r else None

    def _exec(self, sql: str, args: tuple = ()) -> None:
        self._get_conn().execute(sql, args)
        self._get_conn().commit()

    async def initialize(self) -> None:
        await asyncio.to_thread(self._get_conn)
        log.info("security_store.initialized", path=str(self._path))

    # ── 用户 CRUD ─────────────────────────────────────────────────────────────

    async def upsert_user(self, user: SecUser) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT OR REPLACE INTO sec_users (user_id,name,email,role,created_at)"
            " VALUES (?,?,?,?,?)",
            (user.user_id, user.name, user.email, user.role.value, user.created_at or time.time()),
        )

    async def get_user(self, user_id: str) -> SecUser | None:
        row = await asyncio.to_thread(
            self._one, "SELECT * FROM sec_users WHERE user_id=?", (user_id,)
        )
        return _row_to_user(row) if row else None

    async def list_users(self) -> list[SecUser]:
        rows = await asyncio.to_thread(self._q, "SELECT * FROM sec_users ORDER BY name")
        return [_row_to_user(r) for r in rows]

    async def update_user_role(self, user_id: str, role: SystemRole) -> None:
        await asyncio.to_thread(
            self._exec, "UPDATE sec_users SET role=? WHERE user_id=?",
            (role.value, user_id),
        )

    # ── 组织单元 CRUD ─────────────────────────────────────────────────────────

    async def upsert_org_unit(self, unit: OrgUnit) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT OR REPLACE INTO sec_org_units"
            " (unit_id,name,unit_type,parent_id,created_by,created_at) VALUES (?,?,?,?,?,?)",
            (unit.unit_id, unit.name, unit.unit_type.value,
             unit.parent_id, unit.created_by, unit.created_at or time.time()),
        )

    async def get_org_unit(self, unit_id: str) -> OrgUnit | None:
        row = await asyncio.to_thread(
            self._one, "SELECT * FROM sec_org_units WHERE unit_id=?", (unit_id,)
        )
        return _row_to_unit(row) if row else None

    async def list_org_units(self, unit_type: OrgUnitType | None = None) -> list[OrgUnit]:
        if unit_type:
            rows = await asyncio.to_thread(
                self._q,
                "SELECT * FROM sec_org_units WHERE unit_type=? ORDER BY name",
                (unit_type.value,),
            )
        else:
            rows = await asyncio.to_thread(
                self._q, "SELECT * FROM sec_org_units ORDER BY unit_type, name"
            )
        return [_row_to_unit(r) for r in rows]

    async def delete_org_unit(self, unit_id: str) -> None:
        await asyncio.to_thread(
            self._exec, "DELETE FROM sec_org_units WHERE unit_id=?", (unit_id,)
        )

    # ── 组织单元层级查询（递归 CTE）─────────────────────────────────────────

    async def get_ancestors(self, unit_id: str) -> list[str]:
        """返回该单元的所有祖先 unit_id（不含自身）。"""
        sql = """
        WITH RECURSIVE anc(uid, pid) AS (
            SELECT unit_id, parent_id FROM sec_org_units WHERE unit_id=?
            UNION ALL
            SELECT o.unit_id, o.parent_id FROM sec_org_units o
            INNER JOIN anc a ON o.unit_id = a.pid
        )
        SELECT uid FROM anc WHERE uid != ?
        """
        rows = await asyncio.to_thread(self._q, sql, (unit_id, unit_id))
        return [r["uid"] for r in rows]

    async def get_descendants(self, unit_id: str) -> list[str]:
        """返回该单元的所有后代 unit_id（不含自身）。"""
        sql = """
        WITH RECURSIVE desc_cte(uid, pid) AS (
            SELECT unit_id, parent_id FROM sec_org_units WHERE unit_id=?
            UNION ALL
            SELECT o.unit_id, o.parent_id FROM sec_org_units o
            INNER JOIN desc_cte d ON o.parent_id = d.uid
        )
        SELECT uid FROM desc_cte WHERE uid != ?
        """
        rows = await asyncio.to_thread(self._q, sql, (unit_id, unit_id))
        return [r["uid"] for r in rows]

    # ── 成员管理 ──────────────────────────────────────────────────────────────

    async def add_member(self, user_id: str, unit_id: str, is_admin: bool = False) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT OR REPLACE INTO sec_org_memberships (user_id,unit_id,is_admin,joined_at)"
            " VALUES (?,?,?,?)",
            (user_id, unit_id, int(is_admin), time.time()),
        )

    async def remove_member(self, user_id: str, unit_id: str) -> None:
        await asyncio.to_thread(
            self._exec,
            "DELETE FROM sec_org_memberships WHERE user_id=? AND unit_id=?",
            (user_id, unit_id),
        )

    async def get_user_memberships(self, user_id: str) -> list[OrgMembership]:
        rows = await asyncio.to_thread(
            self._q,
            "SELECT * FROM sec_org_memberships WHERE user_id=?", (user_id,)
        )
        return [OrgMembership(
            user_id=r["user_id"], unit_id=r["unit_id"],
            is_admin=bool(r["is_admin"]), joined_at=r["joined_at"]
        ) for r in rows]

    async def get_unit_members(self, unit_id: str) -> list[OrgMembership]:
        rows = await asyncio.to_thread(
            self._q,
            "SELECT * FROM sec_org_memberships WHERE unit_id=?", (unit_id,)
        )
        return [OrgMembership(
            user_id=r["user_id"], unit_id=r["unit_id"],
            is_admin=bool(r["is_admin"]), joined_at=r["joined_at"]
        ) for r in rows]

    # ── 文档权限 CRUD ─────────────────────────────────────────────────────────

    async def upsert_doc_permission(self, perm: DocPermission) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT OR REPLACE INTO sec_doc_permissions"
            " (doc_id,kb_id,security_level,owner_id,org_unit_ids,"
            "  created_at,updated_at,changed_by)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (perm.doc_id, perm.kb_id, perm.security_level.value, perm.owner_id,
             json.dumps(perm.org_unit_ids),
             perm.created_at or time.time(), perm.updated_at or time.time(),
             perm.changed_by),
        )

    async def get_doc_permission(self, doc_id: str, kb_id: str) -> DocPermission | None:
        row = await asyncio.to_thread(
            self._one,
            "SELECT * FROM sec_doc_permissions WHERE doc_id=? AND kb_id=?",
            (doc_id, kb_id),
        )
        return _row_to_doc_perm(row) if row else None

    async def list_doc_permissions_by_kb(
        self, kb_id: str, security_level: SecurityLevel | None = None
    ) -> list[DocPermission]:
        if security_level:
            rows = await asyncio.to_thread(
                self._q,
                "SELECT * FROM sec_doc_permissions WHERE kb_id=? AND security_level=?",
                (kb_id, security_level.value),
            )
        else:
            rows = await asyncio.to_thread(
                self._q,
                "SELECT * FROM sec_doc_permissions WHERE kb_id=?", (kb_id,)
            )
        return [_row_to_doc_perm(r) for r in rows]

    async def delete_doc_permission(self, doc_id: str, kb_id: str) -> None:
        await asyncio.to_thread(
            self._exec,
            "DELETE FROM sec_doc_permissions WHERE doc_id=? AND kb_id=?",
            (doc_id, kb_id),
        )

    # ── 显式授权（CONFIDENTIAL）──────────────────────────────────────────────

    async def add_grant(self, grant: DocGrant) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT OR REPLACE INTO sec_doc_grants"
            " (grant_id,doc_id,kb_id,user_id,operation,granted_by,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (grant.grant_id, grant.doc_id, grant.kb_id, grant.user_id,
             grant.operation.value, grant.granted_by, grant.created_at or time.time()),
        )

    async def remove_grant(self, doc_id: str, kb_id: str, user_id: str,
                           operation: DocOperation | None = None) -> None:
        if operation:
            await asyncio.to_thread(
                self._exec,
                "DELETE FROM sec_doc_grants WHERE doc_id=? AND kb_id=? AND user_id=? AND operation=?",
                (doc_id, kb_id, user_id, operation.value),
            )
        else:
            await asyncio.to_thread(
                self._exec,
                "DELETE FROM sec_doc_grants WHERE doc_id=? AND kb_id=? AND user_id=?",
                (doc_id, kb_id, user_id),
            )

    async def get_grants(self, doc_id: str, kb_id: str) -> list[DocGrant]:
        rows = await asyncio.to_thread(
            self._q,
            "SELECT * FROM sec_doc_grants WHERE doc_id=? AND kb_id=?",
            (doc_id, kb_id),
        )
        return [_row_to_grant(r) for r in rows]

    async def get_user_confidential_doc_ids(self, user_id: str, kb_id: str) -> list[str]:
        """返回用户在指定 kb 中被显式授权的 CONFIDENTIAL 文档 ID 列表。"""
        # 先获取用户有授权的所有文档，再过滤为 CONFIDENTIAL 级别
        rows = await asyncio.to_thread(
            self._q,
            """
            SELECT DISTINCT g.doc_id FROM sec_doc_grants g
            JOIN sec_doc_permissions p ON g.doc_id=p.doc_id AND g.kb_id=p.kb_id
            WHERE g.user_id=? AND g.kb_id=? AND p.security_level='CONFIDENTIAL'
            """,
            (user_id, kb_id),
        )
        return [r["doc_id"] for r in rows]

    # ── 审计日志 ──────────────────────────────────────────────────────────────

    async def append_audit(self, record: PermAuditRecord) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO sec_audit_log"
            " (log_id,doc_id,kb_id,action,old_value,new_value,"
            "  operator_id,reason,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (record.log_id, record.doc_id, record.kb_id, record.action,
             json.dumps(record.old_value, ensure_ascii=False),
             json.dumps(record.new_value, ensure_ascii=False),
             record.operator_id, record.reason,
             record.created_at or time.time()),
        )

    async def query_audit(
        self,
        kb_id: str,
        doc_id: str | None = None,
        operator_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[PermAuditRecord]:
        conditions = ["kb_id=?"]
        args: list[Any] = [kb_id]
        if doc_id:
            conditions.append("doc_id=?"); args.append(doc_id)
        if operator_id:
            conditions.append("operator_id=?"); args.append(operator_id)
        if since:
            conditions.append("created_at>=?"); args.append(since)
        sql = (
            "SELECT * FROM sec_audit_log WHERE " + " AND ".join(conditions)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        args.append(limit)
        rows = await asyncio.to_thread(self._q, sql, tuple(args))
        return [PermAuditRecord(
            log_id=r["log_id"], doc_id=r["doc_id"], kb_id=r["kb_id"],
            action=r["action"],
            old_value=json.loads(r["old_value"] or "{}"),
            new_value=json.loads(r["new_value"] or "{}"),
            operator_id=r["operator_id"], reason=r["reason"],
            created_at=r["created_at"],
        ) for r in rows]


# ── 行转换辅助函数 ─────────────────────────────────────────────────────────────

def _row_to_user(r: dict) -> SecUser:
    return SecUser(
        user_id=r["user_id"], name=r["name"], email=r.get("email", ""),
        role=SystemRole(r["role"]), created_at=r["created_at"],
    )

def _row_to_unit(r: dict) -> OrgUnit:
    return OrgUnit(
        unit_id=r["unit_id"], name=r["name"],
        unit_type=OrgUnitType(r["unit_type"]),
        parent_id=r.get("parent_id"),
        created_by=r.get("created_by", ""),
        created_at=r["created_at"],
    )

def _row_to_doc_perm(r: dict) -> DocPermission:
    return DocPermission(
        doc_id=r["doc_id"], kb_id=r["kb_id"],
        security_level=SecurityLevel(r["security_level"]),
        owner_id=r["owner_id"],
        org_unit_ids=json.loads(r.get("org_unit_ids") or "[]"),
        created_at=r["created_at"], updated_at=r["updated_at"],
        changed_by=r.get("changed_by", ""),
    )

def _row_to_grant(r: dict) -> DocGrant:
    return DocGrant(
        grant_id=r["grant_id"], doc_id=r["doc_id"], kb_id=r["kb_id"],
        user_id=r["user_id"], operation=DocOperation(r["operation"]),
        granted_by=r["granted_by"], created_at=r["created_at"],
    )
