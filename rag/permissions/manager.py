"""
rag/permissions/manager.py — 权限管理器

职责：目录 CRUD、权限授予/撤销、访问检查。
存储：SQLite（与 workspace.db 共用或独立 kb_files.db）
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

import structlog

from rag.permissions.models import (
    Directory, DirectoryType, Permission, PermissionLevel, PermissionRole,
)

log = structlog.get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS kb_directories (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    parent_id    TEXT,
    name         TEXT NOT NULL,
    path         TEXT DEFAULT '',
    dir_type     TEXT DEFAULT 'public',
    created_by   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    metadata     TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_dir_ws     ON kb_directories(workspace_id);
CREATE INDEX IF NOT EXISTS idx_dir_parent ON kb_directories(parent_id);

CREATE TABLE IF NOT EXISTS kb_permissions (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    directory_id TEXT NOT NULL,
    subject_type TEXT DEFAULT 'user',
    subject_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    granted_by   TEXT NOT NULL,
    granted_at   REAL NOT NULL,
    expires_at   REAL
);

CREATE INDEX IF NOT EXISTS idx_perm_dir     ON kb_permissions(directory_id);
CREATE INDEX IF NOT EXISTS idx_perm_subject ON kb_permissions(subject_type, subject_id);
"""


class PermissionManager:
    def __init__(self, db_url: str = "./data/kb_files.db") -> None:
        self._path = db_url
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_DDL)
        conn.commit()
        self._conn = conn
        return conn

    def _run(self, fn, *args):
        return fn(self._get_conn(), *args)

    # ── 目录管理 ─────────────────────────────────────────────────────

    async def create_directory(
        self,
        workspace_id: str,
        name: str,
        created_by: str,
        parent_id: str | None = None,
        dir_type: DirectoryType = DirectoryType.PUBLIC,
        metadata: dict | None = None,
    ) -> Directory:
        now = time.time()
        dir_id = uuid.uuid4().hex
        parent_path = ""
        if parent_id:
            parent = await self.get_directory(parent_id)
            if parent:
                parent_path = parent.path
        path = f"{parent_path}/{name}".lstrip("/")

        d = Directory(
            id=dir_id, workspace_id=workspace_id, parent_id=parent_id,
            name=name, path=path, dir_type=dir_type,
            created_by=created_by, created_at=now,
            metadata=metadata or {},
        )

        def _write(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO kb_directories (id,workspace_id,parent_id,name,path,"
                "dir_type,created_by,created_at,metadata) VALUES (?,?,?,?,?,?,?,?,?)",
                (d.id, d.workspace_id, d.parent_id, d.name, d.path,
                 d.dir_type.value, d.created_by, d.created_at,
                 json.dumps(d.metadata)),
            )
            conn.commit()
        await asyncio.to_thread(self._run, _write)
        return d

    async def get_directory(self, directory_id: str) -> Directory | None:
        def _read(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT * FROM kb_directories WHERE id=?", (directory_id,)
            ).fetchone()
            if not row:
                return None
            return Directory(
                id=row["id"], workspace_id=row["workspace_id"],
                parent_id=row["parent_id"], name=row["name"],
                path=row["path"], dir_type=DirectoryType(row["dir_type"]),
                created_by=row["created_by"], created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        return await asyncio.to_thread(self._run, _read)

    async def list_directories(
        self, workspace_id: str, parent_id: str | None = None
    ) -> list[Directory]:
        def _read(conn: sqlite3.Connection):
            if parent_id is None:
                rows = conn.execute(
                    "SELECT * FROM kb_directories WHERE workspace_id=? AND parent_id IS NULL",
                    (workspace_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM kb_directories WHERE workspace_id=? AND parent_id=?",
                    (workspace_id, parent_id),
                ).fetchall()
            return [
                Directory(
                    id=r["id"], workspace_id=r["workspace_id"],
                    parent_id=r["parent_id"], name=r["name"],
                    path=r["path"], dir_type=DirectoryType(r["dir_type"]),
                    created_by=r["created_by"], created_at=r["created_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                for r in rows
            ]
        return await asyncio.to_thread(self._run, _read)

    # ── 权限授予 / 撤销 ───────────────────────────────────────────────

    async def grant(
        self,
        workspace_id: str,
        directory_id: str,
        subject_type: str,
        subject_id: str,
        role: PermissionRole,
        granted_by: str,
        expires_at: float | None = None,
    ) -> Permission:
        now = time.time()
        perm = Permission(
            id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            directory_id=directory_id,
            subject_type=subject_type,
            subject_id=subject_id,
            role=role,
            granted_by=granted_by,
            granted_at=now,
            expires_at=expires_at,
        )

        def _write(conn: sqlite3.Connection):
            # 先删除旧权限（同 subject 对同 directory 只保留最新）
            conn.execute(
                "DELETE FROM kb_permissions WHERE directory_id=? "
                "AND subject_type=? AND subject_id=?",
                (directory_id, subject_type, subject_id),
            )
            conn.execute(
                "INSERT INTO kb_permissions (id,workspace_id,directory_id,"
                "subject_type,subject_id,role,granted_by,granted_at,expires_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (perm.id, perm.workspace_id, perm.directory_id,
                 perm.subject_type, perm.subject_id, perm.role.value,
                 perm.granted_by, perm.granted_at, perm.expires_at),
            )
            conn.commit()
        await asyncio.to_thread(self._run, _write)
        return perm

    async def revoke(
        self, directory_id: str, subject_type: str, subject_id: str
    ) -> None:
        def _write(conn: sqlite3.Connection):
            conn.execute(
                "DELETE FROM kb_permissions WHERE directory_id=? "
                "AND subject_type=? AND subject_id=?",
                (directory_id, subject_type, subject_id),
            )
            conn.commit()
        await asyncio.to_thread(self._run, _write)

    # ── 访问检查 ──────────────────────────────────────────────────────

    async def check_access(
        self,
        user_id: str,
        directory_id: str,
        level: PermissionLevel = PermissionLevel.READ,
    ) -> bool:
        """
        检查用户是否有指定级别的访问权限。

        规则：
          1. 目录 created_by == user_id → 始终有权限
          2. PUBLIC 目录 + READ → 默认允许（无需显式授权）
          3. 有效的权限记录 role >= 所需级别 → 允许
        """
        directory = await self.get_directory(directory_id)
        if directory is None:
            return False

        if directory.created_by == user_id:
            return True

        if directory.dir_type == DirectoryType.PUBLIC and level == PermissionLevel.READ:
            return True

        required_role = {
            PermissionLevel.READ:  PermissionRole.VIEWER,
            PermissionLevel.WRITE: PermissionRole.EDITOR,
            PermissionLevel.ADMIN: PermissionRole.ADMIN,
        }[level]

        def _read(conn: sqlite3.Connection):
            now = time.time()
            rows = conn.execute(
                "SELECT role, expires_at FROM kb_permissions "
                "WHERE directory_id=? AND subject_type='user' AND subject_id=?",
                (directory_id, user_id),
            ).fetchall()
            for row in rows:
                exp = row["expires_at"]
                if exp is not None and exp < now:
                    continue
                if PermissionRole(row["role"]) >= required_role:
                    return True
            return False

        return await asyncio.to_thread(self._run, _read)

    async def list_accessible_directories(
        self,
        workspace_id: str,
        user_id: str,
        level: PermissionLevel = PermissionLevel.READ,
    ) -> list[Directory]:
        """返回用户在指定 workspace 下有访问权限的目录列表。"""
        all_dirs = await self.list_directories(workspace_id)
        results = []
        for d in all_dirs:
            if await self.check_access(user_id, d.id, level):
                results.append(d)
        return results

    async def list_permissions(self, directory_id: str) -> list[Permission]:
        def _read(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT * FROM kb_permissions WHERE directory_id=?", (directory_id,)
            ).fetchall()
            return [
                Permission(
                    id=r["id"], workspace_id=r["workspace_id"],
                    directory_id=r["directory_id"],
                    subject_type=r["subject_type"], subject_id=r["subject_id"],
                    role=PermissionRole(r["role"]), granted_by=r["granted_by"],
                    granted_at=r["granted_at"], expires_at=r["expires_at"],
                )
                for r in rows
            ]
        return await asyncio.to_thread(self._run, _read)
