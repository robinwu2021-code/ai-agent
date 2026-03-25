"""
rag/files/store.py — 文件元数据持久化（SQLite，可扩展到 PostgreSQL）
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

from rag.files.models import (
    ChangeType, FileStatus, KBFile, KBFileAuditLog,
    KBFileVersion, OperationType,
)

log = structlog.get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS kb_files (
    id                  TEXT PRIMARY KEY,
    workspace_id        TEXT NOT NULL,
    kb_id               TEXT NOT NULL,
    directory_id        TEXT NOT NULL,
    name                TEXT NOT NULL,
    original_name       TEXT NOT NULL,
    mime_type           TEXT DEFAULT '',
    file_size           INTEGER DEFAULT 0,
    file_hash           TEXT DEFAULT '',
    status              TEXT DEFAULT 'uploading',
    current_version_id  TEXT,
    version_count       INTEGER DEFAULT 1,
    summary             TEXT DEFAULT '',
    chunk_count         INTEGER DEFAULT 0,
    token_count         INTEGER DEFAULT 0,
    created_at          REAL NOT NULL,
    created_by          TEXT NOT NULL,
    updated_at          REAL DEFAULT 0,
    updated_by          TEXT DEFAULT '',
    metadata            TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kb_files_kb       ON kb_files(kb_id);
CREATE INDEX IF NOT EXISTS idx_kb_files_dir      ON kb_files(directory_id);
CREATE INDEX IF NOT EXISTS idx_kb_files_ws       ON kb_files(workspace_id);
CREATE INDEX IF NOT EXISTS idx_kb_files_hash     ON kb_files(kb_id, file_hash);

CREATE TABLE IF NOT EXISTS kb_file_versions (
    id                   TEXT PRIMARY KEY,
    file_id              TEXT NOT NULL REFERENCES kb_files(id),
    version_number       INTEGER NOT NULL,
    file_path            TEXT DEFAULT '',
    file_size            INTEGER DEFAULT 0,
    file_hash            TEXT DEFAULT '',
    change_type          TEXT NOT NULL,
    change_description   TEXT DEFAULT '',
    summary              TEXT DEFAULT '',
    diff_summary         TEXT DEFAULT '',
    chunk_count          INTEGER DEFAULT 0,
    token_count          INTEGER DEFAULT 0,
    indexing_duration_ms INTEGER DEFAULT 0,
    processing_error     TEXT DEFAULT '',
    created_at           REAL NOT NULL,
    created_by           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_versions_file  ON kb_file_versions(file_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_versions_uniq ON kb_file_versions(file_id, version_number);

CREATE TABLE IF NOT EXISTS kb_file_audit_log (
    id            TEXT PRIMARY KEY,
    file_id       TEXT NOT NULL,
    version_id    TEXT,
    operation     TEXT NOT NULL,
    operator_id   TEXT NOT NULL,
    operator_name TEXT DEFAULT '',
    detail        TEXT DEFAULT '{}',
    ip_address    TEXT DEFAULT '',
    user_agent    TEXT DEFAULT '',
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_file    ON kb_file_audit_log(file_id);
CREATE INDEX IF NOT EXISTS idx_audit_op_id   ON kb_file_audit_log(operator_id);
CREATE INDEX IF NOT EXISTS idx_audit_time    ON kb_file_audit_log(created_at);
"""


def _row_to_file(row: sqlite3.Row) -> KBFile:
    return KBFile(
        id=row["id"], workspace_id=row["workspace_id"],
        kb_id=row["kb_id"], directory_id=row["directory_id"],
        name=row["name"], original_name=row["original_name"],
        mime_type=row["mime_type"], file_size=row["file_size"],
        file_hash=row["file_hash"],
        status=FileStatus(row["status"]),
        current_version_id=row["current_version_id"],
        version_count=row["version_count"],
        summary=row["summary"], chunk_count=row["chunk_count"],
        token_count=row["token_count"],
        created_at=row["created_at"], created_by=row["created_by"],
        updated_at=row["updated_at"], updated_by=row["updated_by"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _row_to_version(row: sqlite3.Row) -> KBFileVersion:
    return KBFileVersion(
        id=row["id"], file_id=row["file_id"],
        version_number=row["version_number"],
        file_path=row["file_path"], file_size=row["file_size"],
        file_hash=row["file_hash"],
        change_type=ChangeType(row["change_type"]),
        change_description=row["change_description"],
        summary=row["summary"], diff_summary=row["diff_summary"],
        chunk_count=row["chunk_count"], token_count=row["token_count"],
        indexing_duration_ms=row["indexing_duration_ms"],
        processing_error=row["processing_error"],
        created_at=row["created_at"], created_by=row["created_by"],
    )


def _row_to_audit(row: sqlite3.Row) -> KBFileAuditLog:
    return KBFileAuditLog(
        id=row["id"], file_id=row["file_id"],
        version_id=row["version_id"],
        operation=OperationType(row["operation"]),
        operator_id=row["operator_id"], operator_name=row["operator_name"],
        detail=json.loads(row["detail"] or "{}"),
        ip_address=row["ip_address"], user_agent=row["user_agent"],
        created_at=row["created_at"],
    )


class KBFileStore:
    """文件元数据 SQLite 持久化层。"""

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
        log.info("kb_file_store.initialized", path=self._path)
        return conn

    def _run(self, fn, *args) -> Any:
        conn = self._get_conn()
        return fn(conn, *args)

    # ── KBFile ───────────────────────────────────────────────────────

    async def save_file(self, f: KBFile) -> KBFile:
        def _write(conn: sqlite3.Connection):
            conn.execute("""
                INSERT OR REPLACE INTO kb_files
                (id, workspace_id, kb_id, directory_id, name, original_name,
                 mime_type, file_size, file_hash, status, current_version_id,
                 version_count, summary, chunk_count, token_count,
                 created_at, created_by, updated_at, updated_by, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                f.id, f.workspace_id, f.kb_id, f.directory_id,
                f.name, f.original_name, f.mime_type, f.file_size,
                f.file_hash, f.status.value, f.current_version_id,
                f.version_count, f.summary, f.chunk_count, f.token_count,
                f.created_at, f.created_by, f.updated_at, f.updated_by,
                json.dumps(f.metadata),
            ))
            conn.commit()
        await asyncio.to_thread(self._run, _write)
        return f

    async def get_file(self, file_id: str) -> KBFile | None:
        def _read(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT * FROM kb_files WHERE id=?", (file_id,)
            ).fetchone()
            return _row_to_file(row) if row else None
        return await asyncio.to_thread(self._run, _read)

    async def find_by_hash(self, kb_id: str, file_hash: str) -> KBFile | None:
        def _read(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT * FROM kb_files WHERE kb_id=? AND file_hash=? AND status!=?",
                (kb_id, file_hash, FileStatus.DELETED.value),
            ).fetchone()
            return _row_to_file(row) if row else None
        return await asyncio.to_thread(self._run, _read)

    async def list_files(
        self,
        kb_id: str | None = None,
        directory_id: str | None = None,
        workspace_id: str | None = None,
        include_deleted: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> list[KBFile]:
        def _read(conn: sqlite3.Connection):
            conds = []
            params: list = []
            if kb_id:
                conds.append("kb_id=?"); params.append(kb_id)
            if directory_id:
                conds.append("directory_id=?"); params.append(directory_id)
            if workspace_id:
                conds.append("workspace_id=?"); params.append(workspace_id)
            if not include_deleted:
                conds.append("status!=?"); params.append(FileStatus.DELETED.value)
            where = ("WHERE " + " AND ".join(conds)) if conds else ""
            rows = conn.execute(
                f"SELECT * FROM kb_files {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            return [_row_to_file(r) for r in rows]
        return await asyncio.to_thread(self._run, _read)

    async def soft_delete_file(self, file_id: str) -> None:
        def _write(conn: sqlite3.Connection):
            conn.execute(
                "UPDATE kb_files SET status=?, updated_at=? WHERE id=?",
                (FileStatus.DELETED.value, time.time(), file_id),
            )
            conn.commit()
        await asyncio.to_thread(self._run, _write)

    # ── KBFileVersion ────────────────────────────────────────────────

    async def save_version(self, v: KBFileVersion) -> KBFileVersion:
        def _write(conn: sqlite3.Connection):
            conn.execute("""
                INSERT OR REPLACE INTO kb_file_versions
                (id, file_id, version_number, file_path, file_size, file_hash,
                 change_type, change_description, summary, diff_summary,
                 chunk_count, token_count, indexing_duration_ms,
                 processing_error, created_at, created_by)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                v.id, v.file_id, v.version_number, v.file_path,
                v.file_size, v.file_hash, v.change_type.value,
                v.change_description, v.summary, v.diff_summary,
                v.chunk_count, v.token_count, v.indexing_duration_ms,
                v.processing_error, v.created_at, v.created_by,
            ))
            conn.commit()
        await asyncio.to_thread(self._run, _write)
        return v

    async def get_version(self, version_id: str) -> KBFileVersion | None:
        def _read(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT * FROM kb_file_versions WHERE id=?", (version_id,)
            ).fetchone()
            return _row_to_version(row) if row else None
        return await asyncio.to_thread(self._run, _read)

    async def list_versions(
        self, file_id: str, limit: int = 20, offset: int = 0
    ) -> list[KBFileVersion]:
        def _read(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT * FROM kb_file_versions WHERE file_id=? "
                "ORDER BY version_number DESC LIMIT ? OFFSET ?",
                (file_id, limit, offset),
            ).fetchall()
            return [_row_to_version(r) for r in rows]
        return await asyncio.to_thread(self._run, _read)

    # ── KBFileAuditLog ───────────────────────────────────────────────

    async def append_audit(self, log_entry: KBFileAuditLog) -> None:
        def _write(conn: sqlite3.Connection):
            conn.execute("""
                INSERT INTO kb_file_audit_log
                (id, file_id, version_id, operation, operator_id, operator_name,
                 detail, ip_address, user_agent, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                log_entry.id, log_entry.file_id, log_entry.version_id,
                log_entry.operation.value, log_entry.operator_id, log_entry.operator_name,
                json.dumps(log_entry.detail),
                log_entry.ip_address, log_entry.user_agent, log_entry.created_at,
            ))
            conn.commit()
        await asyncio.to_thread(self._run, _write)

    async def query_audit(
        self,
        file_id: str | None = None,
        kb_id: str | None = None,
        operator_id: str | None = None,
        operation: OperationType | None = None,
        since: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KBFileAuditLog]:
        def _read(conn: sqlite3.Connection):
            conds: list[str] = []
            params: list = []
            if file_id:
                conds.append("a.file_id=?"); params.append(file_id)
            if operator_id:
                conds.append("a.operator_id=?"); params.append(operator_id)
            if operation:
                conds.append("a.operation=?"); params.append(operation.value)
            if since:
                conds.append("a.created_at>=?"); params.append(since)
            # kb_id 需要 join kb_files
            join = ""
            if kb_id:
                join = "JOIN kb_files f ON a.file_id=f.id"
                conds.append("f.kb_id=?"); params.append(kb_id)
            where = ("WHERE " + " AND ".join(conds)) if conds else ""
            rows = conn.execute(
                f"SELECT a.* FROM kb_file_audit_log a {join} {where} "
                f"ORDER BY a.created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            return [_row_to_audit(r) for r in rows]
        return await asyncio.to_thread(self._run, _read)
