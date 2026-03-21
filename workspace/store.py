"""
workspace/store.py — 工作区持久化存储

支持两种后端，接口完全一致：
  SQLiteWorkspaceStore   — 零外部依赖，stdlib sqlite3，适合开发/单机部署
  MySQLWorkspaceStore    — 生产级，需安装 pymysql（同步，via asyncio.to_thread）

数据库 Schema（两种后端相同）：
┌─────────────────────────────────────────────────────────────────┐
│  workspaces          工作区基本信息                               │
│  workspace_members   工作区成员（M:N）                            │
│  projects            项目（属于某工作区）                          │
│  project_members     项目成员（M:N，必须是工作区成员）              │
│  memory_entries      记忆条目（三范围：personal/project/workspace）│
│  memory_shares       跨项目共享授权（M:N）                        │
│  user_profiles       用户画像（per workspace）                   │
│  sessions            会话与工作区/项目关联                        │
└─────────────────────────────────────────────────────────────────┘

方言差异处理：
  _Dialect.SQLITE   参数占位符 ?   ，连接用 sqlite3
  _Dialect.MYSQL    参数占位符 %s  ，连接用 pymysql
  其余 SQL 语句两者保持兼容（标准 SQL-92 子集）。

使用方式：
  # SQLite
  store = SQLiteWorkspaceStore("workspace.db")
  await store.initialize()

  # MySQL
  store = MySQLWorkspaceStore(host="127.0.0.1", port=3306,
                              user="root", password="xxx", database="agent")
  await store.initialize()
"""
from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Generator

import structlog

from workspace.models import (
    MemoryScope,
    MemoryShare,
    Project,
    ProjectMember,
    ProjectRole,
    Workspace,
    WorkspaceMember,
    WorkspaceMemoryEntry,
    WorkspaceRole,
)

log = structlog.get_logger(__name__)

_DT_FMT = "%Y-%m-%d %H:%M:%S"


def _dt(s: str | None) -> datetime:
    if not s:
        return datetime.utcnow()
    try:
        return datetime.strptime(s[:19], _DT_FMT)
    except ValueError:
        return datetime.utcnow()


def _jd(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _jl(s: str | None) -> Any:
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
# SQL Schema（两种方言均兼容）
# ──────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id   TEXT         NOT NULL PRIMARY KEY,
    name           TEXT         NOT NULL,
    description    TEXT         NOT NULL DEFAULT '',
    allowed_skills TEXT         NOT NULL DEFAULT '[]',
    system_prompt  TEXT         NOT NULL DEFAULT '',
    token_budget   INTEGER      NOT NULL DEFAULT 16000,
    max_steps      INTEGER      NOT NULL DEFAULT 20,
    metadata       TEXT         NOT NULL DEFAULT '{}',
    created_at     DATETIME     NOT NULL,
    updated_at     DATETIME     NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_members (
    workspace_id   TEXT         NOT NULL,
    user_id        TEXT         NOT NULL,
    role           TEXT         NOT NULL DEFAULT 'member',
    joined_at      DATETIME     NOT NULL,
    PRIMARY KEY (workspace_id, user_id)
);

CREATE TABLE IF NOT EXISTS projects (
    project_id     TEXT         NOT NULL PRIMARY KEY,
    workspace_id   TEXT         NOT NULL,
    name           TEXT         NOT NULL,
    description    TEXT         NOT NULL DEFAULT '',
    allowed_skills TEXT         NOT NULL DEFAULT '[]',
    system_prompt  TEXT         NOT NULL DEFAULT '',
    token_budget   INTEGER      NOT NULL DEFAULT 12000,
    max_steps      INTEGER      NOT NULL DEFAULT 20,
    metadata       TEXT         NOT NULL DEFAULT '{}',
    created_at     DATETIME     NOT NULL,
    updated_at     DATETIME     NOT NULL
);

CREATE TABLE IF NOT EXISTS project_members (
    project_id     TEXT         NOT NULL,
    user_id        TEXT         NOT NULL,
    role           TEXT         NOT NULL DEFAULT 'member',
    joined_at      DATETIME     NOT NULL,
    PRIMARY KEY (project_id, user_id)
);

CREATE TABLE IF NOT EXISTS memory_entries (
    entry_id       TEXT         NOT NULL PRIMARY KEY,
    scope          TEXT         NOT NULL,
    owner_id       TEXT         NOT NULL,
    workspace_id   TEXT,
    project_id     TEXT,
    author_id      TEXT         NOT NULL DEFAULT '',
    memory_type    TEXT         NOT NULL DEFAULT 'semantic',
    text           TEXT         NOT NULL,
    importance     REAL         NOT NULL DEFAULT 0.5,
    tags           TEXT         NOT NULL DEFAULT '[]',
    metadata       TEXT         NOT NULL DEFAULT '{}',
    access_count   INTEGER      NOT NULL DEFAULT 0,
    created_at     DATETIME     NOT NULL,
    accessed_at    DATETIME     NOT NULL,
    expires_at     DATETIME
);

CREATE TABLE IF NOT EXISTS memory_shares (
    entry_id              TEXT     NOT NULL,
    shared_to_project_id  TEXT     NOT NULL,
    shared_by_user_id     TEXT     NOT NULL DEFAULT '',
    shared_at             DATETIME NOT NULL,
    permission            TEXT     NOT NULL DEFAULT 'read',
    note                  TEXT     NOT NULL DEFAULT '',
    PRIMARY KEY (entry_id, shared_to_project_id)
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id        TEXT         NOT NULL,
    workspace_id   TEXT         NOT NULL DEFAULT '',
    profile_data   TEXT         NOT NULL DEFAULT '{}',
    updated_at     DATETIME     NOT NULL,
    PRIMARY KEY (user_id, workspace_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT         NOT NULL PRIMARY KEY,
    user_id        TEXT         NOT NULL,
    workspace_id   TEXT,
    project_id     TEXT,
    created_at     DATETIME     NOT NULL,
    last_active_at DATETIME     NOT NULL,
    metadata       TEXT         NOT NULL DEFAULT '{}'
);
"""

_IDX = """
CREATE INDEX IF NOT EXISTS idx_me_owner     ON memory_entries(owner_id);
CREATE INDEX IF NOT EXISTS idx_me_workspace ON memory_entries(workspace_id);
CREATE INDEX IF NOT EXISTS idx_me_scope     ON memory_entries(scope);
CREATE INDEX IF NOT EXISTS idx_ms_target    ON memory_shares(shared_to_project_id);
CREATE INDEX IF NOT EXISTS idx_proj_ws      ON projects(workspace_id);
CREATE INDEX IF NOT EXISTS idx_sess_user    ON sessions(user_id);
"""


# ──────────────────────────────────────────────────────────────
# 方言
# ──────────────────────────────────────────────────────────────

class _Dialect(Enum):
    SQLITE = "sqlite"
    MYSQL  = "mysql"


def _ph(dialect: _Dialect, n: int = 1) -> str:
    """返回 n 个占位符，逗号分隔。"""
    p = "?" if dialect == _Dialect.SQLITE else "%s"
    return ", ".join([p] * n)


def _set_clause(dialect: _Dialect, cols: list[str]) -> str:
    p = "?" if dialect == _Dialect.SQLITE else "%s"
    return ", ".join(f"{c} = {p}" for c in cols)


# ──────────────────────────────────────────────────────────────
# 抽象基类
# ──────────────────────────────────────────────────────────────

class WorkspaceStore(ABC):
    """工作区持久化接口（与方言无关）。"""

    # ── 初始化 ──────────────────────────────────────────────────
    @abstractmethod
    async def initialize(self) -> None:
        """建表、建索引。"""

    # ── Workspace CRUD ─────────────────────────────────────────
    @abstractmethod
    async def save_workspace(self, ws: Workspace) -> None: ...
    @abstractmethod
    async def get_workspace(self, workspace_id: str) -> Workspace | None: ...
    @abstractmethod
    async def list_workspaces(self, user_id: str | None = None) -> list[Workspace]: ...
    @abstractmethod
    async def delete_workspace(self, workspace_id: str) -> None: ...

    # ── 成员 ────────────────────────────────────────────────────
    @abstractmethod
    async def add_workspace_member(self, workspace_id: str, member: WorkspaceMember) -> None: ...
    @abstractmethod
    async def remove_workspace_member(self, workspace_id: str, user_id: str) -> None: ...
    @abstractmethod
    async def update_workspace_member_role(self, workspace_id: str, user_id: str, role: WorkspaceRole) -> None: ...

    # ── Project CRUD ────────────────────────────────────────────
    @abstractmethod
    async def save_project(self, proj: Project) -> None: ...
    @abstractmethod
    async def get_project(self, project_id: str) -> Project | None: ...
    @abstractmethod
    async def list_projects(self, workspace_id: str) -> list[Project]: ...
    @abstractmethod
    async def delete_project(self, project_id: str) -> None: ...

    @abstractmethod
    async def add_project_member(self, project_id: str, member: ProjectMember) -> None: ...
    @abstractmethod
    async def remove_project_member(self, project_id: str, user_id: str) -> None: ...

    # ── Memory 写入 ─────────────────────────────────────────────
    @abstractmethod
    async def save_memory(self, entry: WorkspaceMemoryEntry) -> None: ...
    @abstractmethod
    async def delete_memory(self, entry_id: str) -> None: ...
    @abstractmethod
    async def touch_memory(self, entry_id: str) -> None:
        """更新 accessed_at 并 +1 access_count。"""

    # ── Memory 读取 ─────────────────────────────────────────────
    @abstractmethod
    async def query_personal(self, user_id: str, limit: int = 200) -> list[WorkspaceMemoryEntry]: ...

    @abstractmethod
    async def query_project(self, workspace_id: str, project_id: str, limit: int = 200) -> list[WorkspaceMemoryEntry]: ...

    @abstractmethod
    async def query_workspace(self, workspace_id: str, limit: int = 200) -> list[WorkspaceMemoryEntry]: ...

    @abstractmethod
    async def query_shared_to_project(self, project_id: str, limit: int = 200) -> list[WorkspaceMemoryEntry]: ...

    # ── 跨项目共享 ──────────────────────────────────────────────
    @abstractmethod
    async def share_memory(self, share: MemoryShare) -> None: ...
    @abstractmethod
    async def revoke_share(self, entry_id: str, project_id: str) -> None: ...
    @abstractmethod
    async def list_shares(self, entry_id: str) -> list[MemoryShare]: ...
    @abstractmethod
    async def list_shares_to_project(self, project_id: str) -> list[MemoryShare]: ...

    # ── 用户画像 ─────────────────────────────────────────────────
    @abstractmethod
    async def get_profile(self, user_id: str, workspace_id: str = "") -> dict[str, Any]: ...
    @abstractmethod
    async def update_profile(self, user_id: str, workspace_id: str, data: dict[str, Any]) -> None: ...

    # ── 裁剪 ─────────────────────────────────────────────────────
    @abstractmethod
    async def prune_personal(self, user_id: str, max_items: int, score_threshold: float) -> int: ...
    @abstractmethod
    async def prune_project(self, workspace_id: str, project_id: str, max_items: int) -> int: ...

    # ── Session ──────────────────────────────────────────────────
    @abstractmethod
    async def save_session(self, session_id: str, user_id: str,
                           workspace_id: str | None, project_id: str | None) -> None: ...
    @abstractmethod
    async def get_session(self, session_id: str) -> dict[str, Any] | None: ...


# ──────────────────────────────────────────────────────────────
# 共享 SQL 逻辑（方言参数化）
# ──────────────────────────────────────────────────────────────

class _BaseRelationalStore(WorkspaceStore):
    """
    持有方言配置并实现所有 SQL 逻辑；
    子类只需实现 _connect / _execute / _fetchall / _fetchone。
    """

    def __init__(self, dialect: _Dialect) -> None:
        self._dialect = dialect

    # ── 子类必须实现的原语 ──────────────────────────────────────

    @abstractmethod
    def _get_conn(self) -> Any:
        """返回同步 DB 连接（sqlite3.Connection / pymysql.Connection）。"""

    def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """在线程池中运行同步 DB 操作（兼容 asyncio）。"""
        return asyncio.get_event_loop().run_in_executor(None, fn, *args, **kwargs)

    # ── execute helpers ─────────────────────────────────────────

    def _exec(self, sql: str, params: tuple = ()) -> list[dict]:
        """同步执行一条 SQL，返回所有行（dict 列表）。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            else:
                rows = []
            conn.commit()
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _exec_many(self, sql: str, param_list: list[tuple]) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.executemany(sql, param_list)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    async def _a(self, fn: Any, *args: Any) -> Any:
        """将同步调用包装为 async（线程池）。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    # ── 初始化 ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        def _init() -> None:
            conn = self._get_conn()
            cur = conn.cursor()
            for stmt in _DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            for stmt in _IDX.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        cur.execute(stmt)
                    except Exception:
                        pass  # 索引已存在时忽略
            conn.commit()
            cur.close()
        await self._a(_init)
        log.info("workspace_store.initialized", dialect=self._dialect.value)

    # ══════════════════════════════════════════════════════════════
    # Workspace CRUD
    # ══════════════════════════════════════════════════════════════

    async def save_workspace(self, ws: Workspace) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _save() -> None:
            conn = self._get_conn()
            cur  = conn.cursor()
            now  = datetime.utcnow().strftime(_DT_FMT)
            # upsert workspace
            if self._dialect == _Dialect.SQLITE:
                cur.execute(
                    f"""INSERT INTO workspaces VALUES ({_ph(self._dialect, 10)})
                        ON CONFLICT(workspace_id) DO UPDATE SET
                        name=excluded.name, description=excluded.description,
                        allowed_skills=excluded.allowed_skills,
                        system_prompt=excluded.system_prompt,
                        token_budget=excluded.token_budget, max_steps=excluded.max_steps,
                        metadata=excluded.metadata, updated_at=excluded.updated_at""",
                    (ws.workspace_id, ws.name, ws.description,
                     _jd(ws.allowed_skills), ws.system_prompt,
                     ws.token_budget, ws.max_steps, _jd(ws.metadata),
                     ws.created_at.strftime(_DT_FMT), now),
                )
            else:
                cur.execute(
                    f"""INSERT INTO workspaces VALUES ({_ph(self._dialect, 10)})
                        ON DUPLICATE KEY UPDATE
                        name=VALUES(name), description=VALUES(description),
                        allowed_skills=VALUES(allowed_skills),
                        system_prompt=VALUES(system_prompt),
                        token_budget=VALUES(token_budget), max_steps=VALUES(max_steps),
                        metadata=VALUES(metadata), updated_at=VALUES(updated_at)""",
                    (ws.workspace_id, ws.name, ws.description,
                     _jd(ws.allowed_skills), ws.system_prompt,
                     ws.token_budget, ws.max_steps, _jd(ws.metadata),
                     ws.created_at.strftime(_DT_FMT), now),
                )
            # 重建成员（简单实现：删除再插入）
            cur.execute(f"DELETE FROM workspace_members WHERE workspace_id = {p}", (ws.workspace_id,))
            for m in ws.members:
                cur.execute(
                    f"INSERT INTO workspace_members VALUES ({_ph(self._dialect, 4)})",
                    (ws.workspace_id, m.user_id, m.role.value,
                     m.joined_at.strftime(_DT_FMT)),
                )
            conn.commit()
            cur.close()

        await self._a(_save)

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _get() -> Workspace | None:
            rows = self._exec(f"SELECT * FROM workspaces WHERE workspace_id = {p}", (workspace_id,))
            if not rows:
                return None
            r = rows[0]
            members = [
                WorkspaceMember(
                    user_id=m["user_id"],
                    role=WorkspaceRole(m["role"]),
                    joined_at=_dt(m["joined_at"]),
                )
                for m in self._exec(
                    f"SELECT * FROM workspace_members WHERE workspace_id = {p}", (workspace_id,)
                )
            ]
            projects = _load_projects_sync(self, workspace_id)
            return Workspace(
                workspace_id=r["workspace_id"],
                name=r["name"],
                description=r["description"],
                allowed_skills=_jl(r["allowed_skills"]),
                system_prompt=r["system_prompt"],
                token_budget=int(r["token_budget"]),
                max_steps=int(r["max_steps"]),
                metadata=_jl(r["metadata"]) if isinstance(_jl(r["metadata"]), dict) else {},
                members=members,
                projects=projects,
                created_at=_dt(r["created_at"]),
                updated_at=_dt(r["updated_at"]),
            )

        return await self._a(_get)

    async def list_workspaces(self, user_id: str | None = None) -> list[Workspace]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _list() -> list[Workspace]:
            if user_id:
                ids = [
                    r["workspace_id"]
                    for r in self._exec(
                        f"SELECT workspace_id FROM workspace_members WHERE user_id = {p}", (user_id,)
                    )
                ]
                if not ids:
                    return []
                rows = self._exec(
                    f"SELECT * FROM workspaces WHERE workspace_id IN ({','.join([p]*len(ids))})",
                    tuple(ids),
                )
            else:
                rows = self._exec("SELECT * FROM workspaces", ())
            result = []
            for r in rows:
                members = [
                    WorkspaceMember(user_id=m["user_id"], role=WorkspaceRole(m["role"]),
                                    joined_at=_dt(m["joined_at"]))
                    for m in self._exec(
                        f"SELECT * FROM workspace_members WHERE workspace_id = {p}", (r["workspace_id"],)
                    )
                ]
                result.append(Workspace(
                    workspace_id=r["workspace_id"], name=r["name"],
                    description=r["description"],
                    allowed_skills=_jl(r["allowed_skills"]),
                    system_prompt=r["system_prompt"],
                    token_budget=int(r["token_budget"]), max_steps=int(r["max_steps"]),
                    metadata={}, members=members, projects=[],
                    created_at=_dt(r["created_at"]), updated_at=_dt(r["updated_at"]),
                ))
            return result

        return await self._a(_list)

    async def delete_workspace(self, workspace_id: str) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _del() -> None:
            for tbl in ("workspace_members", "workspaces"):
                self._exec(f"DELETE FROM {tbl} WHERE workspace_id = {p}", (workspace_id,))

        await self._a(_del)

    # ── 成员 ────────────────────────────────────────────────────

    async def add_workspace_member(self, workspace_id: str, member: WorkspaceMember) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _add() -> None:
            if self._dialect == _Dialect.SQLITE:
                self._exec(
                    f"INSERT OR REPLACE INTO workspace_members VALUES ({_ph(self._dialect, 4)})",
                    (workspace_id, member.user_id, member.role.value,
                     member.joined_at.strftime(_DT_FMT)),
                )
            else:
                self._exec(
                    f"INSERT INTO workspace_members VALUES ({_ph(self._dialect, 4)}) "
                    f"ON DUPLICATE KEY UPDATE role=VALUES(role)",
                    (workspace_id, member.user_id, member.role.value,
                     member.joined_at.strftime(_DT_FMT)),
                )

        await self._a(_add)

    async def remove_workspace_member(self, workspace_id: str, user_id: str) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        await self._a(
            self._exec,
            f"DELETE FROM workspace_members WHERE workspace_id = {p} AND user_id = {p}",
            (workspace_id, user_id),
        )

    async def update_workspace_member_role(self, workspace_id: str, user_id: str,
                                            role: WorkspaceRole) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        await self._a(
            self._exec,
            f"UPDATE workspace_members SET role = {p} WHERE workspace_id = {p} AND user_id = {p}",
            (role.value, workspace_id, user_id),
        )

    # ══════════════════════════════════════════════════════════════
    # Project CRUD
    # ══════════════════════════════════════════════════════════════

    async def save_project(self, proj: Project) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _save() -> None:
            conn = self._get_conn()
            cur  = conn.cursor()
            now  = datetime.utcnow().strftime(_DT_FMT)
            if self._dialect == _Dialect.SQLITE:
                cur.execute(
                    f"""INSERT INTO projects VALUES ({_ph(self._dialect, 11)})
                        ON CONFLICT(project_id) DO UPDATE SET
                        name=excluded.name, description=excluded.description,
                        allowed_skills=excluded.allowed_skills,
                        system_prompt=excluded.system_prompt,
                        token_budget=excluded.token_budget, max_steps=excluded.max_steps,
                        metadata=excluded.metadata, updated_at=excluded.updated_at""",
                    (proj.project_id, proj.workspace_id, proj.name, proj.description,
                     _jd(proj.allowed_skills), proj.system_prompt,
                     proj.token_budget, proj.max_steps, _jd(proj.metadata),
                     proj.created_at.strftime(_DT_FMT), now),
                )
            else:
                cur.execute(
                    f"""INSERT INTO projects VALUES ({_ph(self._dialect, 11)})
                        ON DUPLICATE KEY UPDATE
                        name=VALUES(name), description=VALUES(description),
                        allowed_skills=VALUES(allowed_skills),
                        system_prompt=VALUES(system_prompt),
                        token_budget=VALUES(token_budget), max_steps=VALUES(max_steps),
                        metadata=VALUES(metadata), updated_at=VALUES(updated_at)""",
                    (proj.project_id, proj.workspace_id, proj.name, proj.description,
                     _jd(proj.allowed_skills), proj.system_prompt,
                     proj.token_budget, proj.max_steps, _jd(proj.metadata),
                     proj.created_at.strftime(_DT_FMT), now),
                )
            cur.execute(f"DELETE FROM project_members WHERE project_id = {p}", (proj.project_id,))
            for m in proj.members:
                cur.execute(
                    f"INSERT INTO project_members VALUES ({_ph(self._dialect, 4)})",
                    (proj.project_id, m.user_id, m.role.value, m.joined_at.strftime(_DT_FMT)),
                )
            conn.commit()
            cur.close()

        await self._a(_save)

    async def get_project(self, project_id: str) -> Project | None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _get() -> Project | None:
            rows = self._exec(f"SELECT * FROM projects WHERE project_id = {p}", (project_id,))
            if not rows:
                return None
            return _row_to_project(self, rows[0])

        return await self._a(_get)

    async def list_projects(self, workspace_id: str) -> list[Project]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _list() -> list[Project]:
            rows = self._exec(
                f"SELECT * FROM projects WHERE workspace_id = {p}", (workspace_id,)
            )
            return [_row_to_project(self, r) for r in rows]

        return await self._a(_list)

    async def delete_project(self, project_id: str) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _del() -> None:
            self._exec(f"DELETE FROM project_members WHERE project_id = {p}", (project_id,))
            self._exec(f"DELETE FROM projects WHERE project_id = {p}", (project_id,))

        await self._a(_del)

    async def add_project_member(self, project_id: str, member: ProjectMember) -> None:
        if self._dialect == _Dialect.SQLITE:
            await self._a(
                self._exec,
                f"INSERT OR REPLACE INTO project_members VALUES ({_ph(self._dialect, 4)})",
                (project_id, member.user_id, member.role.value, member.joined_at.strftime(_DT_FMT)),
            )
        else:
            await self._a(
                self._exec,
                f"INSERT INTO project_members VALUES ({_ph(self._dialect, 4)}) "
                f"ON DUPLICATE KEY UPDATE role=VALUES(role)",
                (project_id, member.user_id, member.role.value, member.joined_at.strftime(_DT_FMT)),
            )

    async def remove_project_member(self, project_id: str, user_id: str) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        await self._a(
            self._exec,
            f"DELETE FROM project_members WHERE project_id = {p} AND user_id = {p}",
            (project_id, user_id),
        )

    # ══════════════════════════════════════════════════════════════
    # Memory 写入
    # ══════════════════════════════════════════════════════════════

    async def save_memory(self, entry: WorkspaceMemoryEntry) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        exp = entry.expires_at.strftime(_DT_FMT) if entry.expires_at else None

        if self._dialect == _Dialect.SQLITE:
            await self._a(
                self._exec,
                f"""INSERT INTO memory_entries VALUES ({_ph(self._dialect, 15)})
                    ON CONFLICT(entry_id) DO UPDATE SET
                    text=excluded.text, importance=excluded.importance,
                    tags=excluded.tags, metadata=excluded.metadata,
                    accessed_at=excluded.accessed_at""",
                (entry.entry_id, entry.scope.value, entry.owner_id,
                 entry.workspace_id, entry.project_id, entry.author_id,
                 entry.memory_type, entry.text, entry.importance,
                 _jd(entry.tags), _jd(entry.metadata), entry.access_count,
                 entry.created_at.strftime(_DT_FMT),
                 entry.accessed_at.strftime(_DT_FMT), exp),
            )
        else:
            await self._a(
                self._exec,
                f"""INSERT INTO memory_entries VALUES ({_ph(self._dialect, 15)})
                    ON DUPLICATE KEY UPDATE
                    text=VALUES(text), importance=VALUES(importance),
                    tags=VALUES(tags), metadata=VALUES(metadata),
                    accessed_at=VALUES(accessed_at)""",
                (entry.entry_id, entry.scope.value, entry.owner_id,
                 entry.workspace_id, entry.project_id, entry.author_id,
                 entry.memory_type, entry.text, entry.importance,
                 _jd(entry.tags), _jd(entry.metadata), entry.access_count,
                 entry.created_at.strftime(_DT_FMT),
                 entry.accessed_at.strftime(_DT_FMT), exp),
            )

    async def delete_memory(self, entry_id: str) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        await self._a(
            self._exec,
            f"DELETE FROM memory_entries WHERE entry_id = {p}", (entry_id,)
        )
        await self._a(
            self._exec,
            f"DELETE FROM memory_shares WHERE entry_id = {p}", (entry_id,)
        )

    async def touch_memory(self, entry_id: str) -> None:
        p  = "?" if self._dialect == _Dialect.SQLITE else "%s"
        now = datetime.utcnow().strftime(_DT_FMT)
        await self._a(
            self._exec,
            f"UPDATE memory_entries SET accessed_at = {p}, "
            f"access_count = access_count + 1 WHERE entry_id = {p}",
            (now, entry_id),
        )

    # ══════════════════════════════════════════════════════════════
    # Memory 读取（4 层）
    # ══════════════════════════════════════════════════════════════

    def _rows_to_entries(self, rows: list[dict]) -> list[WorkspaceMemoryEntry]:
        result = []
        for r in rows:
            result.append(WorkspaceMemoryEntry(
                entry_id=r["entry_id"],
                scope=MemoryScope(r["scope"]),
                owner_id=r["owner_id"],
                workspace_id=r.get("workspace_id"),
                project_id=r.get("project_id"),
                author_id=r.get("author_id", ""),
                memory_type=r.get("memory_type", "semantic"),
                text=r["text"],
                importance=float(r.get("importance", 0.5)),
                tags=_jl(r.get("tags", "[]")),
                metadata=_jl(r.get("metadata", "{}")),
                access_count=int(r.get("access_count", 0)),
                created_at=_dt(r.get("created_at")),
                accessed_at=_dt(r.get("accessed_at")),
            ))
        return result

    async def query_personal(self, user_id: str, limit: int = 200) -> list[WorkspaceMemoryEntry]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        rows = await self._a(
            self._exec,
            f"SELECT * FROM memory_entries WHERE scope = {p} AND owner_id = {p} "
            f"ORDER BY importance DESC LIMIT {limit}",
            (MemoryScope.PERSONAL.value, user_id),
        )
        return self._rows_to_entries(rows)

    async def query_project(self, workspace_id: str, project_id: str,
                             limit: int = 200) -> list[WorkspaceMemoryEntry]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        owner = Workspace.proj_memory_key(workspace_id, project_id)
        rows = await self._a(
            self._exec,
            f"SELECT * FROM memory_entries WHERE scope = {p} AND owner_id = {p} "
            f"ORDER BY importance DESC LIMIT {limit}",
            (MemoryScope.PROJECT.value, owner),
        )
        return self._rows_to_entries(rows)

    async def query_workspace(self, workspace_id: str,
                               limit: int = 200) -> list[WorkspaceMemoryEntry]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        rows = await self._a(
            self._exec,
            f"SELECT * FROM memory_entries WHERE scope = {p} AND workspace_id = {p} "
            f"ORDER BY importance DESC LIMIT {limit}",
            (MemoryScope.WORKSPACE.value, workspace_id),
        )
        return self._rows_to_entries(rows)

    async def query_shared_to_project(self, project_id: str,
                                       limit: int = 200) -> list[WorkspaceMemoryEntry]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        rows = await self._a(
            self._exec,
            f"""SELECT me.* FROM memory_entries me
                INNER JOIN memory_shares ms ON me.entry_id = ms.entry_id
                WHERE ms.shared_to_project_id = {p}
                ORDER BY me.importance DESC LIMIT {limit}""",
            (project_id,),
        )
        return self._rows_to_entries(rows)

    # ══════════════════════════════════════════════════════════════
    # 跨项目共享
    # ══════════════════════════════════════════════════════════════

    async def share_memory(self, share: MemoryShare) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        if self._dialect == _Dialect.SQLITE:
            await self._a(
                self._exec,
                f"INSERT OR REPLACE INTO memory_shares VALUES ({_ph(self._dialect, 6)})",
                (share.entry_id, share.shared_to_project_id, share.shared_by_user_id,
                 share.shared_at.strftime(_DT_FMT), share.permission, share.note),
            )
        else:
            await self._a(
                self._exec,
                f"INSERT INTO memory_shares VALUES ({_ph(self._dialect, 6)}) "
                f"ON DUPLICATE KEY UPDATE permission=VALUES(permission), note=VALUES(note)",
                (share.entry_id, share.shared_to_project_id, share.shared_by_user_id,
                 share.shared_at.strftime(_DT_FMT), share.permission, share.note),
            )

    async def revoke_share(self, entry_id: str, project_id: str) -> None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        await self._a(
            self._exec,
            f"DELETE FROM memory_shares WHERE entry_id = {p} AND shared_to_project_id = {p}",
            (entry_id, project_id),
        )

    async def list_shares(self, entry_id: str) -> list[MemoryShare]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        rows = await self._a(
            self._exec,
            f"SELECT * FROM memory_shares WHERE entry_id = {p}", (entry_id,)
        )
        return [
            MemoryShare(
                entry_id=r["entry_id"],
                shared_to_project_id=r["shared_to_project_id"],
                shared_by_user_id=r.get("shared_by_user_id", ""),
                shared_at=_dt(r.get("shared_at")),
                permission=r.get("permission", "read"),
                note=r.get("note", ""),
            )
            for r in rows
        ]

    async def list_shares_to_project(self, project_id: str) -> list[MemoryShare]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        rows = await self._a(
            self._exec,
            f"SELECT * FROM memory_shares WHERE shared_to_project_id = {p}", (project_id,)
        )
        return [
            MemoryShare(
                entry_id=r["entry_id"],
                shared_to_project_id=r["shared_to_project_id"],
                shared_by_user_id=r.get("shared_by_user_id", ""),
                shared_at=_dt(r.get("shared_at")),
                permission=r.get("permission", "read"),
                note=r.get("note", ""),
            )
            for r in rows
        ]

    # ══════════════════════════════════════════════════════════════
    # 用户画像
    # ══════════════════════════════════════════════════════════════

    async def get_profile(self, user_id: str, workspace_id: str = "") -> dict[str, Any]:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        rows = await self._a(
            self._exec,
            f"SELECT profile_data FROM user_profiles WHERE user_id = {p} AND workspace_id = {p}",
            (user_id, workspace_id),
        )
        if rows:
            return _jl(rows[0]["profile_data"]) or {}
        return {}

    async def update_profile(self, user_id: str, workspace_id: str,
                              data: dict[str, Any]) -> None:
        p  = "?" if self._dialect == _Dialect.SQLITE else "%s"
        now = datetime.utcnow().strftime(_DT_FMT)
        existing = await self.get_profile(user_id, workspace_id)
        merged = {**existing, **data}
        if self._dialect == _Dialect.SQLITE:
            await self._a(
                self._exec,
                f"INSERT OR REPLACE INTO user_profiles VALUES ({_ph(self._dialect, 4)})",
                (user_id, workspace_id, _jd(merged), now),
            )
        else:
            await self._a(
                self._exec,
                f"INSERT INTO user_profiles VALUES ({_ph(self._dialect, 4)}) "
                f"ON DUPLICATE KEY UPDATE profile_data=VALUES(profile_data), updated_at=VALUES(updated_at)",
                (user_id, workspace_id, _jd(merged), now),
            )

    # ══════════════════════════════════════════════════════════════
    # 裁剪（按重要性得分淘汰旧记忆）
    # ══════════════════════════════════════════════════════════════

    async def prune_personal(self, user_id: str, max_items: int,
                              score_threshold: float) -> int:
        """删除低于阈值的个人记忆，并限制总条数。"""
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"

        def _prune() -> int:
            rows = self._exec(
                f"SELECT entry_id, importance, created_at FROM memory_entries "
                f"WHERE scope = {p} AND owner_id = {p} ORDER BY importance DESC",
                (MemoryScope.PERSONAL.value, user_id),
            )
            to_del = [r["entry_id"] for r in rows if float(r["importance"]) < score_threshold]
            to_del += [r["entry_id"] for r in rows[max_items:]]
            to_del = list(set(to_del))
            for eid in to_del:
                self._exec(f"DELETE FROM memory_entries WHERE entry_id = {p}", (eid,))
                self._exec(f"DELETE FROM memory_shares  WHERE entry_id = {p}", (eid,))
            return len(to_del)

        return await self._a(_prune)

    async def prune_project(self, workspace_id: str, project_id: str,
                             max_items: int) -> int:
        p  = "?" if self._dialect == _Dialect.SQLITE else "%s"
        owner = Workspace.proj_memory_key(workspace_id, project_id)

        def _prune() -> int:
            rows = self._exec(
                f"SELECT entry_id FROM memory_entries "
                f"WHERE scope = {p} AND owner_id = {p} ORDER BY importance DESC",
                (MemoryScope.PROJECT.value, owner),
            )
            to_del = [r["entry_id"] for r in rows[max_items:]]
            for eid in to_del:
                self._exec(f"DELETE FROM memory_entries WHERE entry_id = {p}", (eid,))
                self._exec(f"DELETE FROM memory_shares  WHERE entry_id = {p}", (eid,))
            return len(to_del)

        return await self._a(_prune)

    # ══════════════════════════════════════════════════════════════
    # Session
    # ══════════════════════════════════════════════════════════════

    async def save_session(self, session_id: str, user_id: str,
                            workspace_id: str | None, project_id: str | None) -> None:
        p   = "?" if self._dialect == _Dialect.SQLITE else "%s"
        now = datetime.utcnow().strftime(_DT_FMT)
        if self._dialect == _Dialect.SQLITE:
            await self._a(
                self._exec,
                f"INSERT OR REPLACE INTO sessions VALUES ({_ph(self._dialect, 7)})",
                (session_id, user_id, workspace_id, project_id, now, now, "{}"),
            )
        else:
            await self._a(
                self._exec,
                f"INSERT INTO sessions VALUES ({_ph(self._dialect, 7)}) "
                f"ON DUPLICATE KEY UPDATE last_active_at=VALUES(last_active_at)",
                (session_id, user_id, workspace_id, project_id, now, now, "{}"),
            )

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        p = "?" if self._dialect == _Dialect.SQLITE else "%s"
        rows = await self._a(
            self._exec,
            f"SELECT * FROM sessions WHERE session_id = {p}", (session_id,)
        )
        return rows[0] if rows else None


# ──────────────────────────────────────────────────────────────
# 内部辅助（避免循环引用）
# ──────────────────────────────────────────────────────────────

def _row_to_project(store: _BaseRelationalStore, r: dict) -> Project:
    p = "?" if store._dialect == _Dialect.SQLITE else "%s"
    members = [
        ProjectMember(
            user_id=m["user_id"],
            role=ProjectRole(m["role"]),
            joined_at=_dt(m["joined_at"]),
        )
        for m in store._exec(
            f"SELECT * FROM project_members WHERE project_id = {p}", (r["project_id"],)
        )
    ]
    return Project(
        project_id=r["project_id"],
        workspace_id=r["workspace_id"],
        name=r["name"],
        description=r["description"],
        allowed_skills=_jl(r["allowed_skills"]),
        system_prompt=r["system_prompt"],
        token_budget=int(r["token_budget"]),
        max_steps=int(r["max_steps"]),
        metadata={},
        members=members,
        created_at=_dt(r["created_at"]),
        updated_at=_dt(r["updated_at"]),
    )


def _load_projects_sync(store: _BaseRelationalStore, workspace_id: str) -> list[Project]:
    p = "?" if store._dialect == _Dialect.SQLITE else "%s"
    rows = store._exec(
        f"SELECT * FROM projects WHERE workspace_id = {p}", (workspace_id,)
    )
    return [_row_to_project(store, r) for r in rows]


# ══════════════════════════════════════════════════════════════
# SQLite 实现
# ══════════════════════════════════════════════════════════════

class SQLiteWorkspaceStore(_BaseRelationalStore):
    """
    使用 stdlib sqlite3，零额外依赖。

    db_path: 文件路径，如 "workspace.db" 或 ":memory:"（测试用）
    """

    def __init__(self, db_path: str = "workspace.db") -> None:
        super().__init__(_Dialect.SQLITE)
        self._db_path = db_path
        self._conn: Any = None  # 懒初始化，_get_conn() 时创建

    def _get_conn(self) -> Any:
        import sqlite3
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,  # 允许跨线程（asyncio.to_thread 需要）
                isolation_level=None,     # autocommit，由我们手动 commit
            )
            self._conn.row_factory = sqlite3.Row
            # 开启 WAL 模式提升并发性能
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            # 关闭 autocommit，启用手动事务
            self._conn.isolation_level = ""
        return self._conn


# ══════════════════════════════════════════════════════════════
# MySQL 实现
# ══════════════════════════════════════════════════════════════

class MySQLWorkspaceStore(_BaseRelationalStore):
    """
    使用 pymysql（同步驱动，包装为 async via asyncio.to_thread）。

    安装：pip install pymysql
    可选更高性能的纯异步驱动：pip install aiomysql（需改写连接层）

    连接示例：
        store = MySQLWorkspaceStore(
            host="127.0.0.1", port=3306,
            user="root", password="xxx", database="agent_db",
            charset="utf8mb4",
        )
    """

    def __init__(
        self,
        host:     str = "127.0.0.1",
        port:     int = 3306,
        user:     str = "root",
        password: str = "",
        database: str = "agent",
        charset:  str = "utf8mb4",
    ) -> None:
        super().__init__(_Dialect.MYSQL)
        self._conn_kwargs = dict(
            host=host, port=port, user=user, password=password,
            database=database, charset=charset,
            autocommit=False,
        )
        self._conn: Any = None

    def _get_conn(self) -> Any:
        try:
            import pymysql
            import pymysql.cursors
        except ImportError as e:
            raise ImportError("MySQLWorkspaceStore 需要 pymysql：pip install pymysql") from e

        if self._conn is None or not self._conn.open:
            self._conn = pymysql.connect(
                **self._conn_kwargs,
                cursorclass=pymysql.cursors.DictCursor,
            )
        return self._conn

    # MySQL 的 CREATE INDEX 语法：只在索引不存在时创建需要 INFORMATION_SCHEMA 判断，
    # 这里改用 ignore exception 策略，父类 initialize() 已处理。


# ══════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════

def create_store(url: str | None = None, **kwargs: Any) -> WorkspaceStore:
    """
    根据 URL 创建存储实例。

    Examples:
        create_store()                                    # SQLite workspace.db
        create_store("sqlite:///workspace.db")            # SQLite 指定路径
        create_store("sqlite:///:memory:")               # 内存（测试）
        create_store("mysql://root:pw@localhost/agent")  # MySQL
        create_store(host="db", user="u", ...)           # MySQL kwargs
    """
    if not url and not kwargs:
        return SQLiteWorkspaceStore("workspace.db")

    if url:
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
            return SQLiteWorkspaceStore(path)
        if url.startswith("sqlite://"):
            return SQLiteWorkspaceStore(":memory:")
        if url.startswith("mysql://") or url.startswith("mysql+pymysql://"):
            import urllib.parse
            parsed = urllib.parse.urlparse(url.replace("mysql+pymysql://", "mysql://"))
            return MySQLWorkspaceStore(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 3306,
                user=parsed.username or "root",
                password=parsed.password or "",
                database=(parsed.path or "/agent").lstrip("/"),
            )

    return MySQLWorkspaceStore(**kwargs)
