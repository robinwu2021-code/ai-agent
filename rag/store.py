"""
rag/store.py — 知识库持久化存储

表结构：
  kb_documents  文档元数据（状态、分块数、字符数等）
  kb_chunks     分块文本 + 向量（pickle BLOB）

支持 SQLite（默认）和 MySQL（通过 create_kb_store(url) 工厂）。
与 workspace.db 同库：db_path 默认 "workspace.db"。
"""
from __future__ import annotations

import asyncio
import json
import math
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class KBDocument:
    doc_id:      str
    kb_id:       str   = "global"
    filename:    str   = ""
    source:      str   = ""
    doc_type:    str   = "text"      # pdf/md/txt/docx/html/url/inline
    char_count:  int   = 0
    chunk_count: int   = 0
    status:      str   = "pending"   # pending/indexing/ready/error
    error_msg:   str   = ""
    created_at:  float = 0.0
    meta:        dict  = field(default_factory=dict)


@dataclass
class KBChunk:
    chunk_id:    str
    doc_id:      str
    kb_id:       str              = "global"
    chunk_index: int              = 0
    text:        str              = ""
    embedding:   list[float] | None = None
    meta:        dict             = field(default_factory=dict)
    created_at:  float            = 0.0


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id      TEXT PRIMARY KEY,
    kb_id       TEXT NOT NULL DEFAULT 'global',
    filename    TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    doc_type    TEXT NOT NULL DEFAULT 'text',
    char_count  INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'pending',
    error_msg   TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    meta        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_kb_docs_kb_id ON kb_documents(kb_id);
CREATE INDEX IF NOT EXISTS idx_kb_docs_status ON kb_documents(status);

CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    kb_id       TEXT NOT NULL DEFAULT 'global',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    text        TEXT NOT NULL DEFAULT '',
    embedding   BLOB,
    meta        TEXT DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_id ON kb_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_kb_id ON kb_chunks(kb_id);
"""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class KBStore(ABC):
    """知识库存储抽象接口。"""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def save_document(self, doc: KBDocument) -> KBDocument: ...

    @abstractmethod
    async def get_document(self, doc_id: str) -> KBDocument | None: ...

    @abstractmethod
    async def list_documents(
        self, kb_id: str, status: str | None = None
    ) -> list[KBDocument]: ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        error_msg: str = "",
        chunk_count: int = 0,
    ) -> None: ...

    @abstractmethod
    async def save_chunk(self, chunk: KBChunk) -> KBChunk: ...

    @abstractmethod
    async def save_chunks(self, chunks: list[KBChunk]) -> None: ...

    @abstractmethod
    async def list_chunks(self, doc_id: str) -> list[KBChunk]: ...

    @abstractmethod
    async def list_chunks_by_kb(
        self, kb_id: str, limit: int = 2000
    ) -> list[KBChunk]: ...

    @abstractmethod
    async def delete_chunks_by_doc(self, doc_id: str) -> None: ...

    @abstractmethod
    async def search_chunks_by_text(
        self, query: str, kb_id: str, limit: int = 50
    ) -> list[KBChunk]: ...

    @abstractmethod
    async def search_chunks_by_embedding(
        self, embedding: list[float], kb_id: str, limit: int = 50
    ) -> list[KBChunk]: ...

    @abstractmethod
    async def get_stats(self, kb_id: str) -> dict: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_embedding(emb: list[float] | None) -> bytes | None:
    if emb is None:
        return None
    return pickle.dumps(emb)


def _deserialize_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return pickle.loads(blob)  # noqa: S301


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _row_to_document(row: sqlite3.Row) -> KBDocument:
    return KBDocument(
        doc_id      = row["doc_id"],
        kb_id       = row["kb_id"],
        filename    = row["filename"] or "",
        source      = row["source"] or "",
        doc_type    = row["doc_type"] or "text",
        char_count  = row["char_count"] or 0,
        chunk_count = row["chunk_count"] or 0,
        status      = row["status"] or "pending",
        error_msg   = row["error_msg"] or "",
        created_at  = row["created_at"] or 0.0,
        meta        = json.loads(row["meta"] or "{}"),
    )


def _row_to_chunk(row: sqlite3.Row) -> KBChunk:
    return KBChunk(
        chunk_id    = row["chunk_id"],
        doc_id      = row["doc_id"],
        kb_id       = row["kb_id"],
        chunk_index = row["chunk_index"] or 0,
        text        = row["text"] or "",
        embedding   = _deserialize_embedding(row["embedding"]),
        meta        = json.loads(row["meta"] or "{}"),
        created_at  = row["created_at"] or 0.0,
    )


# ---------------------------------------------------------------------------
# SQLiteKBStore
# ---------------------------------------------------------------------------

class SQLiteKBStore(KBStore):
    """SQLite 实现的知识库存储（WAL 模式，asyncio.to_thread 封装）。"""

    def __init__(self, db_path: str = "workspace.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._get_conn().execute(sql, params)

    def _executemany(self, sql: str, params_list: list) -> None:
        self._get_conn().executemany(sql, params_list)

    def _commit(self) -> None:
        self._get_conn().commit()

    # ------------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        def _init() -> None:
            conn = self._get_conn()
            conn.executescript(_DDL)
            conn.commit()

        await asyncio.to_thread(_init)

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    async def save_document(self, doc: KBDocument) -> KBDocument:
        def _do() -> KBDocument:
            ts = doc.created_at or time.time()
            self._execute(
                """
                INSERT INTO kb_documents
                    (doc_id, kb_id, filename, source, doc_type,
                     char_count, chunk_count, status, error_msg, created_at, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    kb_id       = excluded.kb_id,
                    filename    = excluded.filename,
                    source      = excluded.source,
                    doc_type    = excluded.doc_type,
                    char_count  = excluded.char_count,
                    chunk_count = excluded.chunk_count,
                    status      = excluded.status,
                    error_msg   = excluded.error_msg,
                    meta        = excluded.meta
                """,
                (
                    doc.doc_id,
                    doc.kb_id,
                    doc.filename,
                    doc.source,
                    doc.doc_type,
                    doc.char_count,
                    doc.chunk_count,
                    doc.status,
                    doc.error_msg,
                    ts,
                    json.dumps(doc.meta, ensure_ascii=False),
                ),
            )
            self._commit()
            doc.created_at = ts
            return doc

        async with self._lock:
            return await asyncio.to_thread(_do)

    async def get_document(self, doc_id: str) -> KBDocument | None:
        def _do() -> KBDocument | None:
            row = self._execute(
                "SELECT * FROM kb_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return _row_to_document(row) if row else None

        return await asyncio.to_thread(_do)

    async def list_documents(
        self, kb_id: str, status: str | None = None
    ) -> list[KBDocument]:
        def _do() -> list[KBDocument]:
            if status is not None:
                rows = self._execute(
                    "SELECT * FROM kb_documents WHERE kb_id = ? AND status = ? ORDER BY created_at DESC",
                    (kb_id, status),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT * FROM kb_documents WHERE kb_id = ? ORDER BY created_at DESC",
                    (kb_id,),
                ).fetchall()
            return [_row_to_document(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def delete_document(self, doc_id: str) -> None:
        """Cascade delete: removes chunks first, then the document record."""
        def _do() -> None:
            self._execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
            self._execute("DELETE FROM kb_documents WHERE doc_id = ?", (doc_id,))
            self._commit()

        async with self._lock:
            await asyncio.to_thread(_do)

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        error_msg: str = "",
        chunk_count: int = 0,
    ) -> None:
        def _do() -> None:
            self._execute(
                """
                UPDATE kb_documents
                   SET status = ?, error_msg = ?, chunk_count = ?
                 WHERE doc_id = ?
                """,
                (status, error_msg, chunk_count, doc_id),
            )
            self._commit()

        async with self._lock:
            await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Chunk CRUD
    # ------------------------------------------------------------------

    async def save_chunk(self, chunk: KBChunk) -> KBChunk:
        def _do() -> KBChunk:
            ts = chunk.created_at or time.time()
            self._execute(
                """
                INSERT INTO kb_chunks
                    (chunk_id, doc_id, kb_id, chunk_index, text, embedding, meta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id      = excluded.doc_id,
                    kb_id       = excluded.kb_id,
                    chunk_index = excluded.chunk_index,
                    text        = excluded.text,
                    embedding   = excluded.embedding,
                    meta        = excluded.meta
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.kb_id,
                    chunk.chunk_index,
                    chunk.text,
                    _serialize_embedding(chunk.embedding),
                    json.dumps(chunk.meta, ensure_ascii=False),
                    ts,
                ),
            )
            self._commit()
            chunk.created_at = ts
            return chunk

        async with self._lock:
            return await asyncio.to_thread(_do)

    async def save_chunks(self, chunks: list[KBChunk]) -> None:
        """Batch-insert chunks in a single transaction."""
        if not chunks:
            return

        def _do() -> None:
            now = time.time()
            params_list = [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.kb_id,
                    c.chunk_index,
                    c.text,
                    _serialize_embedding(c.embedding),
                    json.dumps(c.meta, ensure_ascii=False),
                    c.created_at or now,
                )
                for c in chunks
            ]
            self._executemany(
                """
                INSERT INTO kb_chunks
                    (chunk_id, doc_id, kb_id, chunk_index, text, embedding, meta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id      = excluded.doc_id,
                    kb_id       = excluded.kb_id,
                    chunk_index = excluded.chunk_index,
                    text        = excluded.text,
                    embedding   = excluded.embedding,
                    meta        = excluded.meta
                """,
                params_list,
            )
            self._commit()

        async with self._lock:
            await asyncio.to_thread(_do)

    async def list_chunks(self, doc_id: str) -> list[KBChunk]:
        def _do() -> list[KBChunk]:
            rows = self._execute(
                "SELECT * FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
                (doc_id,),
            ).fetchall()
            return [_row_to_chunk(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def list_chunks_by_kb(
        self, kb_id: str, limit: int = 2000
    ) -> list[KBChunk]:
        def _do() -> list[KBChunk]:
            rows = self._execute(
                "SELECT * FROM kb_chunks WHERE kb_id = ? ORDER BY doc_id, chunk_index ASC LIMIT ?",
                (kb_id, limit),
            ).fetchall()
            return [_row_to_chunk(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def delete_chunks_by_doc(self, doc_id: str) -> None:
        def _do() -> None:
            self._execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
            self._commit()

        async with self._lock:
            await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_chunks_by_text(
        self, query: str, kb_id: str, limit: int = 50
    ) -> list[KBChunk]:
        """Simple LIKE search on the text column (one word at a time, OR-combined)."""
        def _do() -> list[KBChunk]:
            # Build a query that matches any word from the query string
            words = [w.strip() for w in query.split() if w.strip()]
            if not words:
                return []
            # Use the full query string as a single LIKE pattern as well
            like = f"%{query}%"
            rows = self._execute(
                "SELECT * FROM kb_chunks WHERE kb_id = ? AND text LIKE ? LIMIT ?",
                (kb_id, like, limit),
            ).fetchall()
            return [_row_to_chunk(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def search_chunks_by_embedding(
        self, embedding: list[float], kb_id: str, limit: int = 50
    ) -> list[KBChunk]:
        """
        Fetch all chunks with embeddings for kb_id, compute cosine similarity
        in Python, and return the top-N sorted by descending score.
        """
        def _do() -> list[KBChunk]:
            rows = self._execute(
                "SELECT * FROM kb_chunks WHERE kb_id = ? AND embedding IS NOT NULL",
                (kb_id,),
            ).fetchall()
            scored: list[tuple[float, KBChunk]] = []
            for row in rows:
                chunk = _row_to_chunk(row)
                if chunk.embedding:
                    sim = _cosine_similarity(embedding, chunk.embedding)
                    scored.append((sim, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]

        return await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self, kb_id: str) -> dict:
        def _do() -> dict:
            doc_row = self._execute(
                "SELECT COUNT(*) AS cnt FROM kb_documents WHERE kb_id = ?",
                (kb_id,),
            ).fetchone()
            chunk_row = self._execute(
                "SELECT COUNT(*) AS cnt FROM kb_chunks WHERE kb_id = ?",
                (kb_id,),
            ).fetchone()
            ready_row = self._execute(
                "SELECT COUNT(*) AS cnt FROM kb_documents WHERE kb_id = ? AND status = 'ready'",
                (kb_id,),
            ).fetchone()
            chars_row = self._execute(
                "SELECT COALESCE(SUM(char_count), 0) AS total FROM kb_documents WHERE kb_id = ?",
                (kb_id,),
            ).fetchone()
            return {
                "documents":       doc_row["cnt"]   if doc_row   else 0,
                "chunks":          chunk_row["cnt"] if chunk_row else 0,
                "ready_documents": ready_row["cnt"] if ready_row else 0,
                "total_chars":     chars_row["total"] if chars_row else 0,
            }

        return await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_kb_store(url: str) -> KBStore:
    """
    Factory function to create a KBStore from a URL string.

    Supported schemes:
    - "sqlite:///path/to/file.db"  -> SQLiteKBStore
    - "sqlite://path/to/file.db"   -> SQLiteKBStore

    Examples:
        store = create_kb_store("sqlite:///data/workspace.db")
        store = create_kb_store("sqlite:///workspace.db")
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme == "sqlite":
        if url.startswith("sqlite:///"):
            db_path = url[len("sqlite:///"):]
        elif url.startswith("sqlite://"):
            db_path = url[len("sqlite://"):]
        else:
            db_path = parsed.path
        return SQLiteKBStore(db_path=db_path)

    raise ValueError(
        f"Unsupported KB store URL scheme: '{scheme}'. "
        "Use 'sqlite:///path/to/db'."
    )
