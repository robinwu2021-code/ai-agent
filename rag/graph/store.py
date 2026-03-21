"""rag/graph/store.py — 知识图谱存储层"""
from __future__ import annotations

import asyncio
import json
import math
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any
from urllib.parse import urlparse

from rag.graph.models import Community, Edge, Node, NodeType


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class KGStore(ABC):
    """知识图谱存储抽象接口。"""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def upsert_node(self, node: Node) -> Node: ...

    @abstractmethod
    async def upsert_edge(self, edge: Edge) -> Edge: ...

    @abstractmethod
    async def get_node(self, node_id: str) -> Node | None: ...

    @abstractmethod
    async def get_node_by_name(self, name: str, kb_id: str) -> Node | None: ...

    @abstractmethod
    async def list_nodes(
        self,
        kb_id: str,
        node_type: NodeType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Node]: ...

    @abstractmethod
    async def list_edges(
        self,
        kb_id: str,
        src_id: str | None = None,
        dst_id: str | None = None,
    ) -> list[Edge]: ...

    @abstractmethod
    async def get_neighbors(
        self, node_id: str, kb_id: str, hops: int = 1
    ) -> tuple[list[Node], list[Edge]]: ...

    @abstractmethod
    async def search_nodes_by_text(
        self, query: str, kb_id: str, limit: int = 20
    ) -> list[Node]: ...

    @abstractmethod
    async def search_nodes_by_embedding(
        self, embedding: list[float], kb_id: str, limit: int = 20
    ) -> list[Node]: ...

    @abstractmethod
    async def delete_node(self, node_id: str) -> None: ...

    @abstractmethod
    async def delete_edge(self, edge_id: str) -> None: ...

    @abstractmethod
    async def delete_by_doc(self, doc_id: str, kb_id: str) -> None: ...

    @abstractmethod
    async def upsert_community(self, community: Community) -> Community: ...

    @abstractmethod
    async def list_communities(
        self, kb_id: str, level: int | None = None
    ) -> list[Community]: ...

    @abstractmethod
    async def search_communities_by_embedding(
        self,
        embedding: list[float],
        kb_id: str,
        level: int | None = None,
        limit: int = 10,
    ) -> list[Community]: ...

    @abstractmethod
    async def get_stats(self, kb_id: str) -> dict: ...

    @abstractmethod
    async def get_full_graph(
        self, kb_id: str, limit: int = 500
    ) -> tuple[list[Node], list[Edge]]: ...

    @abstractmethod
    async def get_path(
        self, src_id: str, dst_id: str, kb_id: str, max_hops: int = 4
    ) -> list[Edge]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure Python cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _serialize_embedding(emb: list[float] | None) -> bytes | None:
    if emb is None:
        return None
    return pickle.dumps(emb)


def _deserialize_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return pickle.loads(blob)  # noqa: S301


def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        id=row["id"],
        kb_id=row["kb_id"],
        name=row["name"],
        node_type=NodeType(row["node_type"]) if row["node_type"] else NodeType.OTHER,
        aliases=json.loads(row["aliases"] or "[]"),
        description=row["description"] or "",
        doc_ids=json.loads(row["doc_ids"] or "[]"),
        embedding=_deserialize_embedding(row["embedding"]),
        degree=row["degree"] or 0,
        created_at=row["created_at"] or 0.0,
    )


def _row_to_edge(row: sqlite3.Row) -> Edge:
    return Edge(
        id=row["id"],
        kb_id=row["kb_id"],
        src_id=row["src_id"],
        dst_id=row["dst_id"],
        relation=row["relation"],
        weight=row["weight"] if row["weight"] is not None else 1.0,
        doc_id=row["doc_id"] or "",
        chunk_id=row["chunk_id"] or "",
        context=row["context"] or "",
        created_at=row["created_at"] or 0.0,
    )


def _row_to_community(row: sqlite3.Row) -> Community:
    return Community(
        id=row["id"],
        kb_id=row["kb_id"],
        node_ids=json.loads(row["node_ids"] or "[]"),
        summary=row["summary"] or "",
        level=row["level"] or 0,
        embedding=_deserialize_embedding(row["embedding"]),
        created_at=row["created_at"] or 0.0,
    )


# ---------------------------------------------------------------------------
# SQLiteKGStore
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS kg_nodes (
    id          TEXT PRIMARY KEY,
    kb_id       TEXT NOT NULL,
    name        TEXT NOT NULL,
    node_type   TEXT DEFAULT 'OTHER',
    aliases     TEXT DEFAULT '[]',
    description TEXT DEFAULT '',
    doc_ids     TEXT DEFAULT '[]',
    embedding   BLOB,
    degree      INTEGER DEFAULT 0,
    created_at  REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_kb_id ON kg_nodes(kb_id);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_name ON kg_nodes(name);

CREATE TABLE IF NOT EXISTS kg_edges (
    id          TEXT PRIMARY KEY,
    kb_id       TEXT NOT NULL,
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    doc_id      TEXT DEFAULT '',
    chunk_id    TEXT DEFAULT '',
    context     TEXT DEFAULT '',
    created_at  REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_kg_edges_kb_id  ON kg_edges(kb_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_src_id ON kg_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_dst_id ON kg_edges(dst_id);

CREATE TABLE IF NOT EXISTS kg_communities (
    id          TEXT PRIMARY KEY,
    kb_id       TEXT NOT NULL,
    node_ids    TEXT DEFAULT '[]',
    summary     TEXT DEFAULT '',
    level       INTEGER DEFAULT 0,
    embedding   BLOB,
    created_at  REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_kg_communities_kb_id ON kg_communities(kb_id);
"""


class SQLiteKGStore(KGStore):
    """SQLite 实现的知识图谱存储（WAL 模式，asyncio.to_thread 封装）。"""

    def __init__(self, db_path: str) -> None:
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
    # Node CRUD
    # ------------------------------------------------------------------

    async def upsert_node(self, node: Node) -> Node:
        def _do() -> Node:
            self._execute(
                """
                INSERT INTO kg_nodes
                    (id, kb_id, name, node_type, aliases, description, doc_ids, embedding, degree, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name        = excluded.name,
                    node_type   = excluded.node_type,
                    aliases     = excluded.aliases,
                    description = excluded.description,
                    doc_ids     = excluded.doc_ids,
                    embedding   = excluded.embedding,
                    degree      = excluded.degree
                """,
                (
                    node.id,
                    node.kb_id,
                    node.name,
                    node.node_type.value,
                    json.dumps(node.aliases, ensure_ascii=False),
                    node.description,
                    json.dumps(node.doc_ids, ensure_ascii=False),
                    _serialize_embedding(node.embedding),
                    node.degree,
                    node.created_at or time.time(),
                ),
            )
            self._commit()
            return node

        async with self._lock:
            return await asyncio.to_thread(_do)

    async def get_node(self, node_id: str) -> Node | None:
        def _do() -> Node | None:
            row = self._execute(
                "SELECT * FROM kg_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return _row_to_node(row) if row else None

        return await asyncio.to_thread(_do)

    async def get_node_by_name(self, name: str, kb_id: str) -> Node | None:
        def _do() -> Node | None:
            # exact match first
            row = self._execute(
                "SELECT * FROM kg_nodes WHERE name = ? AND kb_id = ?",
                (name, kb_id),
            ).fetchone()
            if row:
                return _row_to_node(row)
            # alias match
            rows = self._execute(
                "SELECT * FROM kg_nodes WHERE kb_id = ? AND aliases LIKE ?",
                (kb_id, f'%"{name}"%'),
            ).fetchall()
            for r in rows:
                node = _row_to_node(r)
                if name in node.aliases:
                    return node
            return None

        return await asyncio.to_thread(_do)

    async def list_nodes(
        self,
        kb_id: str,
        node_type: NodeType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Node]:
        def _do() -> list[Node]:
            if node_type is not None:
                rows = self._execute(
                    "SELECT * FROM kg_nodes WHERE kb_id = ? AND node_type = ? LIMIT ? OFFSET ?",
                    (kb_id, node_type.value, limit, offset),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT * FROM kg_nodes WHERE kb_id = ? LIMIT ? OFFSET ?",
                    (kb_id, limit, offset),
                ).fetchall()
            return [_row_to_node(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def delete_node(self, node_id: str) -> None:
        def _do() -> None:
            # fetch connected edges to update degrees
            edges = self._execute(
                "SELECT src_id, dst_id FROM kg_edges WHERE src_id = ? OR dst_id = ?",
                (node_id, node_id),
            ).fetchall()
            neighbour_ids = set()
            for e in edges:
                if e["src_id"] != node_id:
                    neighbour_ids.add(e["src_id"])
                if e["dst_id"] != node_id:
                    neighbour_ids.add(e["dst_id"])
            # delete edges
            self._execute(
                "DELETE FROM kg_edges WHERE src_id = ? OR dst_id = ?",
                (node_id, node_id),
            )
            # delete node
            self._execute("DELETE FROM kg_nodes WHERE id = ?", (node_id,))
            # update neighbour degrees
            for nid in neighbour_ids:
                self._recalc_degree(nid)
            self._commit()

        async with self._lock:
            await asyncio.to_thread(_do)

    def _recalc_degree(self, node_id: str) -> None:
        row = self._execute(
            "SELECT COUNT(*) AS cnt FROM kg_edges WHERE src_id = ? OR dst_id = ?",
            (node_id, node_id),
        ).fetchone()
        deg = row["cnt"] if row else 0
        self._execute(
            "UPDATE kg_nodes SET degree = ? WHERE id = ?", (deg, node_id)
        )

    async def delete_by_doc(self, doc_id: str, kb_id: str) -> None:
        def _do() -> None:
            # delete edges referencing this doc
            self._execute(
                "DELETE FROM kg_edges WHERE doc_id = ? AND kb_id = ?",
                (doc_id, kb_id),
            )
            # remove doc_id from node doc_ids; delete node if doc_ids becomes empty
            rows = self._execute(
                "SELECT id, doc_ids FROM kg_nodes WHERE kb_id = ? AND doc_ids LIKE ?",
                (kb_id, f'%"{doc_id}"%'),
            ).fetchall()
            for row in rows:
                doc_ids: list[str] = json.loads(row["doc_ids"] or "[]")
                if doc_id in doc_ids:
                    doc_ids.remove(doc_id)
                if not doc_ids:
                    self._execute("DELETE FROM kg_nodes WHERE id = ?", (row["id"],))
                else:
                    self._execute(
                        "UPDATE kg_nodes SET doc_ids = ? WHERE id = ?",
                        (json.dumps(doc_ids, ensure_ascii=False), row["id"]),
                    )
            self._commit()

        async with self._lock:
            await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    async def upsert_edge(self, edge: Edge) -> Edge:
        def _do() -> Edge:
            self._execute(
                """
                INSERT INTO kg_edges
                    (id, kb_id, src_id, dst_id, relation, weight, doc_id, chunk_id, context, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    relation   = excluded.relation,
                    weight     = excluded.weight,
                    doc_id     = excluded.doc_id,
                    chunk_id   = excluded.chunk_id,
                    context    = excluded.context
                """,
                (
                    edge.id,
                    edge.kb_id,
                    edge.src_id,
                    edge.dst_id,
                    edge.relation,
                    edge.weight,
                    edge.doc_id,
                    edge.chunk_id,
                    edge.context,
                    edge.created_at or time.time(),
                ),
            )
            # update degrees for both endpoints
            self._recalc_degree(edge.src_id)
            self._recalc_degree(edge.dst_id)
            self._commit()
            return edge

        async with self._lock:
            return await asyncio.to_thread(_do)

    async def delete_edge(self, edge_id: str) -> None:
        def _do() -> None:
            row = self._execute(
                "SELECT src_id, dst_id FROM kg_edges WHERE id = ?", (edge_id,)
            ).fetchone()
            self._execute("DELETE FROM kg_edges WHERE id = ?", (edge_id,))
            if row:
                self._recalc_degree(row["src_id"])
                self._recalc_degree(row["dst_id"])
            self._commit()

        async with self._lock:
            await asyncio.to_thread(_do)

    async def list_edges(
        self,
        kb_id: str,
        src_id: str | None = None,
        dst_id: str | None = None,
    ) -> list[Edge]:
        def _do() -> list[Edge]:
            if src_id and dst_id:
                rows = self._execute(
                    "SELECT * FROM kg_edges WHERE kb_id = ? AND src_id = ? AND dst_id = ?",
                    (kb_id, src_id, dst_id),
                ).fetchall()
            elif src_id:
                rows = self._execute(
                    "SELECT * FROM kg_edges WHERE kb_id = ? AND src_id = ?",
                    (kb_id, src_id),
                ).fetchall()
            elif dst_id:
                rows = self._execute(
                    "SELECT * FROM kg_edges WHERE kb_id = ? AND dst_id = ?",
                    (kb_id, dst_id),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT * FROM kg_edges WHERE kb_id = ?", (kb_id,)
                ).fetchall()
            return [_row_to_edge(r) for r in rows]

        return await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Neighbor / path queries
    # ------------------------------------------------------------------

    async def get_neighbors(
        self, node_id: str, kb_id: str, hops: int = 1
    ) -> tuple[list[Node], list[Edge]]:
        def _do() -> tuple[list[Node], list[Edge]]:
            visited_nodes: dict[str, Node] = {}
            visited_edges: dict[str, Edge] = {}
            frontier = {node_id}

            # fetch seed node
            row = self._execute(
                "SELECT * FROM kg_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row:
                visited_nodes[node_id] = _row_to_node(row)

            for _ in range(hops):
                if not frontier:
                    break
                next_frontier: set[str] = set()
                placeholders = ",".join("?" * len(frontier))
                edge_rows = self._execute(
                    f"SELECT * FROM kg_edges WHERE kb_id = ? AND (src_id IN ({placeholders}) OR dst_id IN ({placeholders}))",
                    (kb_id, *frontier, *frontier),
                ).fetchall()
                new_node_ids: set[str] = set()
                for er in edge_rows:
                    edge = _row_to_edge(er)
                    visited_edges[edge.id] = edge
                    if edge.src_id not in visited_nodes:
                        new_node_ids.add(edge.src_id)
                        next_frontier.add(edge.src_id)
                    if edge.dst_id not in visited_nodes:
                        new_node_ids.add(edge.dst_id)
                        next_frontier.add(edge.dst_id)
                if new_node_ids:
                    ph2 = ",".join("?" * len(new_node_ids))
                    node_rows = self._execute(
                        f"SELECT * FROM kg_nodes WHERE id IN ({ph2})",
                        tuple(new_node_ids),
                    ).fetchall()
                    for nr in node_rows:
                        n = _row_to_node(nr)
                        visited_nodes[n.id] = n
                frontier = next_frontier

            # exclude seed node from result nodes
            result_nodes = [n for nid, n in visited_nodes.items() if nid != node_id]
            return result_nodes, list(visited_edges.values())

        return await asyncio.to_thread(_do)

    async def get_path(
        self, src_id: str, dst_id: str, kb_id: str, max_hops: int = 4
    ) -> list[Edge]:
        """BFS shortest path between two nodes, returns edges on path."""
        def _do() -> list[Edge]:
            if src_id == dst_id:
                return []

            # BFS
            queue: deque[tuple[str, list[str]]] = deque()
            queue.append((src_id, []))
            visited: set[str] = {src_id}

            while queue:
                current_id, path_edge_ids = queue.popleft()
                if len(path_edge_ids) >= max_hops:
                    continue

                edge_rows = self._execute(
                    "SELECT * FROM kg_edges WHERE kb_id = ? AND (src_id = ? OR dst_id = ?)",
                    (kb_id, current_id, current_id),
                ).fetchall()

                for er in edge_rows:
                    edge = _row_to_edge(er)
                    next_id = edge.dst_id if edge.src_id == current_id else edge.src_id
                    if next_id in visited:
                        continue
                    new_path = path_edge_ids + [edge.id]
                    if next_id == dst_id:
                        # reconstruct edges in order
                        edge_map: dict[str, Edge] = {edge.id: edge}
                        # fetch any other edges we haven't fully fetched
                        if len(new_path) > 1:
                            ph = ",".join("?" * len(new_path))
                            more_rows = self._execute(
                                f"SELECT * FROM kg_edges WHERE id IN ({ph})",
                                tuple(new_path),
                            ).fetchall()
                            for mr in more_rows:
                                e2 = _row_to_edge(mr)
                                edge_map[e2.id] = e2
                        return [edge_map[eid] for eid in new_path if eid in edge_map]
                    visited.add(next_id)
                    queue.append((next_id, new_path))

            return []

        return await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_nodes_by_text(
        self, query: str, kb_id: str, limit: int = 20
    ) -> list[Node]:
        def _do() -> list[Node]:
            like = f"%{query}%"
            rows = self._execute(
                """
                SELECT * FROM kg_nodes
                WHERE kb_id = ? AND (name LIKE ? OR description LIKE ?)
                LIMIT ?
                """,
                (kb_id, like, like, limit),
            ).fetchall()
            return [_row_to_node(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def search_nodes_by_embedding(
        self, embedding: list[float], kb_id: str, limit: int = 20
    ) -> list[Node]:
        def _do() -> list[Node]:
            rows = self._execute(
                "SELECT * FROM kg_nodes WHERE kb_id = ? AND embedding IS NOT NULL",
                (kb_id,),
            ).fetchall()
            scored: list[tuple[float, Node]] = []
            for row in rows:
                node = _row_to_node(row)
                if node.embedding:
                    sim = _cosine_similarity(embedding, node.embedding)
                    scored.append((sim, node))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [n for _, n in scored[:limit]]

        return await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Community CRUD
    # ------------------------------------------------------------------

    async def upsert_community(self, community: Community) -> Community:
        def _do() -> Community:
            self._execute(
                """
                INSERT INTO kg_communities
                    (id, kb_id, node_ids, summary, level, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_ids  = excluded.node_ids,
                    summary   = excluded.summary,
                    level     = excluded.level,
                    embedding = excluded.embedding
                """,
                (
                    community.id,
                    community.kb_id,
                    json.dumps(community.node_ids, ensure_ascii=False),
                    community.summary,
                    community.level,
                    _serialize_embedding(community.embedding),
                    community.created_at or time.time(),
                ),
            )
            self._commit()
            return community

        async with self._lock:
            return await asyncio.to_thread(_do)

    async def list_communities(
        self, kb_id: str, level: int | None = None
    ) -> list[Community]:
        def _do() -> list[Community]:
            if level is not None:
                rows = self._execute(
                    "SELECT * FROM kg_communities WHERE kb_id = ? AND level = ?",
                    (kb_id, level),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT * FROM kg_communities WHERE kb_id = ?", (kb_id,)
                ).fetchall()
            return [_row_to_community(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def search_communities_by_embedding(
        self,
        embedding: list[float],
        kb_id: str,
        level: int | None = None,
        limit: int = 10,
    ) -> list[Community]:
        def _do() -> list[Community]:
            if level is not None:
                rows = self._execute(
                    "SELECT * FROM kg_communities WHERE kb_id = ? AND level >= ? AND embedding IS NOT NULL",
                    (kb_id, level),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT * FROM kg_communities WHERE kb_id = ? AND embedding IS NOT NULL",
                    (kb_id,),
                ).fetchall()
            scored: list[tuple[float, Community]] = []
            for row in rows:
                comm = _row_to_community(row)
                if comm.embedding:
                    sim = _cosine_similarity(embedding, comm.embedding)
                    scored.append((sim, comm))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]

        return await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Stats & full graph
    # ------------------------------------------------------------------

    async def get_stats(self, kb_id: str) -> dict:
        def _do() -> dict:
            node_row = self._execute(
                "SELECT COUNT(*) AS cnt FROM kg_nodes WHERE kb_id = ?", (kb_id,)
            ).fetchone()
            edge_row = self._execute(
                "SELECT COUNT(*) AS cnt FROM kg_edges WHERE kb_id = ?", (kb_id,)
            ).fetchone()
            comm_row = self._execute(
                "SELECT COUNT(*) AS cnt FROM kg_communities WHERE kb_id = ?", (kb_id,)
            ).fetchone()
            # unique doc_ids across nodes
            doc_rows = self._execute(
                "SELECT doc_ids FROM kg_nodes WHERE kb_id = ?", (kb_id,)
            ).fetchall()
            all_docs: set[str] = set()
            for dr in doc_rows:
                ids: list[str] = json.loads(dr["doc_ids"] or "[]")
                all_docs.update(ids)
            return {
                "nodes": node_row["cnt"] if node_row else 0,
                "edges": edge_row["cnt"] if edge_row else 0,
                "communities": comm_row["cnt"] if comm_row else 0,
                "docs": len(all_docs),
            }

        return await asyncio.to_thread(_do)

    async def get_full_graph(
        self, kb_id: str, limit: int = 500
    ) -> tuple[list[Node], list[Edge]]:
        def _do() -> tuple[list[Node], list[Edge]]:
            node_rows = self._execute(
                "SELECT * FROM kg_nodes WHERE kb_id = ? LIMIT ?", (kb_id, limit)
            ).fetchall()
            nodes = [_row_to_node(r) for r in node_rows]
            node_ids = [n.id for n in nodes]
            if not node_ids:
                return nodes, []
            ph = ",".join("?" * len(node_ids))
            edge_rows = self._execute(
                f"SELECT * FROM kg_edges WHERE kb_id = ? AND src_id IN ({ph}) AND dst_id IN ({ph})",
                (kb_id, *node_ids, *node_ids),
            ).fetchall()
            edges = [_row_to_edge(r) for r in edge_rows]
            return nodes, edges

        return await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# Neo4jKGStore stub
# ---------------------------------------------------------------------------

class Neo4jKGStore(KGStore):
    """Neo4j 存储实现（需要安装 neo4j 驱动）。"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import AsyncGraphDatabase  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Neo4j driver not installed. Install it with: pip install neo4j"
            ) from exc
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None

    async def initialize(self) -> None:
        raise NotImplementedError("Neo4jKGStore is a stub. Implement as needed.")

    async def upsert_node(self, node: "Node") -> "Node":
        raise NotImplementedError

    async def upsert_edge(self, edge: "Edge") -> "Edge":
        raise NotImplementedError

    async def get_node(self, node_id: str) -> "Node | None":
        raise NotImplementedError

    async def get_node_by_name(self, name: str, kb_id: str) -> "Node | None":
        raise NotImplementedError

    async def list_nodes(self, kb_id: str, node_type=None, limit: int = 100, offset: int = 0) -> list:
        raise NotImplementedError

    async def list_edges(self, kb_id: str, src_id=None, dst_id=None) -> list:
        raise NotImplementedError

    async def get_neighbors(self, node_id: str, kb_id: str, hops: int = 1) -> tuple:
        raise NotImplementedError

    async def search_nodes_by_text(self, query: str, kb_id: str, limit: int = 20) -> list:
        raise NotImplementedError

    async def search_nodes_by_embedding(self, embedding: list, kb_id: str, limit: int = 20) -> list:
        raise NotImplementedError

    async def delete_node(self, node_id: str) -> None:
        raise NotImplementedError

    async def delete_edge(self, edge_id: str) -> None:
        raise NotImplementedError

    async def delete_by_doc(self, doc_id: str, kb_id: str) -> None:
        raise NotImplementedError

    async def upsert_community(self, community: "Community") -> "Community":
        raise NotImplementedError

    async def list_communities(self, kb_id: str, level=None) -> list:
        raise NotImplementedError

    async def search_communities_by_embedding(self, embedding: list, kb_id: str, level=None, limit: int = 10) -> list:
        raise NotImplementedError

    async def get_stats(self, kb_id: str) -> dict:
        raise NotImplementedError

    async def get_full_graph(self, kb_id: str, limit: int = 500) -> tuple:
        raise NotImplementedError

    async def get_path(self, src_id: str, dst_id: str, kb_id: str, max_hops: int = 4) -> list:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_kg_store(url: str) -> KGStore:
    """
    Factory function to create a KGStore from a URL string.

    Supported schemes:
    - "sqlite:///path/to/file.db"  -> SQLiteKGStore
    - "neo4j://user:password@host/database" -> Neo4jKGStore

    Examples:
        store = create_kg_store("sqlite:///data/workspace.db")
        store = create_kg_store("neo4j://neo4j:secret@localhost:7687/neo4j")
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme == "sqlite":
        # Handle sqlite:///absolute/path or sqlite://relative/path
        if url.startswith("sqlite:///"):
            db_path = url[len("sqlite:///"):]
        elif url.startswith("sqlite://"):
            db_path = url[len("sqlite://"):]
        else:
            db_path = parsed.path
        return SQLiteKGStore(db_path=db_path)

    elif scheme in ("neo4j", "neo4j+s", "bolt", "bolt+s"):
        user = parsed.username or "neo4j"
        password = parsed.password or ""
        host = parsed.hostname or "localhost"
        port = parsed.port or 7687
        database = parsed.path.lstrip("/") or "neo4j"
        uri = f"{scheme}://{host}:{port}"
        return Neo4jKGStore(uri=uri, user=user, password=password, database=database)

    else:
        raise ValueError(
            f"Unsupported KG store URL scheme: '{scheme}'. "
            "Use 'sqlite:///path/to/db' or 'neo4j://user:pass@host/db'."
        )
