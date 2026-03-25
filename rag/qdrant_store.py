"""
rag/qdrant_store.py — Qdrant 向量数据库存储层

功能：
  - 嵌入式模式（embedded，无需启动服务，数据持久化到本地目录）
  - 服务器模式（server，生产环境只需改 url 参数）
  - HNSW 近似最近邻向量检索 O(log n)
  - payload 索引支持 kb_id / doc_id 快速过滤
  - 混合检索：Qdrant dense 向量 + 内存 BM25 → RRF 融合
  - 启动时从 Qdrant payload 重建 BM25，无需重新 embed

依赖：
    pip install qdrant-client
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

import structlog

from rag.vector_store_base import VectorStoreBase

log = structlog.get_logger(__name__)

# Qdrant collection used for all KB chunks (kb_id stored as payload field)
_COLLECTION = "kb_chunks"


def _chunk_id_to_point_id(chunk_id: str) -> int:
    """Convert a string chunk_id to a deterministic 63-bit integer for Qdrant."""
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# QdrantVectorStore
# ---------------------------------------------------------------------------

class QdrantVectorStore(VectorStoreBase):
    """
    向量数据库存储 — Qdrant 实现。

    构造参数：
      path      嵌入式模式，数据目录（与 url 二选一）
      url       服务器模式，如 "http://localhost:6333"
      api_key   服务器模式鉴权 key（可选）
      collection  Qdrant collection 名称，默认 "kb_chunks"
      vector_size embedding 维度，需与 embed_fn 输出一致

    典型用法：
      # 嵌入式（开发 / 单机）
      store = QdrantVectorStore(path="./data/qdrant")

      # 生产服务器
      store = QdrantVectorStore(url="http://qdrant:6333", api_key="secret")
    """

    def __init__(
        self,
        path:        str | None = None,
        url:         str | None = None,
        api_key:     str | None = None,
        collection:  str        = _COLLECTION,
        vector_size: int        = 1536,
    ) -> None:
        if not path and not url:
            path = "./data/qdrant"           # default: embedded in cwd
        self._path        = path
        self._url         = url
        self._api_key     = api_key
        self._collection  = collection
        self._vector_size = vector_size
        self._client: Any = None            # AsyncQdrantClient
        self._bm25:   Any = None            # BM25Index (in-memory)
        self._chunk_map: dict[str, Any] = {}  # chunk_id → Chunk (for BM25)
        self._lock = asyncio.Lock()
        self._initialized = False

    # ------------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            await self._connect()
            await self._ensure_collection()
            await self._rebuild_bm25()
            self._initialized = True
            log.info("qdrant_store.initialized",
                     collection=self._collection,
                     mode="server" if self._url else "embedded",
                     path=self._path)

    async def _connect(self) -> None:
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore
        except ImportError:
            raise RuntimeError(
                "请安装 qdrant-client：pip install qdrant-client"
            )
        if self._url:
            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
            )
        else:
            import os
            os.makedirs(self._path, exist_ok=True)
            self._client = AsyncQdrantClient(path=self._path)

    async def _ensure_collection(self) -> None:
        from qdrant_client.models import (   # type: ignore
            Distance, VectorParams, PayloadSchemaType,
        )
        existing = {
            c.name
            for c in (await self._client.get_collections()).collections
        }
        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                    on_disk=True,        # vectors stored on disk, not RAM
                ),
            )
            log.info("qdrant_store.collection_created",
                     collection=self._collection,
                     vector_size=self._vector_size)

        # Create payload indexes for fast filtering (idempotent)
        for field, schema in [
            ("kb_id",  PayloadSchemaType.KEYWORD),
            ("doc_id", PayloadSchemaType.KEYWORD),
        ]:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass  # Already exists — safe to ignore

    # ------------------------------------------------------------------
    # BM25 (in-memory, rebuilt from Qdrant payloads on startup)
    # ------------------------------------------------------------------

    async def _rebuild_bm25(self) -> None:
        """
        Scroll all payloads from Qdrant and rebuild the in-memory BM25 index.
        Called once on initialize(). No re-embedding needed.
        """
        from rag.knowledge_base import BM25Index, Chunk  # type: ignore
        payloads = await self._scroll_payloads()
        chunks = [
            Chunk(
                id          = p["chunk_id"],
                doc_id      = p["doc_id"],
                text        = p.get("text", ""),
                chunk_index = p.get("chunk_index", 0),
                metadata    = p.get("meta", {}),
            )
            for p in payloads
        ]
        self._bm25 = BM25Index()
        self._bm25.build(chunks)
        self._chunk_map = {c.id: c for c in chunks}
        log.info("qdrant_store.bm25_rebuilt", total=len(chunks))

    async def _scroll_payloads(
        self,
        kb_id:  str | None = None,
        doc_id: str | None = None,
    ) -> list[dict]:
        """Scroll all matching points from Qdrant (paginated, no vectors)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        must = []
        if kb_id:
            must.append(FieldCondition(key="kb_id", match=MatchValue(value=kb_id)))
        if doc_id:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        filter_ = Filter(must=must) if must else None

        results: list[dict] = []
        offset = None
        while True:
            response, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=filter_,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(p.payload for p in response if p.payload)
            if next_offset is None:
                break
            offset = next_offset
        return results

    # ------------------------------------------------------------------
    # upsert_chunks
    # ------------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        Upsert chunks into Qdrant.

        Each dict must have:
          chunk_id, doc_id, kb_id, text, embedding (list[float])
        Optional:
          chunk_index, meta, created_at
        """
        from qdrant_client.models import PointStruct  # type: ignore
        from rag.knowledge_base import Chunk          # type: ignore

        points = []
        new_mem_chunks: list[Any] = []

        for c in chunks:
            embedding = c.get("embedding")
            if not embedding:
                log.warning("qdrant_store.upsert_skip_no_embedding",
                            chunk_id=c.get("chunk_id"))
                continue

            point_id = _chunk_id_to_point_id(c["chunk_id"])
            points.append(PointStruct(
                id     = point_id,
                vector = list(embedding),
                payload = {
                    "chunk_id":    c["chunk_id"],
                    "doc_id":      c["doc_id"],
                    "kb_id":       c["kb_id"],
                    "chunk_index": c.get("chunk_index", 0),
                    "text":        c.get("text", ""),
                    "meta":        c.get("meta", {}),
                    "created_at":  c.get("created_at", time.time()),
                },
            ))
            new_mem_chunks.append(Chunk(
                id          = c["chunk_id"],
                doc_id      = c["doc_id"],
                text        = c.get("text", ""),
                chunk_index = c.get("chunk_index", 0),
                metadata    = c.get("meta", {}),
            ))

        if not points:
            return

        # Batch upsert to Qdrant (max 100 per batch to avoid timeouts)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            await self._client.upsert(
                collection_name=self._collection,
                points=points[i:i + batch_size],
                wait=True,
            )

        # Update in-memory BM25
        for c in new_mem_chunks:
            self._chunk_map[c.id] = c
        if self._bm25 is not None:
            self._bm25.build(list(self._chunk_map.values()))

        log.debug("qdrant_store.upserted", count=len(points))

    # ------------------------------------------------------------------
    # search (vector only)
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query_vec: list[float],
        kb_id:     str,
        top_k:     int = 10,
    ) -> list[tuple[float, dict]]:
        """
        Pure vector search with kb_id filter.
        Returns list of (cosine_score, payload) sorted by descending score.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        filter_ = Filter(
            must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
        )
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=list(query_vec),
            query_filter=filter_,
            limit=top_k,
            with_payload=True,
        )
        return [(r.score, r.payload) for r in results if r.payload]

    # ------------------------------------------------------------------
    # hybrid_search (vector + BM25 → RRF)
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        query_vec:  list[float],
        query_text: str,
        kb_id:      str,
        top_k:      int   = 5,
        rrf_k:      int   = 60,
        doc_ids:    list[str] | None = None,
    ) -> list[tuple[float, dict]]:
        """
        Hybrid retrieval:
          1. Qdrant HNSW vector search (top_k × 3 candidates)
          2. In-memory BM25 keyword search (top_k × 3 candidates)
          3. RRF fusion → return top_k results

        doc_ids: 非空时只检索指定文档范围。
        Score is the RRF fusion score (higher = more relevant).
        """
        doc_set    = set(doc_ids) if doc_ids else None
        candidates = top_k * 3

        # 1. Vector search (Qdrant HNSW)
        vec_hits_all = await self.vector_search(query_vec, kb_id, top_k=candidates)
        vec_hits = [(s, p) for s, p in vec_hits_all
                    if doc_set is None or p.get("doc_id") in doc_set]

        # 2. BM25 keyword search (filtered by kb_id)
        bm25_hits: list[tuple[float, dict]] = []
        if self._bm25:
            raw_hits = self._bm25.search(query_text, top_k=candidates)
            for score, chunk in raw_hits:
                if chunk.metadata.get("kb_id", kb_id) != kb_id:
                    continue
                if doc_set is not None and chunk.doc_id not in doc_set:
                    continue
                payload = {
                    "chunk_id":    chunk.id,
                    "doc_id":      chunk.doc_id,
                    "kb_id":       kb_id,
                    "chunk_index": chunk.chunk_index,
                    "text":        chunk.text,
                    "meta":        chunk.metadata,
                    "created_at":  0.0,
                }
                bm25_hits.append((score, payload))

        # 3. RRF fusion
        scores:      dict[str, float] = {}
        payload_map: dict[str, dict]  = {}

        for rank, (_, payload) in enumerate(vec_hits):
            cid = payload["chunk_id"]
            scores[cid]      = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            payload_map[cid] = payload

        for rank, (_, payload) in enumerate(bm25_hits):
            cid = payload["chunk_id"]
            scores[cid]      = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            payload_map[cid] = payload

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(sc, payload_map[cid]) for cid, sc in ranked[:top_k]]

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        from qdrant_client.models import (  # type: ignore
            Filter, FieldCondition, MatchValue, FilterSelector,
        )
        # Count first
        before = len(await self._scroll_payloads(doc_id=doc_id))
        if before == 0:
            return 0

        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )

        # Remove from in-memory BM25
        to_remove = [
            cid for cid, c in self._chunk_map.items()
            if c.doc_id == doc_id
        ]
        for cid in to_remove:
            del self._chunk_map[cid]
        if self._bm25 is not None:
            self._bm25.build(list(self._chunk_map.values()))

        log.info("qdrant_store.deleted_doc", doc_id=doc_id, count=before)
        return before

    async def delete_by_kb_id(self, kb_id: str) -> int:
        """Delete all chunks for a knowledge base."""
        from qdrant_client.models import (  # type: ignore
            Filter, FieldCondition, MatchValue, FilterSelector,
        )
        before = len(await self._scroll_payloads(kb_id=kb_id))
        if before == 0:
            return 0

        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
                )
            ),
            wait=True,
        )

        to_remove = [
            cid for cid, c in self._chunk_map.items()
            if c.metadata.get("kb_id") == kb_id
        ]
        for cid in to_remove:
            del self._chunk_map[cid]
        if self._bm25 is not None:
            self._bm25.build(list(self._chunk_map.values()))

        return before

    # ------------------------------------------------------------------
    # list / stats
    # ------------------------------------------------------------------

    async def list_chunks(self, doc_id: str) -> list[dict]:
        payloads = await self._scroll_payloads(doc_id=doc_id)
        return sorted(payloads, key=lambda x: x.get("chunk_index", 0))

    async def list_chunks_by_kb(
        self, kb_id: str, limit: int = 2000
    ) -> list[dict]:
        payloads = await self._scroll_payloads(kb_id=kb_id)
        return payloads[:limit]

    async def get_stats(self, kb_id: str) -> dict:
        payloads = await self._scroll_payloads(kb_id=kb_id)
        doc_ids  = list({p["doc_id"] for p in payloads})
        total_chars = sum(len(p.get("text", "")) for p in payloads)
        return {
            "chunk_count": len(payloads),
            "doc_count":   len(doc_ids),
            "total_chars": total_chars,
        }

    async def collection_info(self) -> dict:
        """Return raw Qdrant collection info (vectors count, disk usage, etc.)."""
        info = await self._client.get_collection(self._collection)
        return {
            "vectors_count":   info.vectors_count,
            "points_count":    info.points_count,
            "segments_count":  info.segments_count,
            "status":          str(info.status),
            "config": {
                "vector_size": self._vector_size,
                "distance":    "cosine",
                "on_disk":     True,
            },
        }

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
