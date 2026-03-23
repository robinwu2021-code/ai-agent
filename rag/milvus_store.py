"""
rag/milvus_store.py — Milvus 向量数据库存储层

Milvus 相比 Qdrant 的核心升级：
  ┌─────────────────────────────────────────────────────────────────┐
  │  特性                    Qdrant             Milvus              │
  │  BM25 索引               Python 内存重建     原生稀疏向量 Function │
  │  混合检索 RRF            Python 端融合       内置 RRFRanker       │
  │  中文分词                bigram 降级         内置 Chinese Analyzer│
  │  索引类型                HNSW only           HNSW/DISKANN/IVF_PQ │
  │  超大规模（>1亿向量）     受限               Distributed 原生支持  │
  │  启动重建 BM25           需要全量 scroll     无（索引已持久化）     │
  └─────────────────────────────────────────────────────────────────┘

部署模式：
  Milvus Lite (嵌入式，推荐开发)：   MILVUS_URI=./data/milvus.db
  Milvus Standalone (生产单机)：     MILVUS_URI=http://localhost:19530
  Milvus Distributed (生产集群)：    MILVUS_URI=http://milvus-proxy:19530

依赖：
    pip install "pymilvus>=2.4.0"
    # Milvus Lite (无需额外服务):
    pip install "pymilvus[model]>=2.4.0"

Collection Schema：
  chunk_id    VARCHAR(128)  PRIMARY KEY
  doc_id      VARCHAR(128)  [scalar index]
  kb_id       VARCHAR(64)   [scalar index]
  chunk_index INT32
  text        VARCHAR(65535) [enable_analyzer=True → BM25 → sparse_vec]
  dense_vec   FLOAT_VECTOR(dim)     [HNSW index]
  sparse_vec  SPARSE_FLOAT_VECTOR   [SPARSE_INVERTED_INDEX, auto by BM25]
  created_at  DOUBLE
  meta_json   VARCHAR(8192)
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog

from rag.vector_store_base import VectorStoreBase

log = structlog.get_logger(__name__)

_COLLECTION = "kb_chunks"

# ---------------------------------------------------------------------------
# MilvusVectorStore
# ---------------------------------------------------------------------------

class MilvusVectorStore(VectorStoreBase):
    """
    Milvus 向量数据库后端。

    构造参数：
      uri         Milvus Lite: "./data/milvus.db"
                  Standalone:  "http://localhost:19530"
      token       鉴权 token，格式 "user:password" 或 API key（可选）
      collection  collection 名称，默认 "kb_chunks"
      vector_size embedding 维度，需与 embed_fn 输出一致
      index_type  向量索引类型：HNSW（默认）| DISKANN（超大规模）
      m           HNSW M 参数（越大精度越高，内存越多）
      ef_construction  HNSW 建索引 ef 参数
    """

    def __init__(
        self,
        uri:              str  = "./data/milvus.db",
        token:            str  = "",
        collection:       str  = _COLLECTION,
        vector_size:      int  = 1536,
        index_type:       str  = "HNSW",
        m:                int  = 16,
        ef_construction:  int  = 200,
    ) -> None:
        self._uri             = uri
        self._token           = token
        self._collection      = collection
        self._vector_size     = vector_size
        self._index_type      = index_type
        self._m               = m
        self._ef_construction = ef_construction
        self._client: Any     = None
        self._initialized     = False
        self._lock            = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # initialize                                                           #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._sync_initialize)
            self._initialized = True
            mode = "lite" if self._uri.endswith(".db") else "standalone"
            log.info("milvus_store.initialized",
                     collection=self._collection,
                     mode=mode,
                     uri=self._uri)

    def _sync_initialize(self) -> None:
        try:
            from pymilvus import MilvusClient  # type: ignore
        except ImportError:
            raise RuntimeError(
                "请安装 pymilvus：pip install 'pymilvus>=2.4.0'"
            )
        kw: dict = {"uri": self._uri}
        if self._token:
            kw["token"] = self._token
        self._client = MilvusClient(**kw)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """创建 collection（若不存在）+ 所有索引（幂等）。"""
        from pymilvus import (          # type: ignore
            DataType, Function, FunctionType,
        )

        if self._client.has_collection(self._collection):
            log.debug("milvus_store.collection_exists",
                      collection=self._collection)
            return

        # ── Schema ────────────────────────────────────────────────────
        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        # Primary key
        schema.add_field("chunk_id", DataType.VARCHAR,
                         max_length=128, is_primary=True)

        # Scalar fields（带 scalar index，过滤时 O(1)）
        schema.add_field("doc_id",      DataType.VARCHAR, max_length=128)
        schema.add_field("kb_id",       DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_index", DataType.INT32)
        schema.add_field("created_at",  DataType.DOUBLE)
        schema.add_field("meta_json",   DataType.VARCHAR, max_length=8192)

        # Text field — analyzer 开启后 BM25 Function 可自动生成稀疏向量
        # 尝试 Chinese analyzer（Milvus 2.5+），降级到 standard
        try:
            schema.add_field(
                "text", DataType.VARCHAR, max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},   # 内置结巴分词
            )
            log.info("milvus_store.analyzer", type="chinese")
        except Exception:
            schema.add_field(
                "text", DataType.VARCHAR, max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": "standard"},  # Unicode 分词降级
            )
            log.info("milvus_store.analyzer", type="standard_fallback")

        # Dense vector field
        schema.add_field("dense_vec", DataType.FLOAT_VECTOR,
                         dim=self._vector_size)

        # Sparse vector field — populated automatically by BM25 Function
        schema.add_field("sparse_vec", DataType.SPARSE_FLOAT_VECTOR)

        # ── BM25 Function：text → sparse_vec（全自动，无需外部依赖）──────
        bm25_fn = Function(
            name="bm25_fn",
            input_field_names=["text"],
            output_field_names=["sparse_vec"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_fn)

        # ── Index params ──────────────────────────────────────────────
        index_params = self._client.prepare_index_params()

        # Dense vector: HNSW or DISKANN
        if self._index_type == "DISKANN":
            index_params.add_index(
                field_name="dense_vec",
                index_type="DISKANN",
                metric_type="COSINE",
            )
        else:
            index_params.add_index(
                field_name="dense_vec",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": self._m, "efConstruction": self._ef_construction},
            )

        # Sparse vector: SPARSE_INVERTED_INDEX + BM25 metric
        index_params.add_index(
            field_name="sparse_vec",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        # Scalar indexes for fast kb_id / doc_id filtering
        for field in ("kb_id", "doc_id"):
            index_params.add_index(
                field_name=field,
                index_type="INVERTED",
            )

        # ── Create collection ─────────────────────────────────────────
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        log.info("milvus_store.collection_created",
                 collection=self._collection,
                 index=self._index_type,
                 dim=self._vector_size)

    # ------------------------------------------------------------------ #
    # upsert_chunks                                                        #
    # ------------------------------------------------------------------ #

    async def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        批量 upsert 分块到 Milvus。
        注意：sparse_vec 字段不需要提供，由 BM25 Function 从 text 自动生成。
        """
        if not chunks:
            return
        rows = []
        for c in chunks:
            embedding = c.get("embedding")
            if not embedding:
                log.warning("milvus_store.upsert_skip_no_embedding",
                            chunk_id=c.get("chunk_id"))
                continue
            meta = c.get("meta") or {}
            rows.append({
                "chunk_id":    c["chunk_id"],
                "doc_id":      c["doc_id"],
                "kb_id":       c["kb_id"],
                "chunk_index": int(c.get("chunk_index", 0)),
                "text":        c.get("text", ""),
                "dense_vec":   list(embedding),
                # sparse_vec is NOT provided — Milvus generates it via BM25 Function
                "created_at":  float(c.get("created_at", time.time())),
                "meta_json":   json.dumps(meta, ensure_ascii=False)[:8192],
            })

        if not rows:
            return

        # Batch upsert (100 rows per call)
        batch = 100
        for i in range(0, len(rows), batch):
            await asyncio.to_thread(
                self._client.upsert,
                collection_name=self._collection,
                data=rows[i:i + batch],
            )
        log.debug("milvus_store.upserted", count=len(rows))

    # ------------------------------------------------------------------ #
    # hybrid_search  (Milvus 原生：dense + sparse BM25 → RRFRanker)       #
    # ------------------------------------------------------------------ #

    async def hybrid_search(
        self,
        query_vec:  list[float],
        query_text: str,
        kb_id:      str,
        top_k:      int = 5,
        rrf_k:      int = 60,
        doc_ids:    list[str] | None = None,
    ) -> list[tuple[float, dict]]:
        """
        原生混合检索：
          • Dense ANN  — query_vec  在 dense_vec 上做 HNSW/COSINE 检索
          • Sparse BM25 — query_text 经 BM25 Function 转稀疏向量后检索
          • Milvus 内置 RRFRanker 融合，无需 Python 端 RRF

        doc_ids: 非空时追加 doc_id in [...] 过滤，实现单/多文档范围检索
        """
        from pymilvus import AnnSearchRequest, RRFRanker  # type: ignore

        expr = f'kb_id == "{kb_id}"'
        if doc_ids:
            ids_lit = "[" + ", ".join(f'"{d}"' for d in doc_ids) + "]"
            expr    = f'({expr}) and doc_id in {ids_lit}'
        candidates = top_k * 3
        out_fields = ["chunk_id", "doc_id", "kb_id",
                      "chunk_index", "text", "meta_json", "created_at"]

        # Dense search request
        dense_req = AnnSearchRequest(
            data=[list(query_vec)],
            anns_field="dense_vec",
            param={
                "metric_type": "COSINE",
                "params": {"ef": max(candidates, 64)},
            },
            limit=candidates,
            expr=expr,
        )

        # Sparse BM25 search request
        # data 传入文本字符串，Milvus 自动通过 BM25 Function 生成稀疏向量
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vec",
            param={"metric_type": "BM25"},
            limit=candidates,
            expr=expr,
        )

        results = await asyncio.to_thread(
            self._client.hybrid_search,
            collection_name=self._collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=out_fields,
        )

        hits = []
        # hybrid_search returns list[list[Hit]]
        for hit in (results[0] if results else []):
            payload = self._hit_to_payload(hit)
            hits.append((float(hit.score), payload))
        return hits

    # ------------------------------------------------------------------ #
    # vector_search  (pure dense, no BM25)                                #
    # ------------------------------------------------------------------ #

    async def vector_search(
        self,
        query_vec: list[float],
        kb_id:     str,
        top_k:     int = 10,
    ) -> list[tuple[float, dict]]:
        out_fields = ["chunk_id", "doc_id", "kb_id",
                      "chunk_index", "text", "meta_json", "created_at"]
        results = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection,
            data=[list(query_vec)],
            anns_field="dense_vec",
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": max(top_k * 2, 64)},
            },
            limit=top_k,
            filter=f'kb_id == "{kb_id}"',
            output_fields=out_fields,
        )
        hits = []
        for hit in (results[0] if results else []):
            hits.append((float(hit["distance"]), self._hit_to_payload(hit)))
        return hits

    # ------------------------------------------------------------------ #
    # delete                                                               #
    # ------------------------------------------------------------------ #

    async def delete_by_doc_id(self, doc_id: str) -> int:
        ids = await self._get_chunk_ids(f'doc_id == "{doc_id}"')
        if ids:
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection,
                ids=ids,
            )
        log.info("milvus_store.deleted_doc", doc_id=doc_id, count=len(ids))
        return len(ids)

    async def delete_by_kb_id(self, kb_id: str) -> int:
        ids = await self._get_chunk_ids(f'kb_id == "{kb_id}"')
        if ids:
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection,
                ids=ids,
            )
        log.info("milvus_store.deleted_kb", kb_id=kb_id, count=len(ids))
        return len(ids)

    async def _get_chunk_ids(self, filter_expr: str) -> list[str]:
        """Query all chunk_id (primary key) matching filter_expr."""
        results = await asyncio.to_thread(
            self._client.query,
            collection_name=self._collection,
            filter=filter_expr,
            output_fields=["chunk_id"],
            limit=16384,
        )
        return [r["chunk_id"] for r in results]

    # ------------------------------------------------------------------ #
    # list / stats                                                         #
    # ------------------------------------------------------------------ #

    async def list_chunks(self, doc_id: str) -> list[dict]:
        rows = await asyncio.to_thread(
            self._client.query,
            collection_name=self._collection,
            filter=f'doc_id == "{doc_id}"',
            output_fields=["chunk_id", "doc_id", "kb_id",
                           "chunk_index", "text", "meta_json", "created_at"],
            limit=10000,
        )
        payloads = [self._row_to_payload(r) for r in rows]
        return sorted(payloads, key=lambda x: x.get("chunk_index", 0))

    async def list_chunks_by_kb(
        self, kb_id: str, limit: int = 2000
    ) -> list[dict]:
        rows = await asyncio.to_thread(
            self._client.query,
            collection_name=self._collection,
            filter=f'kb_id == "{kb_id}"',
            output_fields=["chunk_id", "doc_id", "kb_id",
                           "chunk_index", "text", "meta_json", "created_at"],
            limit=limit,
        )
        return [self._row_to_payload(r) for r in rows]

    async def get_stats(self, kb_id: str) -> dict:
        # Milvus limits (offset+limit) to 16384; use pagination to aggregate stats
        _PAGE   = 1000
        offset  = 0
        all_doc_ids:   set[str] = set()
        total_chars = 0
        chunk_count = 0

        def _query_page(off: int) -> list:
            return self._client.query(
                collection_name=self._collection,
                filter=f'kb_id == "{kb_id}"',
                output_fields=["doc_id", "text"],
                limit=_PAGE,
                offset=off,
            )

        while True:
            rows = await asyncio.to_thread(_query_page, offset)
            if not rows:
                break
            chunk_count += len(rows)
            all_doc_ids.update(r["doc_id"] for r in rows)
            total_chars += sum(len(r.get("text", "")) for r in rows)
            if len(rows) < _PAGE:
                break
            offset += _PAGE

        return {
            "chunk_count": chunk_count,
            "doc_count":   len(all_doc_ids),
            "total_chars": total_chars,
        }

    async def collection_info(self) -> dict:
        info = await asyncio.to_thread(
            self._client.describe_collection,
            collection_name=self._collection,
        )
        stats = await asyncio.to_thread(
            self._client.get_collection_stats,
            collection_name=self._collection,
        )
        return {
            "collection":   self._collection,
            "row_count":    stats.get("row_count", 0),
            "index_type":   self._index_type,
            "vector_size":  self._vector_size,
            "bm25_enabled": True,
            "uri":          self._uri,
        }

    async def close(self) -> None:
        if self._client:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hit_to_payload(hit: Any) -> dict:
        """Convert a Milvus search Hit to a plain dict."""
        # Hit may be dict-like or have entity attribute
        entity = hit.get("entity", hit) if hasattr(hit, "get") else hit.entity
        return MilvusVectorStore._parse_entity(entity)

    @staticmethod
    def _row_to_payload(row: dict) -> dict:
        return MilvusVectorStore._parse_entity(row)

    @staticmethod
    def _parse_entity(entity: Any) -> dict:
        def _get(key: str, default: Any = None) -> Any:
            if hasattr(entity, key):
                return getattr(entity, key)
            if hasattr(entity, "get"):
                return entity.get(key, default)
            return default

        meta_raw = _get("meta_json", "{}")
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except (ValueError, TypeError):
            meta = {}

        return {
            "chunk_id":    _get("chunk_id", ""),
            "doc_id":      _get("doc_id", ""),
            "kb_id":       _get("kb_id", ""),
            "chunk_index": int(_get("chunk_index", 0)),
            "text":        _get("text", ""),
            "meta":        meta,
            "created_at":  float(_get("created_at", 0.0)),
        }
