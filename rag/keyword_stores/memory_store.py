"""
rag/keyword_stores/memory_store.py — 内存 BM25 关键词检索（无外部依赖）

基于现有项目的 BM25Index，支持持久化（pickle）。
生产环境建议切换到 elasticsearch backend。
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import structlog

from rag.keyword_stores.base import KeywordHit

log = structlog.get_logger(__name__)


class MemoryKeywordStore:
    """内存 BM25，支持 kb_id 多租户隔离，可选 pickle 持久化。"""

    def __init__(self, persist: bool = True, path: str = "./data/bm25.pkl") -> None:
        self._persist = persist
        self._path = Path(path)
        # kb_id -> list of {chunk_id, doc_id, kb_id, text, metadata}
        self._docs: dict[str, list[dict]] = {}
        # kb_id -> BM25Index
        self._indices: dict[str, object] = {}

    async def initialize(self) -> None:
        if self._persist and self._path.exists():
            try:
                data = await asyncio.to_thread(self._load_pickle)
                self._docs = data.get("docs", {})
                await self._rebuild_all_indices()
                log.info("memory_bm25.loaded", path=str(self._path),
                         kb_count=len(self._docs))
            except Exception as exc:
                log.warning("memory_bm25.load_failed", error=str(exc))

    def _load_pickle(self) -> dict:
        with open(self._path, "rb") as f:
            return pickle.load(f)

    def _save_pickle(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"docs": self._docs}, f)

    async def _rebuild_all_indices(self) -> None:
        for kb_id in self._docs:
            await self._rebuild_index(kb_id)

    async def _rebuild_index(self, kb_id: str) -> None:
        from rag.knowledge_base import BM25Index, Chunk
        idx = BM25Index()
        chunks = [
            Chunk(
                id=d["chunk_id"], doc_id=d["doc_id"],
                text=d["text"], chunk_index=d.get("chunk_index", 0),
                metadata=d.get("metadata", {}),
            )
            for d in self._docs.get(kb_id, [])
        ]
        idx.build(chunks)
        self._indices[kb_id] = idx

    async def index_chunks(self, chunks: list[dict], kb_id: str) -> None:
        if kb_id not in self._docs:
            self._docs[kb_id] = []
        # 去重（按 chunk_id）
        existing_ids = {d["chunk_id"] for d in self._docs[kb_id]}
        new_docs = [c for c in chunks if c["chunk_id"] not in existing_ids]
        self._docs[kb_id].extend(new_docs)
        await self._rebuild_index(kb_id)
        if self._persist:
            await asyncio.to_thread(self._save_pickle)
        log.debug("memory_bm25.indexed", kb_id=kb_id, added=len(new_docs))

    async def search(self, query: str, kb_id: str,
                     top_k: int = 20, doc_ids: list[str] | None = None) -> list[KeywordHit]:
        idx = self._indices.get(kb_id)
        if idx is None:
            return []
        try:
            from rag.knowledge_base import Chunk as _Chunk
            # BM25Index.search returns list of Chunk with .score
            results = idx.search(query, top_k=top_k * 2 if doc_ids else top_k)
            hits: list[KeywordHit] = []
            for chunk in results:
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue
                hits.append(KeywordHit(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    kb_id=kb_id,
                    text=chunk.text,
                    score=getattr(chunk, "score", 0.0),
                    metadata=chunk.metadata or {},
                ))
                if len(hits) >= top_k:
                    break
            return hits
        except Exception as exc:
            log.warning("memory_bm25.search_failed", error=str(exc))
            return []

    async def delete_by_doc_id(self, doc_id: str, kb_id: str) -> None:
        if kb_id in self._docs:
            self._docs[kb_id] = [d for d in self._docs[kb_id] if d["doc_id"] != doc_id]
            await self._rebuild_index(kb_id)
            if self._persist:
                await asyncio.to_thread(self._save_pickle)

    async def delete_by_kb_id(self, kb_id: str) -> None:
        self._docs.pop(kb_id, None)
        self._indices.pop(kb_id, None)
        if self._persist:
            await asyncio.to_thread(self._save_pickle)
