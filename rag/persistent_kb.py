"""
rag/persistent_kb.py — 持久化知识库

在 KnowledgeBase 的基础上增加 SQLite 持久化：
- 文档入库同时写入 kb_store
- 启动时从 kb_store 恢复所有分块到内存检索器
- 支持 kb_id 多租户隔离
"""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from rag.knowledge_base import (
    Chunk,
    DocumentIngester,
    HybridRetriever,
    LLMReranker,
    TextChunker,
)
from rag.store import KBChunk, KBDocument, KBStore

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: Chunk <-> KBChunk conversion
# ---------------------------------------------------------------------------

def _kb_chunk_to_chunk(kbc: KBChunk) -> Chunk:
    """Convert a stored KBChunk to an in-memory Chunk for HybridRetriever."""
    return Chunk(
        id          = kbc.chunk_id,
        doc_id      = kbc.doc_id,
        text        = kbc.text,
        chunk_index = kbc.chunk_index,
        metadata    = kbc.meta,
        embedding   = kbc.embedding,
        score       = 0.0,
    )


def _chunk_to_kb_chunk(
    chunk: Chunk,
    kb_id: str,
    doc_id: str,
) -> KBChunk:
    """Convert an in-memory Chunk to a KBChunk ready for persistence."""
    return KBChunk(
        chunk_id    = chunk.id,
        doc_id      = doc_id,
        kb_id       = kb_id,
        chunk_index = chunk.chunk_index,
        text        = chunk.text,
        embedding   = chunk.embedding,
        meta        = chunk.metadata or {},
        created_at  = time.time(),
    )


# ---------------------------------------------------------------------------
# PersistentKnowledgeBase
# ---------------------------------------------------------------------------

class PersistentKnowledgeBase:
    """
    持久化知识库，支持两种后端：

    模式 A — 传统 SQLite（默认，向后兼容）
        vector_store=None
        分块向量存 SQLite BLOB，in-memory numpy cosine 检索

    模式 B — Qdrant 向量数据库（推荐）
        vector_store=QdrantVectorStore(...)
        SQLite 仅保留文档元数据；向量/分块存 Qdrant
        HNSW ANN 检索 O(log n)，内存 BM25 从 Qdrant payload 重建
    """

    def __init__(
        self,
        store:           KBStore,
        embed_fn:        Any,
        llm_engine:      Any    = None,
        kb_id:           str    = "global",
        use_reranker:    bool   = False,
        chunk_max_chars: int    = 800,
        vector_store:    Any    = None,   # QdrantVectorStore | None
    ) -> None:
        self._store        = store         # SQLite — document metadata
        self._vector_store = vector_store  # Qdrant  — vectors + payloads (optional)
        self._embed_fn     = embed_fn
        self._kb_id        = kb_id
        self._ingester     = DocumentIngester()
        self._chunker      = TextChunker(max_chars=chunk_max_chars)
        self._reranker     = LLMReranker(llm_engine) if (use_reranker and llm_engine) else None
        self._initialized  = False

        # Legacy in-memory retriever (only used when vector_store is None)
        self._retriever = HybridRetriever(embed_fn=embed_fn) if vector_store is None else None

    # ------------------------------------------------------------------
    # initialize — restore in-memory index from store
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Warm up the retrieval backend.

        Qdrant mode:  calls vector_store.initialize() which connects to
                      Qdrant and rebuilds the in-memory BM25 from payloads.
        SQLite mode:  loads chunks+embeddings from SQLite into the
                      in-memory HybridRetriever (legacy behaviour).
        """
        if self._initialized:
            return
        try:
            if self._vector_store is not None:
                try:
                    await self._vector_store.initialize()
                    self._initialized = True
                    log.info("rag.persistent_kb.initialized_vector_store",
                             backend=type(self._vector_store).__name__, kb_id=self._kb_id)
                    return
                except Exception as vs_exc:
                    log.warning("rag.persistent_kb.vector_store_init_failed",
                                backend=type(self._vector_store).__name__,
                                error=str(vs_exc),
                                fallback="sqlite")
                    self._vector_store = None  # fall through to SQLite path
                    # Lazily create in-memory retriever that was skipped in __init__
                    if self._retriever is None:
                        self._retriever = HybridRetriever(embed_fn=self._embed_fn)

            # ── Legacy SQLite path ─────────────────────────────────────
            kb_chunks = await self._store.list_chunks_by_kb(self._kb_id, limit=10_000)
            if not kb_chunks:
                self._initialized = True
                log.info("rag.persistent_kb.initialized_empty", kb_id=self._kb_id)
                return

            chunks_with_emb = [c for c in kb_chunks if c.embedding]
            if not chunks_with_emb:
                self._initialized = True
                log.info(
                    "rag.persistent_kb.initialized_no_embeddings",
                    kb_id=self._kb_id,
                    total=len(kb_chunks),
                )
                return

            mem_chunks: list[Chunk] = [_kb_chunk_to_chunk(c) for c in chunks_with_emb]
            await self._load_chunks_into_retriever(mem_chunks)
            self._initialized = True
            log.info(
                "rag.persistent_kb.initialized",
                kb_id=self._kb_id,
                chunks=len(mem_chunks),
            )
        except Exception as exc:
            log.warning("rag.persistent_kb.init_failed", error=str(exc), kb_id=self._kb_id)
            self._initialized = True   # don't retry indefinitely

    async def _load_chunks_into_retriever(self, chunks: list[Chunk]) -> None:
        """Legacy SQLite mode only — inject pre-embedded chunks into HybridRetriever."""
        for chunk in chunks:
            self._retriever._chunks.append(chunk)
            self._retriever._embeddings.append(chunk.embedding or [])
        self._retriever._bm25.build(self._retriever._chunks)

    # ------------------------------------------------------------------
    # add_file
    # ------------------------------------------------------------------

    async def add_file(
        self,
        path: str | Path,
        kb_id: str | None = None,
        filename: str | None = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> KBDocument:
        """
        Ingest a file, persist metadata + chunks to store, index in memory.
        Returns a KBDocument with status='ready' (or 'error').

        Args:
            path:         Path to the file on disk.
            kb_id:        Knowledge-base tenant ID.
            filename:     Override the display filename (defaults to path.name).
            on_progress:  Optional callback(chunk_idx, total_chunks) called after
                          each chunk is embedded so callers can stream progress.
        """
        effective_kb = kb_id or self._kb_id
        p = Path(path)
        suffix = p.suffix.lower().lstrip(".") or "txt"
        display_name = filename or p.name

        # Derive a stable doc_id from the file path
        doc_id = hashlib.md5(str(p).encode()).hexdigest()[:16]

        doc = KBDocument(
            doc_id     = doc_id,
            kb_id      = effective_kb,
            filename   = display_name,
            source     = str(p),
            doc_type   = suffix,
            created_at = time.time(),
            status     = "pending",
        )
        doc = await self._store.save_document(doc)

        try:
            raw_doc = self._ingester.ingest_file(p)
            raw_doc.metadata["kb_id"] = effective_kb
            doc.char_count = len(raw_doc.content)

            chunks = self._chunker.chunk(raw_doc)
            await self._index_document(doc, chunks, on_progress=on_progress)
        except Exception as exc:
            log.warning(
                "rag.persistent_kb.add_file_failed",
                path=str(p),
                error=str(exc),
            )
            await self._store.update_document_status(
                doc.doc_id, "error", error_msg=str(exc)
            )
            doc.status    = "error"
            doc.error_msg = str(exc)

        return doc

    # ------------------------------------------------------------------
    # add_text
    # ------------------------------------------------------------------

    async def add_text(
        self,
        text:        str,
        source:      str  = "inline",
        kb_id:       str | None = None,
        filename:    str  = "",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> KBDocument:
        """
        Ingest raw text, persist to store, index in memory.
        Returns a KBDocument with status='ready' (or 'error').
        """
        effective_kb = kb_id or self._kb_id
        doc_id = hashlib.md5((text[:200] + source).encode()).hexdigest()[:16]

        doc = KBDocument(
            doc_id     = doc_id,
            kb_id      = effective_kb,
            filename   = filename or source,
            source     = source,
            doc_type   = "inline",
            char_count = len(text),
            created_at = time.time(),
            status     = "pending",
        )
        doc = await self._store.save_document(doc)

        try:
            raw_doc = self._ingester.ingest_text(text, source)
            raw_doc.metadata["kb_id"] = effective_kb
            if filename:
                raw_doc.metadata["filename"] = filename

            chunks = self._chunker.chunk(raw_doc)
            await self._index_document(doc, chunks, on_progress=on_progress)
        except Exception as exc:
            log.warning(
                "rag.persistent_kb.add_text_failed",
                source=source,
                error=str(exc),
            )
            await self._store.update_document_status(
                doc.doc_id, "error", error_msg=str(exc)
            )
            doc.status    = "error"
            doc.error_msg = str(exc)

        return doc

    # ------------------------------------------------------------------
    # _index_document — embed + persist + add to in-memory retriever
    # ------------------------------------------------------------------

    async def _index_document(
        self,
        doc:         KBDocument,
        chunks:      list[Chunk],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        1. Mark doc as 'indexing'
        2. Embed each chunk (calls on_progress(idx, total) after each)
        3a. Qdrant / Milvus mode — upsert vectors+payloads to vector store
        3b. SQLite mode          — save KBChunks to SQLite + load into HybridRetriever
        4. Mark doc as 'ready'
        """
        await self._store.update_document_status(doc.doc_id, "indexing")
        doc.status = "indexing"

        total_chunks = len(chunks)
        try:
            ts = time.time()
            qdrant_payloads: list[dict]    = []
            kb_chunks_sqlite: list[KBChunk] = []
            mem_chunks:       list[Chunk]   = []

            for chunk_idx, chunk in enumerate(chunks):
                embedding: list[float] | None = None
                try:
                    embedding = await self._embed_fn(chunk.text)
                except Exception as emb_exc:
                    log.warning(
                        "rag.persistent_kb.embed_failed",
                        chunk_id=chunk.id,
                        error=str(emb_exc),
                    )

                chunk.embedding = embedding

                # Notify progress after each embedding
                if on_progress is not None:
                    try:
                        on_progress(chunk_idx + 1, total_chunks)
                    except Exception:
                        pass  # never let a callback crash indexing

                meta = {**chunk.metadata, "doc_id": doc.doc_id,
                        "filename": doc.filename, "kb_id": doc.kb_id}

                if self._vector_store is not None:
                    # Vector store mode (Milvus / Qdrant): collect payload dict for batch upsert
                    qdrant_payloads.append({
                        "chunk_id":    chunk.id,
                        "doc_id":      doc.doc_id,
                        "kb_id":       doc.kb_id,
                        "chunk_index": chunk.chunk_index,
                        "text":        chunk.text,
                        "embedding":   embedding,
                        "meta":        meta,
                        "created_at":  ts,
                    })
                else:
                    # SQLite mode: build KBChunk + in-memory Chunk
                    kb_chunks_sqlite.append(KBChunk(
                        chunk_id    = chunk.id,
                        doc_id      = doc.doc_id,
                        kb_id       = doc.kb_id,
                        chunk_index = chunk.chunk_index,
                        text        = chunk.text,
                        embedding   = embedding,
                        meta        = meta,
                        created_at  = ts,
                    ))
                    mem_chunks.append(Chunk(
                        id          = chunk.id,
                        doc_id      = doc.doc_id,
                        text        = chunk.text,
                        chunk_index = chunk.chunk_index,
                        metadata    = meta,
                        embedding   = embedding,
                        score       = 0.0,
                    ))

            if self._vector_store is not None:
                # Batch upsert to vector store (Milvus / Qdrant)
                await self._vector_store.upsert_chunks(qdrant_payloads)
            else:
                # Persist to SQLite + load into HybridRetriever
                await self._store.save_chunks(kb_chunks_sqlite)
                await self._load_chunks_into_retriever(mem_chunks)

            await self._store.update_document_status(
                doc.doc_id, "ready", chunk_count=len(chunks),
            )
            doc.status      = "ready"
            doc.chunk_count = len(chunks)

            log.info(
                "rag.persistent_kb.indexed",
                doc_id=doc.doc_id,
                kb_id=doc.kb_id,
                chunks=len(chunks),
                backend=type(self._vector_store).__name__ if self._vector_store else "sqlite",
            )

        except Exception as exc:
            log.error(
                "rag.persistent_kb.index_failed",
                doc_id=doc.doc_id,
                error=str(exc),
            )
            await self._store.update_document_status(
                doc.doc_id, "error", error_msg=str(exc)
            )
            doc.status    = "error"
            doc.error_msg = str(exc)
            raise

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    async def query(
        self,
        query:   str,
        kb_id:   str | None       = None,
        top_k:   int              = 5,
        doc_ids: list[str] | None = None,
    ) -> list[KBChunk]:
        """
        Hybrid search (vector + BM25, RRF fusion).

        Vector store mode (Milvus / Qdrant):
            embed query → vector HNSW + BM25 → RRF
        SQLite mode:
            embed query → numpy cosine + in-memory BM25 → RRF

        doc_ids: 非空时只在指定文档范围内检索（单/多文件问答）
        """
        if not self._initialized:
            await self.initialize()

        effective_kb = kb_id or self._kb_id
        doc_filter   = doc_ids if doc_ids else None

        if self._vector_store is not None:
            # ── Vector store path (Milvus / Qdrant) ────────────────────
            try:
                query_vec = await self._embed_fn(query)
            except Exception as exc:
                log.warning("rag.persistent_kb.embed_query_failed", error=str(exc))
                query_vec = None

            if query_vec:
                hits = await self._vector_store.hybrid_search(
                    query_vec=query_vec,
                    query_text=query,
                    kb_id=effective_kb,
                    top_k=top_k,
                    doc_ids=doc_filter,
                )
            else:
                hits = []

            result: list[KBChunk] = []
            for score, payload in hits:
                result.append(KBChunk(
                    chunk_id    = payload["chunk_id"],
                    doc_id      = payload["doc_id"],
                    kb_id       = payload.get("kb_id", effective_kb),
                    chunk_index = payload.get("chunk_index", 0),
                    text        = payload.get("text", ""),
                    embedding   = None,
                    meta        = payload.get("meta", {}),
                    created_at  = payload.get("created_at", 0.0),
                    score       = round(score, 6),
                ))
            return result

        # ── Legacy SQLite / HybridRetriever path ──────────────────────
        mem_chunks = await self._retriever.search(query, top_k=top_k * 4)
        filtered = [
            c for c in mem_chunks
            if c.metadata.get("kb_id", effective_kb) == effective_kb
            and (doc_filter is None or c.doc_id in doc_filter)
        ]
        if self._reranker and filtered:
            filtered = await self._reranker.rerank(query, filtered, top_k=top_k)
        filtered = filtered[:top_k]

        return [
            KBChunk(
                chunk_id    = c.id,
                doc_id      = c.doc_id,
                kb_id       = effective_kb,
                chunk_index = c.chunk_index,
                text        = c.text,
                embedding   = None,
                meta        = c.metadata or {},
                created_at  = 0.0,
                score       = getattr(c, "score", 0.0),
            )
            for c in filtered
        ]

    # ------------------------------------------------------------------
    # delete_document
    # ------------------------------------------------------------------

    async def delete_document(self, doc_id: str) -> None:
        """Remove the document and all its chunks from storage + retriever."""
        # Remove from document metadata store (SQLite)
        await self._store.delete_document(doc_id)

        if self._vector_store is not None:
            # Qdrant: targeted delete, BM25 updated inside delete_by_doc_id
            await self._vector_store.delete_by_doc_id(doc_id)
        else:
            # SQLite: rebuild full in-memory retriever
            self._retriever._chunks     = []
            self._retriever._embeddings = []
            self._retriever._bm25.build([])
            self._initialized = False
            await self.initialize()

        log.info("rag.persistent_kb.document_deleted", doc_id=doc_id)

    # ------------------------------------------------------------------
    # reindex_document
    # ------------------------------------------------------------------

    async def reindex_document(self, doc_id: str) -> KBDocument:
        """
        Re-embed any chunks that are missing embeddings for the given document,
        then mark the document as 'ready'.  Useful for recovering docs that
        failed during the initial indexing run.
        """
        doc = await self._store.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id!r} not found")

        kb_chunks = await self._store.list_chunks(doc_id)
        if not kb_chunks:
            raise ValueError(f"No chunks found for document {doc_id!r}")

        await self._store.update_document_status(doc_id, "indexing")
        doc.status = "indexing"

        embedded = 0
        for kbc in kb_chunks:
            if kbc.embedding:
                embedded += 1
                continue
            try:
                emb = await self._embed_fn(kbc.text)
                await self._store.update_chunk_embedding(kbc.chunk_id, emb)
                embedded += 1
            except Exception as exc:
                log.warning(
                    "rag.persistent_kb.reindex_embed_failed",
                    chunk_id=kbc.chunk_id,
                    error=str(exc),
                )

        if self._vector_store is not None:
            # Qdrant: re-upsert updated chunks with new embeddings
            kb_chunks2 = await self._store.list_chunks(doc_id)
            payloads = []
            for kbc in kb_chunks2:
                if kbc.embedding:
                    payloads.append({
                        "chunk_id":    kbc.chunk_id,
                        "doc_id":      kbc.doc_id,
                        "kb_id":       kbc.kb_id,
                        "chunk_index": kbc.chunk_index,
                        "text":        kbc.text,
                        "embedding":   kbc.embedding,
                        "meta":        kbc.meta,
                        "created_at":  kbc.created_at,
                    })
            if payloads:
                await self._vector_store.upsert_chunks(payloads)
        else:
            # SQLite: rebuild full in-memory retriever
            self._retriever._chunks     = []
            self._retriever._embeddings = []
            self._retriever._bm25.build([])
            self._initialized = False
            await self.initialize()

        await self._store.update_document_status(
            doc_id, "ready", chunk_count=embedded
        )
        doc.status      = "ready"
        doc.chunk_count = embedded
        log.info("rag.persistent_kb.reindexed", doc_id=doc_id, chunks=embedded)
        return doc

    # ------------------------------------------------------------------
    # list_documents
    # ------------------------------------------------------------------

    async def list_documents(
        self, kb_id: str | None = None
    ) -> list[KBDocument]:
        effective_kb = kb_id or self._kb_id
        return await self._store.list_documents(effective_kb)

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    async def get_stats(self, kb_id: str | None = None) -> dict:
        effective_kb = kb_id or self._kb_id
        if self._vector_store is not None:
            docs  = await self._store.list_documents(effective_kb)
            vstats = await self._vector_store.get_stats(effective_kb)
            ready = sum(1 for d in docs if d.status == "ready")
            return {
                "doc_count":        len(docs),
                "ready_doc_count":  ready,
                "chunk_count":      vstats["chunk_count"],
                "total_chars":      vstats["total_chars"],
                "backend":          "qdrant",
            }
        stats = await self._store.get_stats(effective_kb)
        stats["backend"] = "sqlite"
        return stats

    # ------------------------------------------------------------------
    # format_context
    # ------------------------------------------------------------------

    @staticmethod
    def format_context(chunks: list[KBChunk]) -> str:
        """
        Format retrieved KBChunks into a string suitable for LLM prompt
        injection, with numeric source citations [1][2]...
        """
        if not chunks:
            return ""
        parts = ["## 相关知识库内容\n"]
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.meta.get("filename", "") or chunk.meta.get("source", chunk.doc_id)
            parts.append(f"[{i}] 来源: {filename}\n{chunk.text}\n")
        return "\n".join(parts)
