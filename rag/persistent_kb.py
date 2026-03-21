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
from typing import Any

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
    Wraps an in-memory HybridRetriever with SQLite persistence via KBStore.

    - Documents and chunks are written to the store on ingestion.
    - On initialize(), all "ready" chunks are loaded back into the
      in-memory retriever so retrieval works after a server restart.
    - Supports multi-tenancy via kb_id.
    """

    def __init__(
        self,
        store:        KBStore,
        embed_fn:     Any,
        llm_engine:   Any    = None,
        kb_id:        str    = "global",
        use_reranker: bool   = False,
        chunk_max_chars: int = 800,
    ) -> None:
        self._store       = store
        self._embed_fn    = embed_fn
        self._kb_id       = kb_id
        self._ingester    = DocumentIngester()
        self._chunker     = TextChunker(max_chars=chunk_max_chars)
        self._retriever   = HybridRetriever(embed_fn=embed_fn)
        self._reranker    = LLMReranker(llm_engine) if (use_reranker and llm_engine) else None
        self._initialized = False

    # ------------------------------------------------------------------
    # initialize — restore in-memory index from store
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Load all 'ready' chunks from the store into the in-memory retriever.
        This is called once at startup so that the retriever is warm.
        """
        if self._initialized:
            return
        try:
            kb_chunks = await self._store.list_chunks_by_kb(self._kb_id, limit=10_000)
            if not kb_chunks:
                self._initialized = True
                log.info("rag.persistent_kb.initialized_empty", kb_id=self._kb_id)
                return

            # Only restore chunks that have embeddings (docs whose status is ready)
            chunks_with_emb = [c for c in kb_chunks if c.embedding]
            if not chunks_with_emb:
                self._initialized = True
                log.info(
                    "rag.persistent_kb.initialized_no_embeddings",
                    kb_id=self._kb_id,
                    total=len(kb_chunks),
                )
                return

            # Convert and inject directly — no re-embedding needed
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
            self._initialized = True   # mark done so we don't retry indefinitely

    async def _load_chunks_into_retriever(self, chunks: list[Chunk]) -> None:
        """
        Add already-embedded Chunk objects to the retriever WITHOUT calling
        embed_fn again.  We directly append to the retriever's internal lists
        and rebuild the BM25 index.
        """
        for chunk in chunks:
            self._retriever._chunks.append(chunk)
            self._retriever._embeddings.append(chunk.embedding or [])
        self._retriever._bm25.build(self._retriever._chunks)

    # ------------------------------------------------------------------
    # add_file
    # ------------------------------------------------------------------

    async def add_file(
        self, path: str | Path, kb_id: str | None = None
    ) -> KBDocument:
        """
        Ingest a file, persist metadata + chunks to store, index in memory.
        Returns a KBDocument with status='ready' (or 'error').
        """
        effective_kb = kb_id or self._kb_id
        p = Path(path)
        suffix = p.suffix.lower().lstrip(".") or "txt"

        # Derive a stable doc_id from the file path
        doc_id = hashlib.md5(str(p).encode()).hexdigest()[:16]

        doc = KBDocument(
            doc_id     = doc_id,
            kb_id      = effective_kb,
            filename   = p.name,
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
            await self._index_document(doc, chunks)
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
        text:     str,
        source:   str  = "inline",
        kb_id:    str | None = None,
        filename: str  = "",
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
            await self._index_document(doc, chunks)
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
        doc:    KBDocument,
        chunks: list[Chunk],
    ) -> None:
        """
        1. Mark doc as 'indexing'
        2. For each chunk: call embed_fn, create KBChunk with embedding, collect
        3. Batch-save KBChunks to store
        4. Build Chunk objects for in-memory HybridRetriever (no re-embed)
        5. Mark doc as 'ready', update chunk_count
        """
        await self._store.update_document_status(doc.doc_id, "indexing")
        doc.status = "indexing"

        try:
            kb_chunks_to_save: list[KBChunk] = []
            mem_chunks:        list[Chunk]   = []

            for chunk in chunks:
                # Embed
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

                # Build KBChunk for persistence
                kb_chunk = KBChunk(
                    chunk_id    = chunk.id,
                    doc_id      = doc.doc_id,
                    kb_id       = doc.kb_id,
                    chunk_index = chunk.chunk_index,
                    text        = chunk.text,
                    embedding   = embedding,
                    meta        = {**chunk.metadata, "doc_id": doc.doc_id},
                    created_at  = time.time(),
                )
                kb_chunks_to_save.append(kb_chunk)

                # Build in-memory Chunk with embedding already set
                mem_chunk = Chunk(
                    id          = chunk.id,
                    doc_id      = doc.doc_id,
                    text        = chunk.text,
                    chunk_index = chunk.chunk_index,
                    metadata    = kb_chunk.meta,
                    embedding   = embedding,
                    score       = 0.0,
                )
                mem_chunks.append(mem_chunk)

            # Persist all chunks in one batch
            await self._store.save_chunks(kb_chunks_to_save)

            # Add to in-memory retriever WITHOUT re-embedding
            await self._load_chunks_into_retriever(mem_chunks)

            # Mark document ready
            await self._store.update_document_status(
                doc.doc_id,
                "ready",
                chunk_count=len(chunks),
            )
            doc.status      = "ready"
            doc.chunk_count = len(chunks)

            log.info(
                "rag.persistent_kb.indexed",
                doc_id=doc.doc_id,
                kb_id=doc.kb_id,
                chunks=len(chunks),
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
        query:  str,
        kb_id:  str | None = None,
        top_k:  int        = 5,
    ) -> list[KBChunk]:
        """
        Hybrid search (vector + BM25) over the in-memory retriever.
        Returns results as KBChunk objects (with embeddings stripped for
        bandwidth efficiency).
        """
        if not self._initialized:
            await self.initialize()

        mem_chunks = await self._retriever.search(query, top_k=top_k * 2)

        if self._reranker:
            mem_chunks = await self._reranker.rerank(query, mem_chunks, top_k=top_k)

        mem_chunks = mem_chunks[:top_k]

        # Convert back to KBChunk (drop embedding for response payload)
        result: list[KBChunk] = []
        for c in mem_chunks:
            result.append(
                KBChunk(
                    chunk_id    = c.id,
                    doc_id      = c.doc_id,
                    kb_id       = kb_id or self._kb_id,
                    chunk_index = c.chunk_index,
                    text        = c.text,
                    embedding   = None,   # omit in query results
                    meta        = c.metadata or {},
                    created_at  = 0.0,
                )
            )
        return result

    # ------------------------------------------------------------------
    # delete_document
    # ------------------------------------------------------------------

    async def delete_document(self, doc_id: str) -> None:
        """
        Remove the document and its chunks from the store, then rebuild
        the in-memory retriever from the remaining stored chunks.
        """
        await self._store.delete_document(doc_id)

        # Rebuild in-memory index from the store (excluding deleted doc)
        self._retriever._chunks     = []
        self._retriever._embeddings = []
        self._retriever._bm25.build([])
        self._initialized = False   # trigger re-load on next query
        await self.initialize()

        log.info("rag.persistent_kb.document_deleted", doc_id=doc_id)

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
        return await self._store.get_stats(effective_kb)

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
