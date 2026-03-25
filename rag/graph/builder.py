"""rag/graph/builder.py — 知识图谱构建管线"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable

import structlog

from rag.graph.community import CommunitySummarizer
from rag.graph.extractor import TripleExtractor
from rag.graph.models import Edge, Node, Triple
from rag.graph.resolver import EntityResolver
from rag.graph.store import KGStore

logger = structlog.get_logger(__name__)

# Default chunk size for text splitting (characters)
_DEFAULT_CHUNK_SIZE = 800
# Overlap between chunks (characters)
_DEFAULT_CHUNK_OVERLAP = 100


def _split_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Simple character-based text splitter with overlap.

    Tries to break at sentence boundaries (。！？\n) within the chunk window.
    Falls back to hard character splits if no boundary found.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Try to find a natural break point (Chinese/English sentence ends)
            boundary_chars = set("。！？\n.!?")
            best_break = -1
            # Search backwards from end for a boundary
            for i in range(end - 1, max(start, end - 100) - 1, -1):
                if text[i] in boundary_chars:
                    best_break = i + 1
                    break
            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start with overlap
        start = max(start + 1, end - overlap)

    return chunks


class GraphBuilder:
    """
    知识图谱构建管线：协调 TripleExtractor、EntityResolver 和存储写入。

    Args:
        store: KGStore 实例（已 initialize）。
        extractor: TripleExtractor 实例。
        resolver: EntityResolver 实例。
        embed_fn: 嵌入函数，用于节点嵌入向量生成（可选）。
        default_concurrency: 批量处理的默认并发数。
    """

    def __init__(
        self,
        store: KGStore,
        extractor: TripleExtractor,
        resolver: EntityResolver,
        embed_fn: Callable[[str], Any] | None = None,
        default_concurrency: int = 4,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._resolver = resolver
        self._embed_fn = embed_fn
        self._concurrency = default_concurrency

        # In-memory job tracking dict
        self._jobs: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_from_chunks(
        self,
        chunks: list[dict],
        kb_id: str,
        doc_id: str,
    ) -> dict:
        """
        从文本块列表构建知识图谱。

        Args:
            chunks: 每个元素为含 text/content/doc_id/chunk_id/id 键的字典。
            kb_id: 知识库 ID。
            doc_id: 文档 ID（当块中未指定时使用）。

        Returns:
            dict: {"nodes_created": int, "edges_created": int, "triples_found": int}
        """
        logger.info(
            "build_from_chunks_start",
            kb_id=kb_id,
            doc_id=doc_id,
            num_chunks=len(chunks),
        )

        # Step 1: Extract triples from all chunks
        all_triples: list[Triple] = await self._extractor.extract_batch(
            chunks, concurrency=self._concurrency
        )
        triples_found = len(all_triples)
        logger.info(
            "triples_extracted",
            kb_id=kb_id,
            doc_id=doc_id,
            count=triples_found,
        )

        if not all_triples:
            return {"nodes_created": 0, "edges_created": 0, "triples_found": 0}

        # Step 2: Resolve entities (deduplicate and create nodes)
        resolved_triples = await self._resolver.resolve(all_triples, kb_id)
        logger.info(
            "entities_resolved",
            kb_id=kb_id,
            doc_id=doc_id,
            resolved_count=len(resolved_triples),
        )

        # Step 3: Write edges to store
        # At this point nodes already exist in store (created by resolver).
        # We build edges from resolved triples.
        edges_created = 0
        nodes_updated = 0

        # Track node_id -> Triple mapping for doc_id association
        node_name_to_id: dict[str, str] = {}

        # Look up canonical node IDs
        for triple in resolved_triples:
            for name in (triple.src_name, triple.dst_name):
                if name not in node_name_to_id:
                    node = await self._store.get_node_by_name(name, kb_id)
                    if node:
                        node_name_to_id[name] = node.id

        # Associate doc_id with nodes
        updated_node_ids: set[str] = set()
        for triple in resolved_triples:
            for name in (triple.src_name, triple.dst_name):
                node_id = node_name_to_id.get(name)
                if node_id and node_id not in updated_node_ids:
                    node = await self._store.get_node(node_id)
                    if node and doc_id not in node.doc_ids:
                        node.doc_ids = list(node.doc_ids) + [doc_id]
                        await self._store.upsert_node(node)
                        updated_node_ids.add(node_id)
                        nodes_updated += 1

        # Create edges from triples
        for triple in resolved_triples:
            src_id = node_name_to_id.get(triple.src_name)
            dst_id = node_name_to_id.get(triple.dst_name)

            if src_id is None or dst_id is None:
                logger.warning(
                    "edge_skipped_missing_node",
                    src_name=triple.src_name,
                    dst_name=triple.dst_name,
                )
                continue

            edge_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{kb_id}:{src_id}:{triple.relation}:{dst_id}",
                )
            )
            edge = Edge(
                id=edge_id,
                kb_id=kb_id,
                src_id=src_id,
                dst_id=dst_id,
                relation=triple.relation,
                weight=triple.confidence,
                doc_id=doc_id,
                chunk_id="",
                context=triple.evidence[:500] if triple.evidence else "",
                created_at=time.time(),
            )
            await self._store.upsert_edge(edge)
            edges_created += 1

        # Count nodes actually created for this doc
        stats = await self._store.get_stats(kb_id)
        nodes_in_kb = stats.get("nodes", 0)

        result = {
            "nodes_created": nodes_updated,
            "edges_created": edges_created,
            "triples_found": triples_found,
        }
        logger.info("build_from_chunks_done", kb_id=kb_id, doc_id=doc_id, **result)
        return result

    async def build_from_text(
        self,
        text: str,
        kb_id: str,
        doc_id: str,
        source: str = "",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> dict:
        """
        从原始文本构建知识图谱（先分块，再调用 build_from_chunks）。

        Args:
            text: 输入文本。
            kb_id: 知识库 ID。
            doc_id: 文档 ID。
            source: 来源描述（用于日志）。
            chunk_size: 每个文本块的字符数（默认 800）。

        Returns:
            dict: {"nodes_created": int, "edges_created": int, "triples_found": int, "chunks": int}
        """
        logger.info(
            "build_from_text_start",
            kb_id=kb_id,
            doc_id=doc_id,
            source=source,
            text_length=len(text),
        )

        # Split text into chunks
        raw_chunks = _split_text(text, chunk_size=chunk_size)
        if not raw_chunks:
            logger.warning("build_from_text_empty", kb_id=kb_id, doc_id=doc_id)
            return {"nodes_created": 0, "edges_created": 0, "triples_found": 0, "chunks": 0}

        # Build chunk dicts
        chunks = [
            {
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}__chunk_{i}",
            }
            for i, chunk_text in enumerate(raw_chunks)
        ]

        result = await self.build_from_chunks(chunks, kb_id=kb_id, doc_id=doc_id)
        result["chunks"] = len(chunks)

        logger.info(
            "build_from_text_done",
            kb_id=kb_id,
            doc_id=doc_id,
            chunks=len(chunks),
            **{k: v for k, v in result.items() if k != "chunks"},
        )
        return result

    async def rebuild_communities(
        self,
        kb_id: str,
        llm_engine: Any | None = None,
    ) -> dict:
        """
        重新运行社区检测和摘要生成。

        Args:
            kb_id: 知识库 ID。
            llm_engine: LLM 引擎（若为 None，则使用 extractor 的引擎）。

        Returns:
            dict: {"communities_created": int, "communities_skipped": int}
        """
        logger.info("rebuild_communities_start", kb_id=kb_id)

        # Use provided llm_engine or fall back to extractor's engine
        effective_llm = llm_engine or self._extractor._llm

        if effective_llm is None:
            logger.warning(
                "rebuild_communities_no_llm",
                kb_id=kb_id,
            )
            return {"communities_created": 0, "communities_skipped": 0}

        summarizer = CommunitySummarizer(
            llm_engine=effective_llm,
            store=self._store,
            embed_fn=self._embed_fn,
        )

        result = await summarizer.summarize_all(kb_id=kb_id, level=0)

        logger.info("rebuild_communities_done", kb_id=kb_id, **result)
        return result

    # ------------------------------------------------------------------
    # Async job tracking
    # ------------------------------------------------------------------

    def start_build_job(
        self,
        chunks: list[dict],
        kb_id: str,
        doc_id: str,
    ) -> str:
        """
        Start a background asyncio task for build_from_chunks and return a job_id.

        The job runs concurrently; use get_job_status(job_id) to check progress.

        Args:
            chunks: Text chunk dicts (same format as build_from_chunks).
            kb_id: Knowledge base ID.
            doc_id: Document ID.

        Returns:
            str: job_id (UUID string).
        """
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "error": None,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "created_at": time.time(),
            "finished_at": None,
        }

        # Fire and forget background task
        asyncio.create_task(self._run_build_job(job_id, chunks, kb_id, doc_id))

        logger.info(
            "build_job_started",
            job_id=job_id,
            kb_id=kb_id,
            doc_id=doc_id,
            num_chunks=len(chunks),
        )
        return job_id

    def start_build_text_job(
        self,
        text: str,
        kb_id: str,
        doc_id: str,
        source: str = "",
    ) -> str:
        """
        Start a background asyncio task for build_from_text and return a job_id.

        Args:
            text: Raw input text.
            kb_id: Knowledge base ID.
            doc_id: Document ID.
            source: Source description (for logging).

        Returns:
            str: job_id (UUID string).
        """
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "error": None,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "created_at": time.time(),
            "finished_at": None,
        }

        asyncio.create_task(
            self._run_build_text_job(job_id, text, kb_id, doc_id, source)
        )

        logger.info(
            "build_text_job_started",
            job_id=job_id,
            kb_id=kb_id,
            doc_id=doc_id,
        )
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        """
        Get the current status of a build job.

        Returns a dict with:
        - status: "pending" | "running" | "done" | "error"
        - progress: float 0.0~1.0
        - result: dict | None (populated when status == "done")
        - error: str | None (populated when status == "error")
        - kb_id: str
        - doc_id: str
        - created_at: float (unix timestamp)
        - finished_at: float | None

        Args:
            job_id: Job ID returned by start_build_job.

        Returns:
            dict: Job status information, or {"status": "not_found"} if unknown.
        """
        if job_id not in self._jobs:
            return {"status": "not_found", "job_id": job_id}
        return dict(self._jobs[job_id])

    def list_jobs(self, kb_id: str | None = None) -> list[dict]:
        """
        List all tracked jobs, optionally filtered by kb_id.

        Args:
            kb_id: Optional filter by knowledge base ID.

        Returns:
            list[dict]: List of job status dicts, sorted by created_at descending.
        """
        jobs = list(self._jobs.values())
        if kb_id is not None:
            jobs = [j for j in jobs if j.get("kb_id") == kb_id]
        return sorted(jobs, key=lambda j: j.get("created_at", 0), reverse=True)

    # ------------------------------------------------------------------
    # Background task runners
    # ------------------------------------------------------------------

    async def _run_build_job(
        self,
        job_id: str,
        chunks: list[dict],
        kb_id: str,
        doc_id: str,
    ) -> None:
        """Background task: run build_from_chunks and update job state."""
        job = self._jobs[job_id]
        job["status"] = "running"
        job["progress"] = 0.1

        try:
            result = await self.build_from_chunks(chunks, kb_id=kb_id, doc_id=doc_id)
            job["status"] = "done"
            job["progress"] = 1.0
            job["result"] = result
            job["finished_at"] = time.time()
            logger.info(
                "build_job_completed",
                job_id=job_id,
                kb_id=kb_id,
                doc_id=doc_id,
                **result,
            )
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            logger.error(
                "build_job_failed",
                job_id=job_id,
                kb_id=kb_id,
                doc_id=doc_id,
                error=str(exc),
                exc_info=True,
            )

    async def _run_build_text_job(
        self,
        job_id: str,
        text: str,
        kb_id: str,
        doc_id: str,
        source: str,
    ) -> None:
        """Background task: run build_from_text and update job state."""
        job = self._jobs[job_id]
        job["status"] = "running"
        job["progress"] = 0.1

        try:
            result = await self.build_from_text(
                text, kb_id=kb_id, doc_id=doc_id, source=source
            )
            job["status"] = "done"
            job["progress"] = 1.0
            job["result"] = result
            job["finished_at"] = time.time()
            logger.info(
                "build_text_job_completed",
                job_id=job_id,
                kb_id=kb_id,
                doc_id=doc_id,
                **result,
            )
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            logger.error(
                "build_text_job_failed",
                job_id=job_id,
                kb_id=kb_id,
                doc_id=doc_id,
                error=str(exc),
                exc_info=True,
            )
