"""
rag/pipeline/ingestion.py — 文档写入流水线

流程：
  解析 → 分块 → Embedding → 并发写入（向量库 + 关键词库 + 图谱）
  → LLM 摘要生成 → 回调 KBFileManager 更新版本记录

设计原则：
  - 解析/Embedding 串行（避免 OOM）
  - 三个写入存储并发（asyncio.gather）
  - 摘要生成与写入并发（不阻塞入库）
  - 任何步骤失败记录 error 但不崩溃整个流水线
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import structlog

log = structlog.get_logger(__name__)


@dataclass
class IngestionResult:
    file_id: str
    version_id: str
    kb_id: str
    success: bool = True
    chunk_count: int = 0
    token_count: int = 0
    indexing_duration_ms: int = 0
    summary: str = ""
    error: str = ""


class IngestionPipeline:
    """
    文档写入流水线。

    依赖注入（均可通过 kb_config.yaml 切换实现）：
      parser         — 文档解析器（DoclingParser / UnstructuredParser / MarkItDownParser）
      chunker        — 分块器（SmartChunker）
      embedder       — Embedding 生成器（QwenEmbedder / OpenAIEmbedder / BGELocalEmbedder）
      vector_store   — 向量库（QdrantVectorStore / MilvusVectorStore / ChromaStore）
      keyword_store  — 关键词库（ESKeywordStore / MemoryKeywordStore）
      graph_builder  — 知识图谱构建器（GraphBuilder，可选）
      file_manager   — 文件生命周期管理（KBFileManager，可选）
      llm_fn         — LLM 摘要生成函数，签名 async (system, user) -> str（可选）
    """

    def __init__(
        self,
        parser,
        chunker,
        embedder,
        vector_store,
        keyword_store,
        graph_builder=None,
        file_manager=None,
        llm_fn: Callable | None = None,
        summary_enabled: bool = True,
        summary_max_chars: int = 3000,
        generate_diff_summary: bool = True,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_store = keyword_store
        self._graph_builder = graph_builder
        self._file_manager = file_manager
        self._llm_fn = llm_fn
        self._summary_enabled = summary_enabled
        self._summary_max_chars = summary_max_chars
        self._generate_diff = generate_diff_summary

    async def ingest(
        self,
        file_path: str,
        file_id: str,
        version_id: str,
        kb_id: str,
        doc_id: str | None = None,
        change_type: str = "created",
        prev_summary: str = "",
        on_progress: Callable[[int, int], None] | None = None,
    ) -> IngestionResult:
        """
        对单个文件执行完整写入流水线。

        Args:
            file_path:    本地文件路径
            file_id:      KBFile.id（用于回调文件管理器）
            version_id:   KBFileVersion.id（用于回调）
            kb_id:        知识库租户 ID
            doc_id:       文档 ID（默认使用 file_id）
            change_type:  created | content_updated | reindexed
            prev_summary: 上一版本摘要（用于 diff_summary 生成）
            on_progress:  进度回调 (current, total)
        """
        started = time.time()
        doc_id = doc_id or file_id
        result = IngestionResult(file_id=file_id, version_id=version_id, kb_id=kb_id)

        try:
            # ① 解析
            log.info("ingestion.parsing", file_id=file_id, path=file_path)
            parsed = await self._parser.parse(file_path)

            # ② 分块
            log.info("ingestion.chunking", file_id=file_id,
                     strategy="auto", has_headings=parsed.has_headings)
            chunks = await self._chunker.chunk(parsed, doc_id)
            total = len(chunks)
            log.info("ingestion.chunks_ready", file_id=file_id, count=total)

            # ③ Embedding（串行，避免内存爆炸）
            texts = [c.text for c in chunks]
            embeddings = await self._embedder.embed_batch(texts)

            # 进度回调
            if on_progress:
                try:
                    on_progress(total, total)
                except Exception:
                    pass

            # ④ 构建写入 payload
            now = time.time()
            chunk_payloads = [
                {
                    "chunk_id":    c.metadata.get("chunk_id", uuid.uuid4().hex),
                    "doc_id":      doc_id,
                    "kb_id":       kb_id,
                    "file_id":     file_id,
                    "chunk_index": c.chunk_index,
                    "text":        c.text,
                    "embedding":   embeddings[i] if i < len(embeddings) else [],
                    "heading_path": c.heading_path,
                    "metadata":    {
                        **c.metadata,
                        "source":   parsed.source,
                        "format":   parsed.format,
                        "title":    parsed.title,
                        "kb_id":    kb_id,
                        "doc_id":   doc_id,
                        "file_id":  file_id,
                    },
                    "created_at": now,
                }
                for i, c in enumerate(chunks)
            ]

            # 给每个 chunk 分配稳定 ID
            for p in chunk_payloads:
                if not p["chunk_id"] or len(p["chunk_id"]) < 8:
                    import hashlib
                    p["chunk_id"] = hashlib.md5(
                        f"{doc_id}_{p['chunk_index']}".encode()
                    ).hexdigest()

            # ⑤ 并发写入三个存储
            write_tasks = [
                self._vector_store.upsert_chunks(chunk_payloads),
                self._keyword_store.index_chunks(chunk_payloads, kb_id),
            ]
            if self._graph_builder:
                write_tasks.append(
                    self._graph_builder.build_from_chunks(chunks, doc_id=doc_id, kb_id=kb_id)
                )

            results = await asyncio.gather(*write_tasks, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    names = ["vector_store", "keyword_store", "graph_builder"]
                    log.warning("ingestion.write_failed",
                                store=names[i] if i < len(names) else f"store_{i}",
                                error=str(r))

            token_count = sum(len(c.text) for c in chunks)
            result.chunk_count = total
            result.token_count = token_count

            # ⑥ LLM 摘要生成（与主流程并发，不阻塞）
            summary = ""
            diff_summary = ""
            if self._llm_fn and self._summary_enabled:
                summary, diff_summary = await asyncio.gather(
                    self._gen_summary(parsed.content),
                    self._gen_diff_summary(parsed.content, prev_summary)
                    if change_type == "content_updated" and prev_summary else _noop(),
                )
            result.summary = summary

            # ⑦ 回调文件管理器更新版本记录
            elapsed_ms = int((time.time() - started) * 1000)
            result.indexing_duration_ms = elapsed_ms
            if self._file_manager:
                await self._file_manager.on_version_indexed(
                    file_id=file_id,
                    version_id=version_id,
                    summary=summary,
                    diff_summary=diff_summary,
                    chunk_count=total,
                    token_count=token_count,
                    indexing_duration_ms=elapsed_ms,
                )

            log.info("ingestion.done",
                     file_id=file_id, chunks=total, elapsed_ms=elapsed_ms)

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            log.error("ingestion.failed", file_id=file_id, error=str(exc))
            if self._file_manager:
                await self._file_manager.on_version_indexed(
                    file_id=file_id, version_id=version_id,
                    summary="", diff_summary="",
                    chunk_count=0, token_count=0,
                    indexing_duration_ms=int((time.time() - started) * 1000),
                    error=str(exc),
                )

        return result

    async def delete_doc(self, doc_id: str, kb_id: str) -> None:
        """从所有存储中删除指定文档的所有 chunk。"""
        tasks = [
            self._vector_store.delete_by_doc_id(doc_id),
            self._keyword_store.delete_by_doc_id(doc_id, kb_id),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("ingestion.doc_deleted", doc_id=doc_id, kb_id=kb_id)

    # ── LLM 摘要 ──────────────────────────────────────────────────────

    async def _gen_summary(self, content: str) -> str:
        try:
            return await self._llm_fn(
                system="用3-5句话概括以下文档的主要内容、关键信息和适用场景：",
                user=content[:self._summary_max_chars],
            )
        except Exception as exc:
            log.warning("ingestion.summary_failed", error=str(exc))
            return ""

    async def _gen_diff_summary(self, new_content: str, prev_summary: str) -> str:
        if not self._generate_diff or not prev_summary:
            return ""
        try:
            return await self._llm_fn(
                system="根据新旧文档信息，用1-2句话描述主要变更内容：",
                user=f"旧版本摘要：{prev_summary}\n\n新版本前500字：{new_content[:500]}",
            )
        except Exception as exc:
            log.warning("ingestion.diff_summary_failed", error=str(exc))
            return ""


async def _noop(*_):
    return ""
