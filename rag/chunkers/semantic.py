"""
rag/chunkers/semantic.py — 语义分块器（Chonkie）

基于相邻句子 Embedding 余弦相似度检测语义边界切分。
无结构文档的首选分块策略。

安装：pip install "chonkie[sentence]"
降级：自动回退到 SentenceChunker
"""
from __future__ import annotations

import asyncio

import structlog

from rag.chunkers.base import Chunk
from rag.parsers.base import ParsedDocument

log = structlog.get_logger(__name__)


class SemanticChunker:
    def __init__(
        self,
        threshold: float = 0.5,
        target_size: int = 512,
        max_size: int = 800,
    ) -> None:
        self._threshold = threshold
        self._target_size = target_size
        self._max_size = max_size
        self._chunker = None

    def _get_chunker(self):
        if self._chunker is not None:
            return self._chunker
        try:
            from chonkie import SemanticChunker as _SC
            self._chunker = _SC(chunk_size=self._target_size, threshold=self._threshold)
        except ImportError:
            log.warning("chonkie.not_installed", hint="pip install 'chonkie[sentence]'")
        except Exception as exc:
            log.warning("chonkie.init_failed", error=str(exc))
        return self._chunker

    async def chunk(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        chunker = self._get_chunker()
        if chunker is None:
            return await self._fallback(doc, doc_id)
        try:
            raw = await asyncio.to_thread(chunker.chunk, doc.content)
            chunks = []
            for i, rc in enumerate(raw):
                text = getattr(rc, "text", str(rc)).strip()
                if text:
                    chunks.append(Chunk(
                        text=text, chunk_index=i,
                        doc_id=doc_id, source=doc.source,
                        metadata={"format": doc.format, "title": doc.title},
                    ))
            log.debug("semantic_chunker.done", doc_id=doc_id, chunks=len(chunks))
            return chunks
        except Exception as exc:
            log.warning("semantic_chunker.failed", error=str(exc))
            return await self._fallback(doc, doc_id)

    async def _fallback(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        from rag.chunkers.sentence import SentenceChunker
        return await SentenceChunker(
            chunk_size=self._target_size, chunk_overlap=64
        ).chunk(doc, doc_id)
