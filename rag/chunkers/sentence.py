"""
rag/chunkers/sentence.py — 句子分块器（Chonkie SentenceChunker + 简单回退）

安装：pip install "chonkie[sentence]"
"""
from __future__ import annotations

import asyncio

import structlog

from rag.chunkers.base import Chunk
from rag.parsers.base import ParsedDocument

log = structlog.get_logger(__name__)


class SentenceChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunker = None

    def _get_chunker(self):
        if self._chunker is not None:
            return self._chunker
        try:
            from chonkie import SentenceChunker as _SC
            self._chunker = _SC(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        except ImportError:
            log.warning("chonkie.not_installed", hint="pip install 'chonkie[sentence]'")
        return self._chunker

    async def chunk(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        chunker = self._get_chunker()
        if chunker is not None:
            try:
                raw = await asyncio.to_thread(chunker.chunk, doc.content)
                return [
                    Chunk(
                        text=getattr(rc, "text", str(rc)).strip(),
                        chunk_index=i, doc_id=doc_id, source=doc.source,
                        metadata={"format": doc.format, "title": doc.title},
                    )
                    for i, rc in enumerate(raw)
                    if getattr(rc, "text", str(rc)).strip()
                ]
            except Exception as exc:
                log.warning("sentence_chunker.failed", error=str(exc))

        return self._simple_chunk(doc, doc_id)

    def _simple_chunk(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        """无 Chonkie 时按句子边界做字符级切分。"""
        text = doc.content
        size = self._chunk_size * 3
        overlap = self._chunk_overlap * 3
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                for boundary in ('。', '！', '？', '\n', '.', '!', '?'):
                    pos = text.rfind(boundary, start, end)
                    if pos > start:
                        end = pos + 1
                        break
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text, chunk_index=idx,
                    doc_id=doc_id, source=doc.source,
                    metadata={"format": doc.format},
                ))
                idx += 1
            start = max(start + 1, end - overlap)
        return chunks
