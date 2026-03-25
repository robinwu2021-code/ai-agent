"""rag/chunkers/fixed.py — 固定大小分块器（无外部依赖兜底方案）"""
from __future__ import annotations

from rag.chunkers.base import Chunk
from rag.parsers.base import ParsedDocument


class FixedChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._size = chunk_size * 3
        self._overlap = chunk_overlap * 3

    async def chunk(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        text = doc.content
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self._size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text, chunk_index=idx,
                    doc_id=doc_id, source=doc.source,
                    metadata={"format": doc.format},
                ))
                idx += 1
            start = max(start + 1, end - self._overlap)
        return chunks
