"""
rag/chunkers/structural.py — 结构化分块器

按 Markdown 标题层级（H1/H2/H3）切分文档，保留层级面包屑上下文。
适用于有明确标题结构的文档（Markdown、DOCX 导出、PDF 带书签）。
"""
from __future__ import annotations

import re

import structlog

from rag.chunkers.base import Chunk
from rag.parsers.base import ParsedDocument

log = structlog.get_logger(__name__)

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')


class StructuralChunker:
    def __init__(
        self,
        min_size: int = 100,
        max_size: int = 1500,
        levels: list[int] | None = None,
        include_heading: bool = True,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        self._levels = set(levels or [1, 2, 3])
        self._include_heading = include_heading

    async def chunk(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        sections = self._split_sections(doc.content)
        chunks: list[Chunk] = []
        idx = 0

        for sec in sections:
            text = sec["text"].strip()
            if len(text) < self._min_size:
                continue
            sub_texts = self._split_long(text) if len(text) > self._max_size else [text]
            for sub in sub_texts:
                if not sub.strip():
                    continue
                chunks.append(Chunk(
                    text=sub.strip(),
                    chunk_index=idx,
                    doc_id=doc_id,
                    source=doc.source,
                    heading_path=sec.get("heading_path", ""),
                    metadata={
                        "heading_path": sec.get("heading_path", ""),
                        "format": doc.format,
                        "title": doc.title,
                    },
                ))
                idx += 1

        log.debug("structural_chunker.done", doc_id=doc_id, chunks=len(chunks))
        return chunks

    def _split_sections(self, content: str) -> list[dict]:
        lines = content.splitlines()
        sections: list[dict] = []
        current_headings: dict[int, str] = {}
        current_lines: list[str] = []
        heading_path = ""

        def flush():
            text = "\n".join(current_lines).strip()
            if text:
                prefix = f"{heading_path}\n\n" if self._include_heading and heading_path else ""
                sections.append({"heading_path": heading_path, "text": prefix + text})
            current_lines.clear()

        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                if level in self._levels:
                    flush()
                    current_headings[level] = title
                    for l in list(current_headings):
                        if l > level:
                            del current_headings[l]
                    heading_path = " > ".join(
                        current_headings[l] for l in sorted(current_headings)
                    )
                    if self._include_heading:
                        current_lines.append(line)
                    continue
            current_lines.append(line)

        flush()
        return sections

    def _split_long(self, text: str) -> list[str]:
        paragraphs = re.split(r'\n{2,}', text)
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= self._max_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)
        return chunks
