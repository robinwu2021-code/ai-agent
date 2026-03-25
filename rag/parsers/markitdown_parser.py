"""
rag/parsers/markitdown_parser.py — MarkItDown 解析器（Microsoft 开源，轻量备选）

安装：pip install "markitdown[all]"
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from rag.parsers.base import ParsedDocument

log = structlog.get_logger(__name__)

_SUPPORTED = frozenset({
    ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm",
    ".md", ".txt", ".csv", ".jpg", ".jpeg", ".png",
})


class MarkItDownParser:
    def __init__(self, enable_plugins: bool = True) -> None:
        self._enable_plugins = enable_plugins
        self._md = None

    def _get_md(self):
        if self._md is not None:
            return self._md
        try:
            from markitdown import MarkItDown
            self._md = MarkItDown()
        except ImportError:
            log.warning("markitdown.not_installed", hint="pip install 'markitdown[all]'")
        return self._md

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in _SUPPORTED

    async def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        md = self._get_md()
        if md is None:
            return ParsedDocument(content="", source=str(path),
                                  format=path.suffix.lower().lstrip("."))
        try:
            result = await asyncio.to_thread(md.convert, str(path))
            content = result.text_content or ""
            headings = []
            for line in content.splitlines():
                s = line.lstrip()
                for lvl, prefix in [(1, "# "), (2, "## "), (3, "### ")]:
                    if s.startswith(prefix):
                        headings.append({"level": lvl, "text": s[len(prefix):].strip()})
                        break
            return ParsedDocument(
                content=content,
                source=str(path),
                format=path.suffix.lower().lstrip("."),
                has_headings=bool(headings),
                title=headings[0]["text"] if headings else path.stem,
                headings=headings,
                metadata={"parser": "markitdown"},
            )
        except Exception as exc:
            log.warning("markitdown.failed", path=str(path), error=str(exc))
            return ParsedDocument(content="", source=str(path),
                                  format=path.suffix.lower().lstrip("."))
