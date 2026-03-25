"""
rag/parsers/unstructured_parser.py — Unstructured 解析器（备选）

安装：pip install "unstructured[all-docs]"
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from rag.parsers.base import ParsedDocument

log = structlog.get_logger(__name__)

_SUPPORTED = frozenset({
    ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm",
    ".md", ".txt", ".eml", ".msg", ".rst",
})


class UnstructuredParser:
    def __init__(self, strategy: str = "hi_res",
                 api_url: str = "", api_key: str = "") -> None:
        self._strategy = strategy
        self._api_url = api_url
        self._api_key = api_key

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in _SUPPORTED

    async def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        try:
            elements = await asyncio.to_thread(self._partition, str(path))
            return self._to_doc(elements, path)
        except Exception as exc:
            log.warning("unstructured.failed", path=str(path), error=str(exc))
            return ParsedDocument(content="", source=str(path),
                                  format=path.suffix.lower().lstrip("."))

    def _partition(self, path: str):
        from unstructured.partition.auto import partition
        kwargs: dict = {"filename": path, "strategy": self._strategy}
        if self._api_url:
            kwargs["api_url"] = self._api_url
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return partition(**kwargs)

    def _to_doc(self, elements, path: Path) -> ParsedDocument:
        parts: list[str] = []
        headings: list[dict] = []
        tables: list[dict] = []

        try:
            from unstructured.documents.elements import Title, Table
            for el in elements:
                if isinstance(el, Title):
                    text = str(el)
                    parts.append(f"## {text}")
                    headings.append({"level": 2, "text": text})
                elif isinstance(el, Table):
                    html = getattr(getattr(el, "metadata", None), "text_as_html", str(el))
                    parts.append(html)
                    tables.append({"html": html})
                else:
                    parts.append(str(el))
        except ImportError:
            parts = [str(e) for e in elements]

        content = "\n\n".join(p for p in parts if p.strip())
        return ParsedDocument(
            content=content,
            source=str(path),
            format=path.suffix.lower().lstrip("."),
            has_headings=bool(headings),
            title=headings[0]["text"] if headings else path.stem,
            headings=headings,
            tables=tables,
            metadata={"parser": "unstructured", "strategy": self._strategy},
        )
