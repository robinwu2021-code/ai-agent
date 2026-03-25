"""rag/parsers/factory.py — 解析器工厂"""
from __future__ import annotations

from rag.config import ParserConfig
from rag.parsers.base import BaseParser


class ParserFactory:
    @staticmethod
    def create(config: ParserConfig) -> BaseParser:
        backend = config.backend.lower()

        if backend == "docling":
            from rag.parsers.docling_parser import DoclingParser
            c = config.docling
            return DoclingParser(
                do_ocr=c.do_ocr,
                do_table_structure=c.do_table_structure,
                table_mode=c.table_mode,
                ocr_lang=c.ocr_lang,
            )
        if backend == "unstructured":
            from rag.parsers.unstructured_parser import UnstructuredParser
            c = config.unstructured
            return UnstructuredParser(strategy=c.strategy, api_url=c.api_url, api_key=c.api_key)

        if backend == "markitdown":
            from rag.parsers.markitdown_parser import MarkItDownParser
            return MarkItDownParser(enable_plugins=config.markitdown.enable_plugins)

        raise ValueError(
            f"未知解析器 backend: {backend!r}，支持: docling | unstructured | markitdown"
        )
