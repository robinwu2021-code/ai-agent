"""rag/chunkers/base.py — 分块器基础协议与数据模型"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag.parsers.base import ParsedDocument


@dataclass
class Chunk:
    """分块后的文本单元。"""
    text: str
    chunk_index: int
    doc_id: str
    source: str
    heading_path: str = ""              # 所属标题面包屑，如 "第一章 > 背景"
    metadata: dict = field(default_factory=dict)
    summary: str = ""                   # 原语言摘要（由 IngestionPipeline 生成，可选）


@runtime_checkable
class BaseChunker(Protocol):
    async def chunk(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]: ...
