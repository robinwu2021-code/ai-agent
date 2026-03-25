"""rag/parsers/base.py — 文档解析基础协议与统一数据模型"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ParsedDocument:
    """解析后的文档统一表示，所有解析器输出此格式。"""
    content: str                              # 主文本（Markdown 格式）
    source: str                               # 原始文件路径
    format: str                               # pdf|docx|xlsx|pptx|html|md|csv|txt
    has_headings: bool = False                # 是否有标题结构（驱动分块策略选择）
    title: str = ""
    language: str = ""
    page_count: int = 0
    metadata: dict = field(default_factory=dict)
    tables: list[dict] = field(default_factory=list)    # [{markdown: "..."}]
    images: list[dict] = field(default_factory=list)    # [{description: "..."}]
    headings: list[dict] = field(default_factory=list)  # [{level, text}]


@runtime_checkable
class BaseParser(Protocol):
    """文档解析器协议——所有解析器须实现这两个方法。"""

    async def parse(self, file_path: str) -> ParsedDocument: ...

    def supports(self, file_path: str) -> bool: ...
