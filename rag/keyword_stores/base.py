"""rag/keyword_stores/base.py — 关键词检索基础协议"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class KeywordHit:
    chunk_id: str
    doc_id: str
    kb_id: str
    text: str
    score: float
    metadata: dict


@runtime_checkable
class BaseKeywordStore(Protocol):
    async def initialize(self) -> None: ...
    async def index_chunks(self, chunks: list[dict], kb_id: str) -> None: ...
    async def search(self, query: str, kb_id: str,
                     top_k: int = 20, doc_ids: list[str] | None = None) -> list[KeywordHit]: ...
    async def delete_by_doc_id(self, doc_id: str, kb_id: str) -> None: ...
    async def delete_by_kb_id(self, kb_id: str) -> None: ...
