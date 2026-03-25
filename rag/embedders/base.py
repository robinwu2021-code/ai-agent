"""rag/embedders/base.py — Embedding 生成器协议"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseEmbedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...
