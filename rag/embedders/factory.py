"""rag/embedders/factory.py — Embedder 工厂"""
from __future__ import annotations

from rag.config import EmbedderConfig
from rag.embedders.base import BaseEmbedder


class EmbedderFactory:
    @staticmethod
    def create(config: EmbedderConfig) -> BaseEmbedder:
        backend = config.backend.lower()

        if backend == "qwen":
            from rag.embedders.qwen_embedder import QwenEmbedder
            c = config.qwen
            return QwenEmbedder(
                model=c.model, dimensions=c.dimensions,
                batch_size=c.batch_size, api_key=c.api_key, base_url=c.base_url,
            )
        if backend == "openai":
            from rag.embedders.openai_embedder import OpenAIEmbedder
            c = config.openai
            return OpenAIEmbedder(
                model=c.model, dimensions=c.dimensions,
                batch_size=c.batch_size, api_key=c.api_key, base_url=c.base_url,
            )
        if backend == "bge_local":
            from rag.embedders.bge_local_embedder import BGELocalEmbedder
            c = config.bge_local
            return BGELocalEmbedder(model=c.model, device=c.device, batch_size=c.batch_size)

        raise ValueError(
            f"未知 embedder backend: {backend!r}，支持: qwen | openai | bge_local"
        )
