"""
rag/embedders/qwen_embedder.py — Qwen Embedding API（OpenAI 兼容接口）

适配现有项目的 Qwen embedding 方案，通过 OpenAI SDK 调用 DashScope。
"""
from __future__ import annotations

import os

import structlog

log = structlog.get_logger(__name__)


class QwenEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-v3",
        dimensions: int = 1024,
        batch_size: int = 32,
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key, base_url=self._base_url
            )
        except ImportError:
            log.warning("qwen_embedder.openai_missing", hint="pip install openai")
        return self._client

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        if client is None:
            return [[] for _ in texts]

        all_embs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            try:
                resp = await client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self._dimensions,
                )
                batch_embs = [
                    item.embedding
                    for item in sorted(resp.data, key=lambda x: x.index)
                ]
                all_embs.extend(batch_embs)
            except Exception as exc:
                log.warning("qwen_embedder.batch_failed",
                            batch_start=i, size=len(batch), error=str(exc))
                all_embs.extend([[] for _ in batch])
        return all_embs
