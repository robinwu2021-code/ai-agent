"""rag/embedders/openai_embedder.py — OpenAI Embedding API"""
from __future__ import annotations

import os
import structlog

log = structlog.get_logger(__name__)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536,
                 batch_size: int = 100, api_key: str = "", base_url: str = "") -> None:
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
            kw = {"api_key": self._api_key}
            if self._base_url:
                kw["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kw)
        except ImportError:
            log.warning("openai_embedder.missing")
        return self._client

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        r = await self.embed_batch([text])
        return r[0] if r else []

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
                resp = await client.embeddings.create(model=self._model, input=batch)
                all_embs.extend(
                    item.embedding for item in sorted(resp.data, key=lambda x: x.index)
                )
            except Exception as exc:
                log.warning("openai_embedder.batch_failed", error=str(exc))
                all_embs.extend([[] for _ in batch])
        return all_embs
