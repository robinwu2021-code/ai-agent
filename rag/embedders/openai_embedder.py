"""
rag/embedders/openai_embedder.py — OpenAI / Ollama / vLLM 兼容 Embedding API

关键修复：
  - 传递 `dimensions` 参数给 API（Ollama ≥0.6 / OpenAI 支持 Matryoshka 截断）
  - 若 API 忽略 dimensions 参数仍返回全维向量，在客户端做截断兜底
  - 两种情况均保证输出维度严格等于 self._dimensions

兼容性说明：
  Ollama   ≥ 0.6.x  qwen3-embedding:8b 支持 dimensions 参数（Matryoshka）
  OpenAI           text-embedding-3-* 支持 dimensions 参数
  Azure OpenAI     同 OpenAI
  DashScope        text-embedding-v3  支持 dimensions 参数
  vLLM             取决于模型，不支持时 API 报错 → 回退到客户端截断
"""
from __future__ import annotations

import os
import structlog

log = structlog.get_logger(__name__)


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
        api_key: str = "",
        base_url: str = "",
    ) -> None:
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
            kw: dict = {}
            # api_key 必须非空，Ollama 接受任意非空值
            kw["api_key"] = self._api_key or "ollama"
            if self._base_url:
                kw["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kw)
        except ImportError:
            log.warning("openai_embedder.missing", hint="pip install openai")
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
            batch = texts[i : i + self._batch_size]
            embs = await self._embed_batch_once(client, batch, with_dimensions=True)
            all_embs.extend(embs)
        return all_embs

    async def _embed_batch_once(
        self,
        client,
        batch: list[str],
        *,
        with_dimensions: bool,
    ) -> list[list[float]]:
        """
        发送一次批量 Embedding 请求。

        with_dimensions=True 时传递 dimensions 参数（触发 Matryoshka 截断）。
        若 API 不支持该参数而报错，自动重试（不传 dimensions），
        再在客户端截断到目标维度。
        """
        kw: dict = {"model": self._model, "input": batch}
        if with_dimensions and self._dimensions > 0:
            kw["dimensions"] = self._dimensions

        try:
            resp = await client.embeddings.create(**kw)
            embs = [
                item.embedding
                for item in sorted(resp.data, key=lambda x: x.index)
            ]
            # 客户端截断兜底：若 API 忽略 dimensions 仍返回全维向量
            if self._dimensions > 0:
                embs = [e[: self._dimensions] for e in embs]
            return embs

        except Exception as exc:
            err_str = str(exc).lower()
            # API 不支持 dimensions 参数 → 重试不带该参数，客户端截断
            if with_dimensions and (
                "dimensions" in err_str
                or "unexpected" in err_str
                or "invalid" in err_str
                or "unsupported" in err_str
            ):
                log.warning(
                    "openai_embedder.dimensions_not_supported",
                    model=self._model,
                    base_url=self._base_url or "(sdk-default)",
                    hint="将在客户端截断向量到目标维度",
                )
                return await self._embed_batch_once(
                    client, batch, with_dimensions=False
                )

            log.warning(
                "openai_embedder.batch_failed",
                batch_start=0,
                size=len(batch),
                error=str(exc),
            )
            return [[] for _ in batch]
