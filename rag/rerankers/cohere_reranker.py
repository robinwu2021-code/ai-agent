"""
rag/rerankers/cohere_reranker.py — Cohere Rerank API 重排序器

依赖：pip install cohere>=5.0.0
模型：rerank-multilingual-v3.0（支持中文）
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from rag.pipeline.query import RetrievedChunk

log = structlog.get_logger(__name__)


class CohereReranker:
    """
    Cohere Rerank API 重排序器（云端调用）。

    环境变量：COHERE_API_KEY
    模型默认：rerank-multilingual-v3.0
    """

    def __init__(
        self,
        model: str = "rerank-multilingual-v3.0",
        api_key: str | None = None,
        max_tokens_per_doc: int = 512,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self._max_tokens = max_tokens_per_doc
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import cohere
            self._client = cohere.AsyncClientV2(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "cohere not installed. Run: pip install 'cohere>=5.0.0'"
            )
        return self._client

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        调用 Cohere Rerank API 重新排序，返回 top_k 结果。
        """
        if not chunks:
            return chunks

        k = top_k or len(chunks)

        try:
            client = self._get_client()
            docs = [c.text for c in chunks]
            response = await client.rerank(
                model=self._model,
                query=query,
                documents=docs,
                top_n=k,
                max_tokens_per_doc=self._max_tokens,
            )
            result = []
            for item in response.results:
                chunk = chunks[item.index]
                chunk.score = item.relevance_score
                result.append(chunk)
            return result
        except Exception as exc:
            log.warning("cohere_reranker.failed", error=str(exc))
            return chunks[:k]
