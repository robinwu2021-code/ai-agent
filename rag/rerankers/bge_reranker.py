"""
rag/rerankers/bge_reranker.py — BGE Cross-Encoder 本地重排序器

依赖：pip install FlagEmbedding>=1.2.0
模型：BAAI/bge-reranker-v2-m3（多语言，支持中文）
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from rag.pipeline.query import RetrievedChunk

log = structlog.get_logger(__name__)


class BGEReranker:
    """
    本地 BGE Cross-Encoder 重排序器。

    使用 FlagEmbedding.FlagReranker 对候选 chunk 打分后重新排序。
    模型在首次调用时懒加载，避免启动时内存占用。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self._model_name,
                use_fp16=self._use_fp16,
            )
            log.info("bge_reranker.loaded", model=self._model_name)
        except ImportError:
            raise ImportError(
                "FlagEmbedding not installed. "
                "Run: pip install 'FlagEmbedding>=1.2.0'"
            )
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        对 chunks 进行 cross-encoder 重排序，返回 top_k 结果。
        """
        if not chunks:
            return chunks

        k = top_k or len(chunks)

        def _score_sync():
            model = self._get_model()
            pairs = [[query, c.text] for c in chunks]
            scores = model.compute_score(pairs, max_length=self._max_length)
            return scores if isinstance(scores, list) else [scores]

        try:
            scores = await asyncio.to_thread(_score_sync)
            ranked = sorted(
                zip(scores, chunks), key=lambda x: x[0], reverse=True
            )
            result = []
            for score, chunk in ranked[:k]:
                chunk.score = float(score)
                result.append(chunk)
            return result
        except Exception as exc:
            log.warning("bge_reranker.failed", error=str(exc))
            return chunks[:k]
