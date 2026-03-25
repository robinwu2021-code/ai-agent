"""rag/chunkers/factory.py — 分块器工厂（SmartChunker 自动路由）"""
from __future__ import annotations

from rag.chunkers.base import BaseChunker, Chunk
from rag.config import ChunkerConfig
from rag.parsers.base import ParsedDocument


class ChunkerFactory:
    @staticmethod
    def create(config: ChunkerConfig, llm_fn=None) -> "SmartChunker":
        return SmartChunker(config=config, llm_fn=llm_fn)


class SmartChunker:
    """
    智能分块器，根据文档特征自动选择最优策略。

    auto 决策树：
      has_headings=True  → StructuralChunker（按标题层级，零成本）
      has_headings=False → SemanticChunker（余弦语义边界）
      SemanticChunker 失败 → SentenceChunker → FixedChunker

    叠加增强：可选 ContextualEnhancer（LLM 生成上下文描述 prepend 到 chunk 前面）
    """

    def __init__(self, config: ChunkerConfig, llm_fn=None) -> None:
        self._config = config
        self._llm_fn = llm_fn
        self._enhancer = None
        if config.contextual_enhancement.enabled and llm_fn:
            from rag.chunkers.contextual import ContextualEnhancer
            self._enhancer = ContextualEnhancer(
                llm_fn=llm_fn,
                prompt_template=config.contextual_enhancement.prompt,
                batch_size=config.contextual_enhancement.batch_size,
                use_prompt_cache=config.contextual_enhancement.use_prompt_cache,
            )

    def _build_inner(self, has_headings: bool) -> BaseChunker:
        strategy = self._config.strategy
        if strategy == "auto":
            strategy = "structural" if has_headings else "semantic"

        if strategy == "structural":
            from rag.chunkers.structural import StructuralChunker
            c = self._config.structural
            return StructuralChunker(
                min_size=c.min_size, max_size=c.max_size,
                levels=c.levels, include_heading=c.include_heading,
            )
        if strategy == "semantic":
            from rag.chunkers.semantic import SemanticChunker
            c = self._config.semantic
            return SemanticChunker(
                threshold=c.threshold, target_size=c.target_size, max_size=c.max_size,
            )
        if strategy == "sentence":
            from rag.chunkers.sentence import SentenceChunker
            c = self._config.sentence
            return SentenceChunker(chunk_size=c.chunk_size, chunk_overlap=c.chunk_overlap)
        if strategy == "fixed":
            from rag.chunkers.fixed import FixedChunker
            c = self._config.fixed
            return FixedChunker(chunk_size=c.chunk_size, chunk_overlap=c.chunk_overlap)

        raise ValueError(f"未知分块策略: {strategy!r}")

    async def chunk(self, doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        inner = self._build_inner(doc.has_headings)
        chunks = await inner.chunk(doc, doc_id)
        if self._enhancer and chunks:
            chunks = await self._enhancer.enhance(chunks, doc.content)
        return chunks
