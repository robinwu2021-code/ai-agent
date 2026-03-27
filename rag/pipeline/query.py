"""
rag/pipeline/query.py — 混合检索查询流水线

流程：
  1. 并发执行三路检索：向量检索 + 关键词(BM25)检索 + 图谱检索
  2. RRF（Reciprocal Rank Fusion）融合排名
  3. 可选 LLM / BGE / Cohere 重排序（Reranker）
  4. 返回 top-k 去重结果

RRF 公式：score(d) = Σ 1 / (k + rank(d, list_i))
  k=60 是 Cormack 2009 论文中的推荐值，实践中效果稳健。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    kb_id: str
    text: str
    score: float
    source: str = ""
    heading_path: str = ""
    metadata: dict = field(default_factory=dict)
    retrieval_sources: list[str] = field(default_factory=list)  # [vector, keyword, graph]


@dataclass
class QueryResult:
    query: str
    kb_id: str
    chunks: list[RetrievedChunk]
    context: str = ""           # 格式化后的 LLM 上下文字符串


class QueryPipeline:
    """
    混合检索查询流水线。

    向量、关键词、图谱三路并发检索 → RRF 融合 → 可选重排序。
    """

    def __init__(
        self,
        embedder,
        vector_store,
        keyword_store,
        graph_retriever=None,
        reranker=None,
        vector_top_k: int = 20,
        keyword_top_k: int = 20,
        graph_top_k: int = 10,
        fusion: str = "rrf",           # rrf | weighted
        rrf_k: int = 60,
        weights: dict | None = None,   # {"vector": 0.5, "keyword": 0.3, "graph": 0.2}
        enable_graph: bool = True,
        final_top_k: int = 5,
        use_summary_search: bool = False,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_store = keyword_store
        self._graph_retriever = graph_retriever
        self._reranker = reranker
        self._vec_k = vector_top_k
        self._kw_k = keyword_top_k
        self._graph_k = graph_top_k
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._weights = weights or {"vector": 0.5, "keyword": 0.3, "graph": 0.2}
        self._enable_graph = enable_graph and graph_retriever is not None
        self._final_k = final_top_k
        self._use_summary = use_summary_search

    async def query(
        self,
        query_text: str,
        kb_id: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        directory_ids: list[str] | None = None,
        # ── 场景 A：相似推荐去重 ──────────────────────────────────────────
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        # ── 场景 E：精确编号强制通道 ─────────────────────────────────────
        exact_terms: list[str] | None = None,
        exact_boost: float = 0.3,
    ) -> QueryResult:
        """
        执行混合检索并返回 QueryResult。

        Args:
            query_text:    用户查询文本
            kb_id:         知识库租户 ID
            top_k:         最终返回条数（默认使用初始化时的 final_top_k）
            doc_ids:       限定检索范围的文档 ID 列表
            directory_ids: 限定检索范围的目录 ID 列表（与 doc_ids 可同时使用）
            use_mmr:       [场景A] 启用 MMR 去重，提升相似推荐多样性
            mmr_lambda:    [场景A] MMR λ 参数，0.5=相关性/多样性平衡
            exact_terms:   [场景E] 精确匹配词列表，命中时额外加分（编号/标准号等）
            exact_boost:   [场景E] 精确命中额外加分值（0.0~1.0）
        """
        k = top_k or self._final_k

        # ① Embed 查询
        try:
            query_vec = await self._embedder.embed(query_text)
        except Exception as exc:
            log.warning("query.embed_failed", error=str(exc))
            query_vec = []

        # ② 三路并发检索
        tasks = [
            self._vector_search(query_vec, kb_id, doc_ids),
            self._keyword_search(query_text, kb_id, doc_ids),
        ]
        if self._enable_graph:
            tasks.append(self._graph_search(query_text, kb_id))
        else:
            tasks.append(self._empty_list())

        vec_hits, kw_hits, graph_hits = await asyncio.gather(*tasks, return_exceptions=True)
        vec_hits   = vec_hits   if isinstance(vec_hits, list)   else []
        kw_hits    = kw_hits    if isinstance(kw_hits, list)    else []
        graph_hits = graph_hits if isinstance(graph_hits, list) else []

        log.debug("query.retrieved",
                  vector=len(vec_hits), keyword=len(kw_hits), graph=len(graph_hits))

        # ③ 融合
        if self._fusion == "rrf":
            fused = self._rrf_fusion(vec_hits, kw_hits, graph_hits, k=k * 3)
        else:
            fused = self._weighted_fusion(vec_hits, kw_hits, graph_hits, k=k * 3)

        # ④ [场景E] 精确编号加分 → 在 rerank 前提权，避免被语义相似度压制
        if exact_terms:
            from rag.pipeline.advanced_query import boost_exact_match
            fused = boost_exact_match(fused, exact_terms, boost=exact_boost)

        # ⑤ 可选重排序
        if self._reranker and fused:
            try:
                fused = await self._reranker.rerank(query_text, fused, top_k=k)
            except Exception as exc:
                log.warning("query.rerank_failed", error=str(exc))
                fused = fused[:k]
        else:
            fused = fused[:k]

        # ⑥ [场景A] MMR 多样性去重（在 rerank 之后应用，保证基础质量）
        if use_mmr and fused:
            from rag.pipeline.advanced_query import apply_mmr
            fused = apply_mmr(fused, query_vec, top_k=k, lambda_=mmr_lambda)

        context = self._format_context(fused)
        return QueryResult(query=query_text, kb_id=kb_id, chunks=fused, context=context)

    # ── 各路检索 ──────────────────────────────────────────────────────

    async def _vector_search(
        self, query_vec: list[float], kb_id: str, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        if not query_vec:
            return []
        try:
            hits = await self._vector_store.hybrid_search(
                query_vec=query_vec,
                query_text="",
                kb_id=kb_id,
                top_k=self._vec_k,
                doc_ids=doc_ids,
                use_summary=self._use_summary,
            )
            return [
                RetrievedChunk(
                    chunk_id=p.get("chunk_id", ""),
                    doc_id=p.get("doc_id", ""),
                    kb_id=p.get("kb_id", kb_id),
                    text=p.get("text", ""),
                    score=score,
                    source=p.get("metadata", {}).get("source", ""),
                    heading_path=p.get("heading_path", ""),
                    metadata=p.get("metadata", {}),
                    retrieval_sources=["vector"],
                )
                for score, p in hits
            ]
        except Exception as exc:
            log.warning("query.vector_search_failed", error=str(exc))
            return []

    async def _keyword_search(
        self, query_text: str, kb_id: str, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        try:
            hits = await self._keyword_store.search(
                query=query_text, kb_id=kb_id,
                top_k=self._kw_k, doc_ids=doc_ids,
            )
            return [
                RetrievedChunk(
                    chunk_id=h.chunk_id, doc_id=h.doc_id, kb_id=h.kb_id,
                    text=h.text, score=h.score,
                    metadata=h.metadata, retrieval_sources=["keyword"],
                )
                for h in hits
            ]
        except Exception as exc:
            log.warning("query.keyword_search_failed", error=str(exc))
            return []

    async def _graph_search(self, query_text: str, kb_id: str) -> list[RetrievedChunk]:
        try:
            results = await self._graph_retriever.search(query_text, kb_id=kb_id, top_k=self._graph_k)
            return [
                RetrievedChunk(
                    chunk_id=r.get("chunk_id", ""),
                    doc_id=r.get("doc_id", ""),
                    kb_id=kb_id,
                    text=r.get("text", ""),
                    score=r.get("score", 0.5),
                    metadata=r,
                    retrieval_sources=["graph"],
                )
                for r in (results if isinstance(results, list) else [])
            ]
        except Exception as exc:
            log.warning("query.graph_search_failed", error=str(exc))
            return []

    # ── 融合算法 ──────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        vec_hits: list[RetrievedChunk],
        kw_hits: list[RetrievedChunk],
        graph_hits: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion — 不需要分数归一化，对跨检索器排名鲁棒。"""
        scores: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}

        def _add(hits: list[RetrievedChunk], source: str):
            for rank, chunk in enumerate(hits):
                cid = chunk.chunk_id or chunk.text[:40]
                if not cid:
                    continue
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                if cid not in best:
                    best[cid] = chunk
                else:
                    # 合并检索来源
                    existing = best[cid]
                    if source not in existing.retrieval_sources:
                        existing.retrieval_sources.append(source)

        _add(vec_hits, "vector")
        _add(kw_hits, "keyword")
        _add(graph_hits, "graph")

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        result = []
        for cid in sorted_ids:
            chunk = best[cid]
            chunk.score = scores[cid]
            result.append(chunk)
        return result

    def _weighted_fusion(
        self,
        vec_hits: list[RetrievedChunk],
        kw_hits: list[RetrievedChunk],
        graph_hits: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        """加权融合——分数归一化后乘以权重。"""
        def _norm(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
            if not hits:
                return hits
            mx = max(c.score for c in hits) or 1.0
            for c in hits:
                c.score = c.score / mx
            return hits

        scores: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}
        w = self._weights

        for hits, weight, src in [
            (_norm(vec_hits), w.get("vector", 0.5), "vector"),
            (_norm(kw_hits), w.get("keyword", 0.3), "keyword"),
            (_norm(graph_hits), w.get("graph", 0.2), "graph"),
        ]:
            for chunk in hits:
                cid = chunk.chunk_id or chunk.text[:40]
                if not cid:
                    continue
                scores[cid] = scores.get(cid, 0.0) + chunk.score * weight
                if cid not in best:
                    best[cid] = chunk
                elif src not in best[cid].retrieval_sources:
                    best[cid].retrieval_sources.append(src)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        result = []
        for cid in sorted_ids:
            chunk = best[cid]
            chunk.score = round(scores[cid], 6)
            result.append(chunk)
        return result

    # ── 上下文格式化 ──────────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""
        parts = ["## 相关知识库内容\n"]
        for i, c in enumerate(chunks, 1):
            src = c.metadata.get("source", "") or c.source or c.doc_id
            heading = f" | {c.heading_path}" if c.heading_path else ""
            parts.append(f"[{i}] 来源: {src}{heading}\n{c.text}\n")
        return "\n".join(parts)

    @staticmethod
    async def _empty_list():
        return []
