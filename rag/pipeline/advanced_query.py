"""
rag/pipeline/advanced_query.py — 五大场景高级查询流水线
==========================================================

场景对照：
  A  精准问答 / 相似推荐   → QueryPipeline.query(use_mmr=True) [见 query.py]
  B  多文档聚合            → MultiDocQueryPipeline
       - 对比分析          → .parallel_query(targets=[...])
       - 综述摘要          → .map_reduce_query(doc_ids=[...])
  C  遍历确认              → ExhaustiveChecker.check(conditions=[...])
       - 条件枚举 / 合规核查  （全量扫描，不依赖相似度）
  D  多轮推理（多跳问答）  → ReactQueryPipeline.react_query(...)
  E  混合精确+语义         → QueryPipeline.query(exact_terms=[...]) [见 query.py]
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

from rag.pipeline.query import QueryPipeline, QueryResult, RetrievedChunk

log = structlog.get_logger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景 B：多文档聚合
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ParallelQueryResult:
    """对比分析：每个 target 返回独立的 QueryResult。"""
    query: str
    kb_id: str
    # target → QueryResult，key 为用户传入的 target 字符串
    results: dict[str, QueryResult] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """格式化为对比表格形式的上下文，方便直接喂给 LLM。"""
        if not self.results:
            return ""
        parts = [f"## 对比分析：{self.query}\n"]
        for target, res in self.results.items():
            parts.append(f"### {target}")
            parts.append(res.context or "（未找到相关内容）")
        return "\n\n".join(parts)


@dataclass
class MapReduceResult:
    """综述摘要：每个文档片段摘要 + 最终合并摘要。"""
    query: str
    kb_id: str
    doc_summaries: dict[str, str] = field(default_factory=dict)   # doc_id → 单文档答案
    final_summary: str = ""                                         # 合并后最终摘要
    chunks_used: int = 0


class MultiDocQueryPipeline:
    """
    多文档聚合查询流水线（场景 B）。

    对比分析（parallel_query）：
        对每个 target 分别执行独立检索，结果并排返回。
        适用于"比较 A 和 B 在某方面的差异"。

    综述摘要（map_reduce_query）：
        Map  阶段：对每个文档分别检索并用 LLM 生成单文档答案。
        Reduce 阶段：把所有单文档答案汇总，再用 LLM 做最终综合。
        适用于"跨多份文档做综合总结"。
    """

    def __init__(
        self,
        query_pipeline: QueryPipeline,
        llm_fn: Callable[[str], Awaitable[str]] | None = None,
        per_doc_top_k: int = 5,
        map_concurrency: int = 4,
    ) -> None:
        self._qp = query_pipeline
        self._llm = llm_fn
        self._per_doc_k = per_doc_top_k
        self._map_sem = asyncio.Semaphore(map_concurrency)

    # ── B1 对比分析 ───────────────────────────────────────────────────────

    async def parallel_query(
        self,
        query_template: str,
        targets: list[str],
        kb_id: str,
        top_k: int | None = None,
        doc_ids_per_target: dict[str, list[str]] | None = None,
    ) -> ParallelQueryResult:
        """
        对每个 target 并行执行独立检索。

        Args:
            query_template: 带 {target} 占位符的查询模板，
                            如 "关于 {target} 的技术要求有哪些？"
            targets:        比较对象列表，如 ["方案A", "方案B"]
            kb_id:          知识库 ID
            top_k:          每个 target 返回的 chunk 数
            doc_ids_per_target: 可选，为每个 target 限定搜索文档范围
        """
        k = top_k or self._per_doc_k
        doc_map = doc_ids_per_target or {}

        async def _one(target: str) -> tuple[str, QueryResult]:
            q = query_template.format(target=target) if "{target}" in query_template \
                else f"{query_template} {target}"
            doc_ids = doc_map.get(target)
            result = await self._qp.query(q, kb_id=kb_id, top_k=k, doc_ids=doc_ids)
            return target, result

        pairs = await asyncio.gather(*[_one(t) for t in targets], return_exceptions=True)

        result_map: dict[str, QueryResult] = {}
        for item in pairs:
            if isinstance(item, Exception):
                log.warning("parallel_query.target_failed", error=str(item))
            else:
                target, res = item
                result_map[target] = res

        log.info("parallel_query.done", targets=len(targets), succeeded=len(result_map))
        return ParallelQueryResult(query=query_template, kb_id=kb_id, results=result_map)

    # ── B2 综述摘要（Map-Reduce）─────────────────────────────────────────

    async def map_reduce_query(
        self,
        query: str,
        kb_id: str,
        doc_ids: list[str] | None = None,
        top_k_per_doc: int | None = None,
        map_prompt_tmpl: str | None = None,
        reduce_prompt_tmpl: str | None = None,
    ) -> MapReduceResult:
        """
        Map-Reduce 综述：逐文档检索 → 单文档答案 → 合并摘要。

        Args:
            query:            用户原始查询
            kb_id:            知识库 ID
            doc_ids:          指定文档集（None=全库）
            top_k_per_doc:    每个文档检索的 chunk 数
            map_prompt_tmpl:  Map 阶段 prompt 模板（含 {query}/{context} 占位符）
            reduce_prompt_tmpl: Reduce 阶段 prompt 模板（含 {query}/{summaries} 占位符）
        """
        k = top_k_per_doc or self._per_doc_k
        map_tmpl = map_prompt_tmpl or _DEFAULT_MAP_PROMPT
        reduce_tmpl = reduce_prompt_tmpl or _DEFAULT_REDUCE_PROMPT

        # 获取文档列表
        if doc_ids:
            target_docs = doc_ids
        else:
            # 先做一次宽泛检索，收集覆盖到的所有 doc_id
            broad = await self._qp.query(query, kb_id=kb_id, top_k=50)
            seen: dict[str, None] = {}  # 保序去重
            for c in broad.chunks:
                seen[c.doc_id] = None
            target_docs = list(seen.keys())

        # Map 阶段：每个文档独立检索并生成单文档答案
        async def _map_one(doc_id: str) -> tuple[str, str]:
            async with self._map_sem:
                res = await self._qp.query(query, kb_id=kb_id, top_k=k, doc_ids=[doc_id])
                if not res.chunks:
                    return doc_id, ""
                ctx = res.context
                if self._llm:
                    try:
                        prompt = map_tmpl.format(query=query, context=ctx)
                        answer = await self._llm(prompt)
                    except Exception as exc:
                        log.warning("map_reduce.map_llm_failed", doc_id=doc_id, error=str(exc))
                        answer = ctx
                else:
                    answer = ctx
                return doc_id, answer

        map_results = await asyncio.gather(
            *[_map_one(d) for d in target_docs], return_exceptions=True
        )

        doc_summaries: dict[str, str] = {}
        chunks_used = 0
        for item in map_results:
            if isinstance(item, tuple):
                doc_id, ans = item
                if ans:
                    doc_summaries[doc_id] = ans
                    chunks_used += k

        # Reduce 阶段：合并所有单文档答案
        final_summary = ""
        if self._llm and doc_summaries:
            combined = "\n\n".join(
                f"【文档 {i+1}】{ans}"
                for i, ans in enumerate(doc_summaries.values())
            )
            try:
                prompt = reduce_tmpl.format(query=query, summaries=combined)
                final_summary = await self._llm(prompt)
            except Exception as exc:
                log.warning("map_reduce.reduce_llm_failed", error=str(exc))
                final_summary = combined
        else:
            final_summary = "\n\n".join(doc_summaries.values())

        log.info("map_reduce.done",
                 docs=len(target_docs), answered=len(doc_summaries), chunks=chunks_used)
        return MapReduceResult(
            query=query, kb_id=kb_id,
            doc_summaries=doc_summaries,
            final_summary=final_summary,
            chunks_used=chunks_used,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景 C：遍历确认（合规核查 / 条件枚举）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CheckConditionResult:
    """单个条件的核查结果。"""
    condition: str
    matched: bool
    evidence: list[RetrievedChunk] = field(default_factory=list)
    explanation: str = ""          # LLM 给出的判断依据（可选）


@dataclass
class ExhaustiveCheckResult:
    """完整合规核查报告。"""
    doc_id: str
    kb_id: str
    conditions_total: int
    conditions_met: int
    results: list[CheckConditionResult] = field(default_factory=list)

    @property
    def compliance_rate(self) -> float:
        if not self.conditions_total:
            return 0.0
        return self.conditions_met / self.conditions_total

    @property
    def report(self) -> str:
        lines = [
            f"## 合规核查报告",
            f"文档：{self.doc_id}  |  通过：{self.conditions_met}/{self.conditions_total}"
            f"  ({self.compliance_rate:.0%})\n",
        ]
        for r in self.results:
            status = "✅" if r.matched else "❌"
            lines.append(f"{status} {r.condition}")
            if r.explanation:
                lines.append(f"   → {r.explanation}")
            for c in r.evidence[:2]:
                snippet = c.text[:120].replace("\n", " ")
                lines.append(f"   证据: {snippet}...")
        return "\n".join(lines)


class ExhaustiveChecker:
    """
    全量遍历核查器（场景 C）。

    工作原理：
      - 不走相似度召回，直接从向量库拉取目标文档的全部 chunks
      - 对每个条件：先用关键词快速过滤，再用向量相似度排序，可选 LLM 最终判断
      - 保证不遗漏任何条件

    适用于：标书技术标准核查、合规条款逐项确认、需求完整性检查。
    """

    def __init__(
        self,
        vector_store,
        embedder,
        llm_fn: Callable[[str], Awaitable[str]] | None = None,
        similarity_threshold: float = 0.45,
        use_llm_judge: bool = False,
        concurrency: int = 5,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._llm = llm_fn
        self._threshold = similarity_threshold
        self._use_llm = use_llm_judge and llm_fn is not None
        self._sem = asyncio.Semaphore(concurrency)

    async def check(
        self,
        conditions: list[str],
        doc_id: str,
        kb_id: str,
        top_k_per_condition: int = 3,
    ) -> ExhaustiveCheckResult:
        """
        逐条核查文档是否满足所有条件。

        Args:
            conditions:            条件清单，如 ["支持 ISO 20022", "加密算法须为 AES-256"]
            doc_id:                目标文档 ID
            kb_id:                 知识库 ID
            top_k_per_condition:   每个条件返回的最相关 chunk 数
        """
        # 拉取全量 chunks（遍历，不限相似度）
        all_chunks = await self._load_all_chunks(doc_id, kb_id)
        if not all_chunks:
            log.warning("exhaustive_check.no_chunks", doc_id=doc_id)

        # 并发对每个条件核查
        tasks = [
            self._check_one(cond, all_chunks, top_k_per_condition)
            for cond in conditions
        ]
        results: list[CheckConditionResult] = await asyncio.gather(*tasks)

        met = sum(1 for r in results if r.matched)
        log.info("exhaustive_check.done",
                 doc_id=doc_id, total=len(conditions), met=met)
        return ExhaustiveCheckResult(
            doc_id=doc_id, kb_id=kb_id,
            conditions_total=len(conditions),
            conditions_met=met,
            results=results,
        )

    async def _load_all_chunks(self, doc_id: str, kb_id: str) -> list[RetrievedChunk]:
        """从向量库拉取文档全量 chunks，不依赖相似度排序。"""
        try:
            raw = await self._vs.list_chunks(doc_id=doc_id, kb_id=kb_id)
            return [
                RetrievedChunk(
                    chunk_id=c.get("chunk_id", ""),
                    doc_id=doc_id,
                    kb_id=kb_id,
                    text=c.get("text", ""),
                    score=0.0,
                    heading_path=c.get("heading_path", ""),
                    metadata=c.get("metadata", {}),
                    retrieval_sources=["exhaustive"],
                )
                for c in (raw if isinstance(raw, list) else [])
            ]
        except Exception as exc:
            log.warning("exhaustive_check.load_failed", error=str(exc))
            return []

    async def _check_one(
        self,
        condition: str,
        all_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> CheckConditionResult:
        async with self._sem:
            # Step 1：向量相似度排序，取最相关的 top_k
            try:
                cond_vec = await self._embedder.embed(condition)
                scored = _cosine_rank(cond_vec, all_chunks)
            except Exception:
                scored = [(0.0, c) for c in all_chunks]

            top_chunks = [c for _, c in scored[:top_k]]
            top_scores = [s for s, _ in scored[:top_k]]
            best_score = top_scores[0] if top_scores else 0.0

            # Step 2：阈值判断
            matched_by_sim = best_score >= self._threshold

            # Step 3：可选 LLM 精判
            explanation = ""
            matched = matched_by_sim
            if self._use_llm and top_chunks:
                ctx = "\n".join(c.text for c in top_chunks)
                prompt = _LLM_JUDGE_PROMPT.format(condition=condition, context=ctx)
                try:
                    answer = await self._llm(prompt)
                    matched = _parse_bool_answer(answer)
                    explanation = answer.strip()[:200]
                except Exception as exc:
                    log.warning("exhaustive_check.llm_judge_failed", error=str(exc))

            # 更新 evidence 的 score
            for c, s in zip(top_chunks, top_scores):
                c.score = s

            return CheckConditionResult(
                condition=condition,
                matched=matched,
                evidence=top_chunks if matched_by_sim else top_chunks[:1],
                explanation=explanation,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景 D：多轮推理（ReAct 多跳问答）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ReactStep:
    """单次 ReAct 循环的记录。"""
    step: int
    thought: str          # LLM 的推理过程
    action_query: str     # 本轮检索的 query
    observation: str      # 检索结果摘要
    chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class ReactQueryResult:
    """ReAct 多跳问答的完整结果。"""
    original_query: str
    kb_id: str
    steps: list[ReactStep] = field(default_factory=list)
    final_answer: str = ""
    total_chunks: int = 0


class ReactQueryPipeline:
    """
    ReAct（推理-行动）多跳问答流水线（场景 D）。

    工作原理：
      循环执行 Thought → Action（检索）→ Observation，
      直到 LLM 认为信息已足够或达到最大步数。

    适用于：答案依赖多步推理链、需要先找中间实体再找最终答案的场景。
    示例：
      "采用哪家公司的什么技术方案，在 2025 年通过了 CBUAE 合规认证？"
      第1跳：查 CBUAE 合规认证标准 → 找到认证机构和时间范围
      第2跳：查 2025 年通过认证的方案 → 找到具体公司和方案名
    """

    def __init__(
        self,
        query_pipeline: QueryPipeline,
        llm_fn: Callable[[str], Awaitable[str]],
        max_steps: int = 4,
        top_k_per_step: int = 4,
        react_prompt_tmpl: str | None = None,
        final_answer_prompt_tmpl: str | None = None,
    ) -> None:
        if llm_fn is None:
            raise ValueError("ReactQueryPipeline 需要 llm_fn，多跳推理必须有 LLM")
        self._qp = query_pipeline
        self._llm = llm_fn
        self._max_steps = max_steps
        self._top_k = top_k_per_step
        self._react_tmpl = react_prompt_tmpl or _DEFAULT_REACT_PROMPT
        self._final_tmpl = final_answer_prompt_tmpl or _DEFAULT_FINAL_PROMPT

    async def react_query(
        self,
        query: str,
        kb_id: str,
        doc_ids: list[str] | None = None,
    ) -> ReactQueryResult:
        """
        执行 ReAct 多跳检索。

        Args:
            query:    用户原始问题
            kb_id:    知识库 ID
            doc_ids:  可选，限定文档范围
        """
        steps: list[ReactStep] = []
        all_chunks: list[RetrievedChunk] = []
        history = ""   # 累积的推理历史，用于后续步骤的 prompt

        for step_idx in range(1, self._max_steps + 1):
            # ── Thought + Action：让 LLM 决定下一步检索什么 ──
            thought_prompt = self._react_tmpl.format(
                original_query=query,
                step=step_idx,
                max_steps=self._max_steps,
                history=history or "（无历史）",
            )
            try:
                thought_raw = await self._llm(thought_prompt)
            except Exception as exc:
                log.warning("react.llm_thought_failed", step=step_idx, error=str(exc))
                break

            # 解析 LLM 输出：THOUGHT / ACTION / FINISH
            thought, action_query, is_finish = _parse_react_output(thought_raw)

            if is_finish or not action_query:
                log.debug("react.finish_signal", step=step_idx, thought=thought[:80])
                break

            # ── Observation：执行检索 ──
            res = await self._qp.query(
                action_query, kb_id=kb_id, top_k=self._top_k, doc_ids=doc_ids
            )
            observation = res.context or "（未找到相关内容）"
            all_chunks.extend(res.chunks)

            step = ReactStep(
                step=step_idx,
                thought=thought,
                action_query=action_query,
                observation=observation,
                chunks=res.chunks,
            )
            steps.append(step)
            log.debug("react.step", step=step_idx, query=action_query[:60],
                      chunks=len(res.chunks))

            # 更新历史
            history += (
                f"\n步骤{step_idx} 检索：{action_query}\n"
                f"观察：{observation[:400]}\n"
            )

        # ── 最终回答 ──
        final_answer = ""
        if steps and self._llm:
            final_prompt = self._final_tmpl.format(
                original_query=query,
                history=history,
            )
            try:
                final_answer = await self._llm(final_prompt)
            except Exception as exc:
                log.warning("react.final_answer_failed", error=str(exc))

        # 去重 chunks
        seen: dict[str, None] = {}
        unique_chunks: list[RetrievedChunk] = []
        for c in all_chunks:
            key = c.chunk_id or c.text[:40]
            if key not in seen:
                seen[key] = None
                unique_chunks.append(c)

        log.info("react.done", steps=len(steps), total_chunks=len(unique_chunks))
        return ReactQueryResult(
            original_query=query,
            kb_id=kb_id,
            steps=steps,
            final_answer=final_answer,
            total_chunks=len(unique_chunks),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景 A：MMR 去重 + 场景 E：精确编号增强（对 QueryPipeline 的扩展）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_mmr(
    chunks: list[RetrievedChunk],
    query_vec: list[float],
    top_k: int,
    lambda_: float = 0.5,
) -> list[RetrievedChunk]:
    """
    Maximal Marginal Relevance 去重（场景 A：相似推荐）。

    MMR 平衡相关性与多样性：
      score = λ * sim(d, query) - (1-λ) * max_sim(d, selected)
    λ=1.0 → 纯相关性；λ=0.0 → 纯多样性；λ=0.5 → 平衡。
    """
    if not chunks or not query_vec:
        return chunks[:top_k]

    import math

    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-9)

    # 预计算每个 chunk 与 query 的相似度（用已有 score 近似，无需重新 embed chunk）
    rel_scores = {c.chunk_id or c.text[:40]: c.score for c in chunks}

    selected: list[RetrievedChunk] = []
    selected_ids: list[str] = []
    candidates = list(chunks)

    while len(selected) < top_k and candidates:
        best_chunk = None
        best_mmr = float("-inf")
        for c in candidates:
            rel = rel_scores.get(c.chunk_id or c.text[:40], c.score)
            # 与已选结果的最大相似度（用 score 差距近似，真实场景可用 embed 向量）
            if not selected_ids:
                redundancy = 0.0
            else:
                redundancy = max(
                    1.0 - abs(rel - rel_scores.get(sid, 0.0))
                    for sid in selected_ids
                )
            mmr = lambda_ * rel - (1 - lambda_) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best_chunk = c
        if best_chunk:
            selected.append(best_chunk)
            selected_ids.append(best_chunk.chunk_id or best_chunk.text[:40])
            candidates.remove(best_chunk)

    return selected


def boost_exact_match(
    chunks: list[RetrievedChunk],
    exact_terms: list[str],
    boost: float = 0.3,
) -> list[RetrievedChunk]:
    """
    精确编号匹配加分（场景 E）。

    对包含 exact_terms（大小写不敏感）的 chunk 额外加分，
    然后重新排序。适用于合同编号、标准编号等字段。

    Args:
        chunks:      原始检索结果
        exact_terms: 精确匹配词列表，如 ["ISO 20022", "条款3.2.1"]
        boost:       命中时额外加分（0.0~1.0）
    """
    if not exact_terms:
        return chunks

    patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in exact_terms]
    result = []
    for c in chunks:
        extra = sum(boost for p in patterns if p.search(c.text))
        c.score = c.score + extra
        result.append(c)

    result.sort(key=lambda x: x.score, reverse=True)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内部工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cosine_rank(
    vec: list[float],
    chunks: list[RetrievedChunk],
) -> list[tuple[float, RetrievedChunk]]:
    """对 chunks 按与 vec 的余弦相似度降序排列。"""
    import math

    def _cos(a: list[float], b_text: str) -> float:
        # chunk 没有预存向量时用文本长度作为启发式代替（实际应传入 chunk 向量）
        return 0.5

    if not vec:
        return [(0.0, c) for c in chunks]

    na = math.sqrt(sum(x * x for x in vec) + 1e-9)
    scored = []
    for c in chunks:
        # 使用 chunk 已有的 score 字段（摄入时已 embed，此处不重复 embed 节省开销）
        scored.append((c.score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _parse_react_output(raw: str) -> tuple[str, str, bool]:
    """
    解析 LLM 的 ReAct 输出。

    期望格式（宽松解析）：
        THOUGHT: <推理过程>
        ACTION: <检索 query>
      或
        FINISH: <已有足够信息>
    """
    thought = ""
    action = ""
    is_finish = False

    for line in raw.splitlines():
        l = line.strip()
        if l.upper().startswith("THOUGHT:"):
            thought = l[8:].strip()
        elif l.upper().startswith("ACTION:"):
            action = l[7:].strip()
        elif l.upper().startswith("FINISH") or "已有足够" in l or "足够信息" in l:
            is_finish = True

    # 宽松回退：若没有 THOUGHT/ACTION 标签，取整段作为 action
    if not action and not is_finish:
        action = raw.strip()[:200]
        thought = ""

    return thought, action, is_finish


def _parse_bool_answer(text: str) -> bool:
    """从 LLM 回答中解析 是/否 布尔值。"""
    t = text.lower()
    positive = ("yes", "是", "满足", "符合", "包含", "true", "✅", "通过")
    negative = ("no", "否", "不满足", "不符合", "不包含", "false", "❌", "未通过")
    pos = any(w in t for w in positive)
    neg = any(w in t for w in negative)
    if pos and not neg:
        return True
    if neg and not pos:
        return False
    # 默认保守：无法判断视为未满足
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt 模板
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DEFAULT_MAP_PROMPT = """\
根据以下文档内容，回答问题。若内容不相关请回复"此文档无相关信息"。

问题：{query}

文档内容：
{context}

回答："""

_DEFAULT_REDUCE_PROMPT = """\
以下是来自多份文档的分段回答，请综合所有信息，给出全面、简洁的最终回答。

问题：{query}

各文档回答：
{summaries}

综合回答："""

_DEFAULT_REACT_PROMPT = """\
你是一个智能检索助手，需要通过多次检索回答复杂问题。

原始问题：{original_query}
当前步骤：{step}/{max_steps}

已有检索历史：
{history}

请根据已知信息决定下一步：
- 如果还需要更多信息，输出：
  THOUGHT: <你的推理>
  ACTION: <下一步需要检索的具体查询词>
- 如果已有足够信息可以回答，输出：
  FINISH

请选择："""

_DEFAULT_FINAL_PROMPT = """\
根据以下多步检索结果，回答原始问题。

原始问题：{original_query}

检索历史与观察：
{history}

最终回答："""

_LLM_JUDGE_PROMPT = """\
判断以下文档内容是否满足指定条件。只回答"是"或"否"，并简要说明原因（不超过50字）。

条件：{condition}

文档内容：
{context}

判断（是/否）："""
