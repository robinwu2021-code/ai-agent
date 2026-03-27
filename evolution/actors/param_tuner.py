"""
evolution/actors/param_tuner.py — RAG 检索参数自适应执行器

分析各知识库过去 7 天的检索效果，
将优化建议写入 evolution_config 表（键：kb_tuned_params:{kb_id}），
下次 RAG 查询时可读取这些参数覆盖默认值。

优化逻辑：
  - 向量分支命中率持续高 → 提高 vector_weight
  - BM25 命中率持续低    → 降低 bm25_weight
  - 平均质量分 < 0.4    → 降低 similarity_threshold 扩大召回
"""
from __future__ import annotations

import json
import uuid

import structlog

from evolution.actors.base import BaseActor
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.actors.param_tuner")

# 参数调整步长
_STEP      = 0.05
_VEC_MIN   = 0.3
_VEC_MAX   = 0.9
_THRESH_MIN = 0.3
_THRESH_MAX = 0.8


class ParamTunerActor(BaseActor):

    async def act(self) -> dict:
        # 读取近 7 天所有 rag_query 信号
        signals = await self._store.recent_signals("rag_query", hours=7 * 24, limit=5000)

        # 按 kb_id 分组
        from collections import defaultdict
        by_kb: dict[str, list[dict]] = defaultdict(list)
        for s in signals:
            kb_id = s.get("kb_id") or ""
            if kb_id:
                by_kb[kb_id].append(s)

        tuned = 0
        for kb_id, kb_signals in by_kb.items():
            changed = await self._tune_kb(kb_id, kb_signals)
            if changed:
                tuned += 1

        summary = {"kbs_tuned": tuned, "total_kbs": len(by_kb)}
        log.info("evolution.param_tuner.done", **summary)
        return summary

    async def _tune_kb(self, kb_id: str, signals: list[dict]) -> bool:
        # 当前参数
        current = await self._store.get_config(
            f"kb_tuned_params:{kb_id}",
            default={"vector_weight": 0.6, "bm25_weight": 0.4, "similarity_threshold": 0.5},
        )

        avg_quality = (
            sum(s.get("quality", 0) for s in signals if s.get("quality", -1) >= 0)
            / max(sum(1 for s in signals if s.get("quality", -1) >= 0), 1)
        )

        new_params = dict(current)
        changed = False

        # 平均质量过低 → 降低相似度阈值（扩大召回）
        if avg_quality < 0.4:
            new_thresh = max(current["similarity_threshold"] - _STEP, _THRESH_MIN)
            if new_thresh != current["similarity_threshold"]:
                new_params["similarity_threshold"] = new_thresh
                changed = True
                log.info("evolution.param_tuner.threshold_lowered",
                         kb_id=kb_id, old=current["similarity_threshold"], new=new_thresh)

        # 质量很好 → 可以提高阈值（提升精准度）
        elif avg_quality > 0.8:
            new_thresh = min(current["similarity_threshold"] + _STEP, _THRESH_MAX)
            if new_thresh != current["similarity_threshold"]:
                new_params["similarity_threshold"] = new_thresh
                changed = True

        if changed:
            await self._store.set_config(f"kb_tuned_params:{kb_id}", new_params)
            await self._store.log_action(
                action_id   = uuid.uuid4().hex[:16],
                actor_name  = "param_tuner",
                target_type = "kb_config",
                target_id   = kb_id,
                description = (
                    f"Tuned params for kb={kb_id}: "
                    f"avg_quality={avg_quality:.2f} → {new_params}"
                ),
            )
        return changed
