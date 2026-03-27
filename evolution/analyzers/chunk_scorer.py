"""
evolution/analyzers/chunk_scorer.py — Chunk 质量评分分析器

每天跑一次，根据过去所有命中统计计算每个 chunk 的质量分，
写回 kb_chunk_stats.quality_score，供 KbInjectorActor 决定清理或保留。

质量分公式（0-1）：
  used_ratio    = used_count / max(hit_count, 1)         → 引用率
  feedback_avg  = feedback_sum / max(feedback_count, 1)  → 用户平均满意度（1-5, 归一化）
  recency_score = exp(-days_since_last_hit / 30)         → 近期活跃度

  score = 0.5×used_ratio + 0.3×(feedback_avg/5) + 0.2×recency_score
"""
from __future__ import annotations

import math
import time
import uuid

import structlog

from evolution.analyzers.base import BaseAnalyzer
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.analyzers.chunk_scorer")


class ChunkScorerAnalyzer(BaseAnalyzer):

    def __init__(self, store: EvolutionStore, config: dict | None = None) -> None:
        super().__init__(store, config)
        self._low_threshold = float(self._config.get("low_quality_threshold", 0.3))

    async def analyze(self) -> dict:
        """计算所有 kb 的 chunk 质量分。"""
        all_stats = await self._store._fetchall_by(
            "SELECT DISTINCT kb_id FROM kb_chunk_stats"
        )
        kb_ids = [r["kb_id"] for r in all_stats if r.get("kb_id")]

        total_scored = 0
        total_low    = 0

        for kb_id in kb_ids:
            stats = await self._store.get_chunk_stats(kb_id)
            now   = time.time()

            for stat in stats:
                score = self._compute_score(stat, now)
                await self._store.update_chunk_quality(stat["chunk_id"], score)
                total_scored += 1
                if score < self._low_threshold:
                    total_low += 1

        summary = {
            "kb_count":    len(kb_ids),
            "total_scored": total_scored,
            "low_quality":  total_low,
        }
        log.info("evolution.chunk_scorer.done", **summary)

        await self._store.log_action(
            action_id   = uuid.uuid4().hex[:16],
            actor_name  = "chunk_scorer",
            target_type = "chunk",
            description = f"Scored {total_scored} chunks, {total_low} low-quality",
        )
        return summary

    @staticmethod
    def _compute_score(stat: dict, now: float) -> float:
        hit_count      = max(stat.get("hit_count", 0), 1)
        used_count     = stat.get("used_count", 0)
        feedback_sum   = stat.get("feedback_sum", 0.0)
        feedback_count = max(stat.get("feedback_count", 0), 1)
        last_hit_at    = stat.get("last_hit_at") or now

        used_ratio    = used_count / hit_count
        feedback_avg  = feedback_sum / feedback_count   # 1-5 原始分
        days_idle     = (now - last_hit_at) / 86400
        recency_score = math.exp(-days_idle / 30.0)

        score = (
            0.5 * used_ratio
            + 0.3 * (feedback_avg / 5.0)
            + 0.2 * recency_score
        )
        return round(min(max(score, 0.0), 1.0), 4)


# 在 EvolutionStore 上补一个通用方法（避免修改 store.py 原文）
async def _fetchall_by(self, sql: str, args: tuple = ()) -> list[dict]:
    import asyncio
    return await asyncio.to_thread(self._fetchall, sql, args)

EvolutionStore._fetchall_by = _fetchall_by   # type: ignore[attr-defined]
