"""
evolution/analyzers/bi_profiler.py — BI 门店数据画像分析器

每天从 bi_query 信号中，按门店汇总：
  - 无数据日期列表（前置校验缓存）
  - 数据存在的最早/最晚时间戳
  - 常见无数据指标

画像被 BiSignalCollector 实时更新（every query），
此 Analyzer 做**全量修正**（修正漏更新或过期数据）。
"""
from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import structlog

from evolution.analyzers.base import BaseAnalyzer
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.analyzers.bi_profiler")

_DAYS_TO_KEEP = 180   # 只保留近 180 天的无数据记录


class BiProfilerAnalyzer(BaseAnalyzer):

    async def analyze(self) -> dict:
        """汇总近 30 天的 bi_query 信号，全量重建各门店画像。"""
        signals = await self._store.recent_signals("bi_query", hours=30 * 24)

        # 按 bra_id 聚合
        by_store: dict[str, list[dict]] = defaultdict(list)
        for s in signals:
            try:
                payload = json.loads(s.get("payload") or "{}")
                bra_id  = s.get("bra_id") or payload.get("bra_id") or ""
                if bra_id:
                    by_store[bra_id].append(payload)
            except Exception:
                continue

        updated = 0
        for bra_id, payloads in by_store.items():
            await self._rebuild_profile(bra_id, payloads)
            updated += 1

        summary = {"stores_updated": updated, "signals_processed": len(signals)}
        log.info("evolution.bi_profiler.done", **summary)

        await self._store.log_action(
            action_id   = uuid.uuid4().hex[:16],
            actor_name  = "bi_profiler",
            target_type = "store_profile",
            description = f"Rebuilt profiles for {updated} stores",
        )
        return summary

    async def _rebuild_profile(self, bra_id: str, payloads: list[dict]) -> None:
        no_data_dates: set[str] = set()
        has_data_ts_list: list[float] = []

        now = time.time()
        cutoff = now - _DAYS_TO_KEEP * 86400

        for p in payloads:
            no_data     = p.get("no_data", False)
            api_payload = p.get("api_payload") or {}
            range_start = api_payload.get("rangeStart")

            if not range_start:
                continue

            ts_sec = range_start / 1000

            if no_data and ts_sec >= cutoff:
                try:
                    date_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d")
                    no_data_dates.add(date_str)
                except Exception:
                    pass
            elif not no_data:
                has_data_ts_list.append(ts_sec)

        # 写入画像（全量覆盖）
        first_ts = min(has_data_ts_list) if has_data_ts_list else 0
        last_ts  = max(has_data_ts_list) if has_data_ts_list else 0

        def _sync():
            conn = self._store._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO bi_store_profile
                   (bra_id, metric, no_data_dates, first_data_ts, last_data_ts,
                    total_queries, no_data_count, updated_at)
                   VALUES (?, '*', ?, ?, ?, ?, ?, ?)""",
                (
                    bra_id,
                    json.dumps(sorted(no_data_dates)),
                    first_ts,
                    last_ts,
                    len(payloads),
                    len(no_data_dates),
                    now,
                ),
            )
            conn.commit()

        import asyncio
        await asyncio.to_thread(_sync)
