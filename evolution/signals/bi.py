"""
evolution/signals/bi.py — BI 查询信号采集器

监听 BiQueryEvent 和 FeedbackEvent，写入 evolution.db：
  1. 原始信号 → signals 表
  2. 门店画像更新 → bi_store_profile 表
  3. 成功查询候选 → 供 TemplateBuilderActor 处理
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import structlog

from evolution.bus import EventBus
from evolution.models import BiQueryEvent, FeedbackEvent
from evolution.signals.base import BaseSignalCollector
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.signals.bi")


class BiSignalCollector(BaseSignalCollector):

    def register(self) -> None:
        self._bus.subscribe("bi_query",  self._on_bi_query)
        self._bus.subscribe("feedback",  self._on_feedback)
        log.info("evolution.signals.bi.registered")

    async def _on_bi_query(self, event: BiQueryEvent) -> None:
        payload = {
            "user_text":    event.user_text,
            "bra_id":       event.bra_id,
            "mode":         event.mode,
            "api_payload":  event.api_payload,
            "result_rows":  event.result_rows,
            "no_data":      event.no_data,
            "error":        event.error,
            "latency_ms":   event.latency_ms,
            "retry_count":  event.retry_count,
        }

        # 基础质量分：成功有数据=1.0, 无数据=0.3, 有错误=0.0, 重试>1=0.5
        if event.error:
            quality = 0.0
        elif event.no_data:
            quality = 0.3
        elif event.retry_count > 1:
            quality = 0.5
        else:
            quality = 1.0

        await self._store.save_signal(
            signal_id   = event.event_id,
            signal_type = "bi_query",
            source_id   = event.event_id,
            payload     = payload,
            bra_id      = event.bra_id,
            session_id  = event.session_id,
            quality     = quality,
        )

        # 更新门店画像
        if event.bra_id:
            await self._update_store_profile(event)

        log.debug("evolution.signals.bi.recorded",
                  event_id=event.event_id,
                  no_data=event.no_data,
                  quality=quality)

    async def _update_store_profile(self, event: BiQueryEvent) -> None:
        """更新门店数据画像：记录无数据日期 或 最新有效数据时间戳。"""
        payload    = event.api_payload
        range_start = payload.get("rangeStart")

        if event.no_data and range_start:
            # 将无数据的日期记录到画像
            try:
                date_str = datetime.fromtimestamp(
                    range_start / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                await self._store.upsert_store_profile(
                    bra_id       = event.bra_id,
                    no_data_date = date_str,
                )
            except Exception as exc:
                log.warning("evolution.signals.bi.profile_update_failed", error=str(exc))

        elif event.result_rows > 0 and range_start:
            # 记录有效数据的时间范围
            try:
                await self._store.upsert_store_profile(
                    bra_id       = event.bra_id,
                    has_data_ts  = range_start / 1000,
                )
            except Exception as exc:
                log.warning("evolution.signals.bi.profile_update_failed", error=str(exc))

    async def _on_feedback(self, event: FeedbackEvent) -> None:
        if event.source != "bi":
            return
        # 找到对应的 bi_query 信号，更新其质量分
        if event.query_id:
            normalized = event.score / 5.0   # 标准化到 0-1
            await self._store.update_signal_quality(event.query_id, normalized)
            log.debug("evolution.signals.bi.feedback_recorded",
                      query_id=event.query_id, score=event.score)
