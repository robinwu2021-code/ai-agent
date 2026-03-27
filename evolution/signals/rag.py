"""
evolution/signals/rag.py — RAG 检索信号采集器

监听 RagQueryEvent 和 FeedbackEvent，写入 evolution.db：
  1. 原始信号 → signals 表
  2. Chunk 命中统计 → kb_chunk_stats 表
  3. 反馈分 → 关联到对应 chunks
"""
from __future__ import annotations

import json
import re

import structlog

from evolution.bus import EventBus
from evolution.models import FeedbackEvent, RagQueryEvent
from evolution.signals.base import BaseSignalCollector
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.signals.rag")


class RagSignalCollector(BaseSignalCollector):

    def register(self) -> None:
        self._bus.subscribe("rag_query", self._on_rag_query)
        self._bus.subscribe("feedback",  self._on_feedback)
        log.info("evolution.signals.rag.registered")

    async def _on_rag_query(self, event: RagQueryEvent) -> None:
        # 判断哪些 chunk 被"引用"：answer_text 里是否包含 chunk text 的关键词
        used_ids = self._detect_used_chunks(event)

        payload = {
            "kb_id":              event.kb_id,
            "query_text":         event.query_text,
            "chunk_ids":          [c.get("chunk_id") for c in event.retrieved_chunks],
            "used_chunk_ids":     used_ids,
            "retrieval_strategy": event.retrieval_strategy,
            "latency_ms":         event.latency_ms,
        }

        quality = 1.0 if used_ids else 0.5   # 有引用=好，无引用=一般

        await self._store.save_signal(
            signal_id   = event.event_id,
            signal_type = "rag_query",
            source_id   = event.event_id,
            payload     = payload,
            kb_id       = event.kb_id,
            session_id  = event.session_id,
            quality     = quality,
        )

        # 更新 chunk 命中统计
        for chunk in event.retrieved_chunks:
            cid = chunk.get("chunk_id", "")
            if cid:
                used = cid in used_ids
                await self._store.record_chunk_hit(cid, event.kb_id, used=used)

        log.debug("evolution.signals.rag.recorded",
                  event_id=event.event_id,
                  chunks=len(event.retrieved_chunks),
                  used=len(used_ids))

    async def _on_feedback(self, event: FeedbackEvent) -> None:
        if event.source != "rag":
            return
        if not event.query_id:
            return

        # 找到原始信号中的 chunk_ids，将反馈分摊到每个 chunk
        signals = await self._store.recent_signals("rag_query", hours=72)
        for s in signals:
            if s.get("source_id") == event.query_id:
                payload    = json.loads(s.get("payload") or "{}")
                chunk_ids  = payload.get("chunk_ids") or []
                used_ids   = payload.get("used_chunk_ids") or []
                # 优先给被引用的 chunk 打分
                target_ids = used_ids if used_ids else chunk_ids
                if target_ids:
                    await self._store.record_chunk_feedback(target_ids, event.score)
                # 更新信号质量分
                await self._store.update_signal_quality(s["signal_id"], event.score / 5.0)
                break

    @staticmethod
    def _detect_used_chunks(event: RagQueryEvent) -> list[str]:
        """
        简单启发式：检测 answer_text 是否包含 chunk 的关键词（前 30 字）。
        更精确的方案是在生成时记录 LLM 引用了哪些 chunk，此处为降级实现。
        """
        if not event.answer_text or not event.retrieved_chunks:
            return []
        answer_lower = event.answer_text.lower()
        used = []
        for chunk in event.retrieved_chunks:
            cid  = chunk.get("chunk_id", "")
            text = chunk.get("text", "")[:50].lower()  # 取前 50 字做关键词
            if text and len(text) > 10:
                # 提取中文词或英文词（3字符以上）
                words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-z]{3,}', text)
                if words and any(w in answer_lower for w in words[:3]):
                    used.append(cid)
        return used
