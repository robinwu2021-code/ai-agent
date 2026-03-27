"""
evolution/analyzers/qa_extractor.py — 高质量 Q&A 对提取分析器

从 rag_query 信号中找出高质量交互（feedback ≥ 阈值 或 implicit_signal=accepted），
生成结构化 Q&A 对，写入 signals 表，供 KbInjectorActor 注入知识库。

Q&A 生成策略（不调用 LLM，纯规则提取）：
  - query_text  → Question
  - answer_text → Answer
  - 来源标注    → "[自动生成] 来自用户反馈 session_id"
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import structlog

from evolution.analyzers.base import BaseAnalyzer
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.analyzers.qa_extractor")

_MIN_ANSWER_LEN = 50   # 答案过短不值得注入


class QaExtractorAnalyzer(BaseAnalyzer):

    def __init__(self, store: EvolutionStore, config: dict | None = None) -> None:
        super().__init__(store, config)
        self._min_score = float(self._config.get("qa_min_feedback_score", 4.0))

    async def analyze(self) -> dict:
        """从近 7 天高质量信号中提取 Q&A 对。"""
        signals = await self._store.recent_signals("rag_query", hours=7 * 24)

        extracted = 0
        skipped   = 0

        for s in signals:
            # 质量过滤
            quality = s.get("quality", -1)
            if quality < 0:
                # 未有用户反馈，看隐式分（0.5=无引用，1.0=有引用）
                if quality == -1 or quality < 0.8:
                    skipped += 1
                    continue
            else:
                # 有用户反馈时，按反馈阈值过滤（quality 已归一化到 0-1）
                if quality < self._min_score / 5.0:
                    skipped += 1
                    continue

            payload  = json.loads(s.get("payload") or "{}")
            query    = payload.get("query_text", "").strip()
            answer   = payload.get("answer_text", "").strip()   # answer 需要从 signal 取

            if not query or not answer or len(answer) < _MIN_ANSWER_LEN:
                skipped += 1
                continue

            # 去重：相同 query 已经提取过则跳过
            existing = await self._store._fetchone(
                "SELECT signal_id FROM signals WHERE signal_type='qa_pair' AND "
                "json_extract(payload, '$.query') = ?",
                (query,),
            )
            if existing:
                skipped += 1
                continue

            qa_id = uuid.uuid4().hex[:16]
            qa_payload = {
                "query":      query,
                "answer":     answer,
                "kb_id":      s.get("kb_id", ""),
                "session_id": s.get("session_id", ""),
                "source_signal_id": s.get("signal_id"),
                "extracted_at": time.time(),
                "injected":   False,    # KbInjectorActor 更新此字段
            }

            await self._store.save_signal(
                signal_id   = qa_id,
                signal_type = "qa_pair",
                source_id   = s.get("signal_id", ""),
                payload     = qa_payload,
                kb_id       = s.get("kb_id", ""),
                session_id  = s.get("session_id", ""),
                quality     = quality,
            )
            extracted += 1

        summary = {
            "extracted":  extracted,
            "skipped":    skipped,
        }
        log.info("evolution.qa_extractor.done", **summary)

        if extracted:
            await self._store.log_action(
                action_id   = uuid.uuid4().hex[:16],
                actor_name  = "qa_extractor",
                target_type = "qa_pair",
                description = f"Extracted {extracted} Q&A pairs",
            )
        return summary
