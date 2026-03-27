"""
evolution/actors/template_builder.py — BI 查询模板构建执行器

从近期成功的 bi_query 信号（quality=1.0，result_rows>0）中，
提炼查询模板，写入 bi_templates 表。

模板去重策略（无 embedding，纯规则）：
  key = frozenset(metrics) + mode + aggregation_or_none
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid

import structlog

from evolution.actors.base import BaseActor
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.actors.template_builder")


class TemplateBuilderActor(BaseActor):

    async def act(self) -> dict:
        signals = await self._store.recent_signals("bi_query", hours=30 * 24)

        built   = 0
        updated = 0

        for s in signals:
            payload = json.loads(s.get("payload") or "{}")
            quality = s.get("quality", 0)

            # 只处理成功查询
            if quality < 0.9 or payload.get("no_data") or payload.get("error"):
                continue

            api_payload  = payload.get("api_payload") or {}
            user_text    = payload.get("user_text", "").strip()
            mode         = payload.get("mode", "default")
            result_rows  = payload.get("result_rows", 0)

            if result_rows == 0 or not user_text:
                continue

            # 生成模板 key（去重用）
            canonical = self._canonicalize(api_payload, mode)
            key       = self._make_key(canonical)

            existing = await self._store._fetchone(
                "SELECT template_id, hit_count FROM bi_templates WHERE template_id=?",
                (key,),
            )

            if existing is None:
                await self._store.upsert_bi_template(
                    template_id      = key,
                    intent_text      = user_text[:200],
                    canonical_params = canonical,
                    bra_id           = payload.get("bra_id", ""),
                )
                built += 1
            else:
                await self._store.upsert_bi_template(
                    template_id      = key,
                    intent_text      = user_text[:200],
                    canonical_params = canonical,
                    bra_id           = payload.get("bra_id", ""),
                )
                updated += 1

        summary = {"built": built, "updated": updated, "signals": len(signals)}
        log.info("evolution.template_builder.done", **summary)

        if built:
            await self._store.log_action(
                action_id   = uuid.uuid4().hex[:16],
                actor_name  = "template_builder",
                target_type = "bi_template",
                description = f"Built {built} new templates, updated {updated}",
            )
        return summary

    @staticmethod
    def _canonicalize(api_payload: dict, mode: str) -> dict:
        """提取可复用的参数骨架（去掉时间戳，保留结构）。"""
        canonical: dict = {"mode": mode}
        if "metrics" in api_payload:
            canonical["metrics"]     = sorted(api_payload["metrics"])
        if "aggregation" in api_payload:
            canonical["aggregation"] = api_payload["aggregation"]
        if "dimensions" in api_payload:
            canonical["dimensions"]  = api_payload["dimensions"]
        if "timeGranularity" in api_payload:
            canonical["timeGranularity"] = api_payload["timeGranularity"]
        return canonical

    @staticmethod
    def _make_key(canonical: dict) -> str:
        raw = json.dumps(canonical, sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]
