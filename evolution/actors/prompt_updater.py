"""
evolution/actors/prompt_updater.py — System Prompt 动态更新执行器

为每个门店生成数据可用性的上下文提示，写入 evolution_config 表。
下次构建 BI 系统提示时，由 server.py 读取并追加到 [BI Report Context] 中。

输出 key 格式：prompt_bi_store_ctx:{bra_id}
输出 value：字符串，例如：
  [Store Data Profile - B001]
  - 数据最早从 2024-01-15 开始
  - 2026-03-27 暂无数据
  - 历史上 store_delivery_amount 字段经常为空
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import structlog

from evolution.actors.base import BaseActor
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.actors.prompt_updater")


class PromptUpdaterActor(BaseActor):

    async def act(self) -> dict:
        # 获取所有门店的 profile
        all_profiles = await self._store._fetchall_by(
            "SELECT DISTINCT bra_id FROM bi_store_profile"
        )
        updated = 0

        for row in all_profiles:
            bra_id = row.get("bra_id")
            if not bra_id:
                continue

            profile = await self._store.get_store_profile(bra_id)
            ctx     = self._build_context(bra_id, profile)
            if ctx:
                await self._store.set_config(f"prompt_bi_store_ctx:{bra_id}", ctx)
                updated += 1

        summary = {"stores_updated": updated}
        log.info("evolution.prompt_updater.done", **summary)

        if updated:
            await self._store.log_action(
                action_id   = uuid.uuid4().hex[:16],
                actor_name  = "prompt_updater",
                target_type = "prompt",
                description = f"Updated store context prompts for {updated} stores",
            )
        return summary

    @staticmethod
    def _build_context(bra_id: str, profile: dict) -> str:
        if not profile:
            return ""

        base = profile.get("*") or {}
        lines = [f"[Store Data Profile - {bra_id}]"]

        # 数据起始时间
        first_ts = base.get("first_data_ts") or 0
        if first_ts:
            first_date = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            lines.append(f"- 数据最早从 {first_date} 开始")

        # 最新数据时间
        last_ts = base.get("last_data_ts") or 0
        if last_ts:
            last_date = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            lines.append(f"- 最新有数据记录至 {last_date}")

        # 无数据日期（最近 5 条）
        no_dates_raw = base.get("no_data_dates")
        if no_dates_raw:
            no_dates: list = json.loads(no_dates_raw) if isinstance(no_dates_raw, str) else no_dates_raw
            if no_dates:
                recent = sorted(no_dates)[-5:]
                lines.append(f"- 以下日期暂无数据: {', '.join(recent)}")

        # 无数据占比高的提示
        total_q  = base.get("total_queries") or 0
        no_data_c = base.get("no_data_count") or 0
        if total_q > 10 and no_data_c / total_q > 0.3:
            lines.append(
                f"- 注意：近期约 {int(no_data_c/total_q*100)}% 的查询无数据，"
                "请先确认查询时间段是否有效"
            )

        if len(lines) == 1:
            return ""
        return "\n".join(lines)
