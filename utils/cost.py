"""
utils/cost.py — 成本控制与 Token 配额管理

功能：
  - CostTracker      实时 Token 用量追踪 + 费用估算
  - QuotaManager     每用户/每组织配额管理
  - ModelDowngrader  成本超限时自动降级到更便宜的模型
  - CostReport       用量报告生成
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Model Pricing（每 1M token 美元，2025 年参考价）
# ─────────────────────────────────────────────

MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6":              {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-20250514":     {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001":    {"input": 0.25,  "output": 1.25},
    "gpt-4o":                       {"input": 2.5,   "output": 10.0},
    "gpt-4o-mini":                  {"input": 0.15,  "output": 0.6},
    "gpt-3.5-turbo":                {"input": 0.5,   "output": 1.5},
}

# 降级链：当成本超限时依次尝试
MODEL_DOWNGRADE_CHAIN: dict[str, list[str]] = {
    "claude-opus-4-6":           ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
    "claude-sonnet-4-20250514":  ["claude-haiku-4-5-20251001"],
    "gpt-4o":                    ["gpt-4o-mini", "gpt-3.5-turbo"],
}


# ─────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────

@dataclass
class UsageRecord:
    user_id:          str
    session_id:       str
    model:            str
    prompt_tokens:    int
    completion_tokens: int
    cost_usd:         float
    timestamp:        str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    task_id:          str = ""


@dataclass
class QuotaConfig:
    """用户/组织的配额配置。"""
    daily_token_limit:   int   = 500_000    # 每日 token 上限
    daily_cost_limit:    float = 5.0        # 每日费用上限（美元）
    monthly_cost_limit:  float = 50.0       # 每月费用上限
    max_tokens_per_task: int   = 50_000     # 单任务 token 上限
    allow_downgrade:     bool  = True       # 是否允许自动降级模型


# ─────────────────────────────────────────────
# Cost Tracker
# ─────────────────────────────────────────────

class CostTracker:
    """追踪 Token 用量和费用，支持按用户/日/月汇总。"""

    def __init__(self) -> None:
        self._records:     list[UsageRecord]               = []
        self._daily_cache: dict[tuple, dict[str, float]]   = {}

    def record(
        self,
        user_id: str,
        session_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        task_id: str = "",
    ) -> UsageRecord:
        cost = self._calc_cost(model, prompt_tokens, completion_tokens)
        rec  = UsageRecord(
            user_id=user_id, session_id=session_id, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost_usd=cost, task_id=task_id,
        )
        self._records.append(rec)
        self._daily_cache.clear()   # 清缓存
        log.debug(
            "cost.recorded",
            user=user_id, model=model,
            tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
        )
        return rec

    def get_daily_usage(self, user_id: str, day: date | None = None) -> dict[str, float]:
        d = day or date.today()
        cache_key = (user_id, str(d))
        if cache_key in self._daily_cache:
            return self._daily_cache[cache_key]

        result: dict[str, float] = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0,
        }
        for rec in self._records:
            if rec.user_id != user_id:
                continue
            if rec.timestamp[:10] != str(d):
                continue
            result["prompt_tokens"]     += rec.prompt_tokens
            result["completion_tokens"] += rec.completion_tokens
            result["total_tokens"]      += rec.prompt_tokens + rec.completion_tokens
            result["cost_usd"]          += rec.cost_usd

        self._daily_cache[cache_key] = result
        return result

    def get_monthly_cost(self, user_id: str, year: int, month: int) -> float:
        prefix = f"{year:04d}-{month:02d}"
        return sum(
            r.cost_usd for r in self._records
            if r.user_id == user_id and r.timestamp.startswith(prefix)
        )

    def get_top_users(self, limit: int = 10) -> list[dict]:
        totals: dict[str, float] = defaultdict(float)
        for rec in self._records:
            totals[rec.user_id] += rec.cost_usd
        return [
            {"user_id": uid, "total_cost_usd": round(cost, 4)}
            for uid, cost in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]

    @staticmethod
    def _calc_cost(model: str, prompt: int, completion: int) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 2.0})
        return (prompt * pricing["input"] + completion * pricing["output"]) / 1_000_000


# ─────────────────────────────────────────────
# Quota Manager
# ─────────────────────────────────────────────

class QuotaManager:
    """每用户/每组织配额检查。"""

    def __init__(self, tracker: CostTracker) -> None:
        self._tracker = tracker
        self._configs: dict[str, QuotaConfig] = {}
        self._default  = QuotaConfig()

    def set_quota(self, user_id: str, config: QuotaConfig) -> None:
        self._configs[user_id] = config
        log.info("quota.set", user=user_id,
                 daily_tokens=config.daily_token_limit,
                 daily_cost=config.daily_cost_limit)

    def get_quota(self, user_id: str) -> QuotaConfig:
        return self._configs.get(user_id, self._default)

    def check_daily_quota(self, user_id: str) -> tuple[bool, str]:
        """返回 (within_quota, reason)。"""
        quota = self.get_quota(user_id)
        usage = self._tracker.get_daily_usage(user_id)

        if usage["total_tokens"] >= quota.daily_token_limit:
            return False, (
                f"Daily token quota exceeded "
                f"({usage['total_tokens']:,} / {quota.daily_token_limit:,})"
            )
        if usage["cost_usd"] >= quota.daily_cost_limit:
            return False, (
                f"Daily cost quota exceeded "
                f"(${usage['cost_usd']:.2f} / ${quota.daily_cost_limit:.2f})"
            )
        return True, ""

    def check_monthly_quota(self, user_id: str) -> tuple[bool, str]:
        quota  = self.get_quota(user_id)
        now    = datetime.now(timezone.utc)
        cost   = self._tracker.get_monthly_cost(user_id, now.year, now.month)
        if cost >= quota.monthly_cost_limit:
            return False, f"Monthly cost limit exceeded (${cost:.2f} / ${quota.monthly_cost_limit:.2f})"
        return True, ""

    def remaining_daily_tokens(self, user_id: str) -> int:
        quota = self.get_quota(user_id)
        usage = self._tracker.get_daily_usage(user_id)
        return max(0, quota.daily_token_limit - int(usage["total_tokens"]))


# ─────────────────────────────────────────────
# Model Downgrader
# ─────────────────────────────────────────────

class ModelDowngrader:
    """
    当用户配额接近上限时，自动降级到更便宜的模型。
    集成点：在 LLMEngine.chat() 调用前由 AgentConfig 注入。
    """

    def __init__(
        self,
        quota_manager: QuotaManager,
        downgrade_threshold: float = 0.8,   # 配额使用到 80% 时触发降级
    ) -> None:
        self._quota     = quota_manager
        self._threshold = downgrade_threshold

    def suggest_model(self, user_id: str, requested_model: str) -> str:
        """
        根据当前用量建议合适的模型。
        如果配额充足，返回原始请求的模型；否则返回更便宜的替代。
        """
        quota = self._quota.get_quota(user_id)
        if not quota.allow_downgrade:
            return requested_model

        usage = self._quota._tracker.get_daily_usage(user_id)
        token_ratio = usage["total_tokens"] / max(quota.daily_token_limit, 1)
        cost_ratio  = usage["cost_usd"]     / max(quota.daily_cost_limit,  0.01)
        ratio       = max(token_ratio, cost_ratio)

        if ratio < self._threshold:
            return requested_model

        chain = MODEL_DOWNGRADE_CHAIN.get(requested_model, [])
        if not chain:
            return requested_model

        # 选择链中第一个便宜的模型
        suggested = chain[0]
        log.info(
            "cost.downgrade",
            user=user_id, original=requested_model, suggested=suggested,
            quota_ratio=round(ratio, 2),
        )
        return suggested


# ─────────────────────────────────────────────
# Cost Report
# ─────────────────────────────────────────────

class CostReport:
    """生成用量报告。"""

    def __init__(self, tracker: CostTracker) -> None:
        self._tracker = tracker

    def daily_report(self, user_id: str | None = None) -> dict:
        today = date.today()
        if user_id:
            return {
                "date":  str(today),
                "user":  user_id,
                "usage": self._tracker.get_daily_usage(user_id),
            }
        # 所有用户
        users  = {r.user_id for r in self._tracker._records}
        return {
            "date":  str(today),
            "users": {u: self._tracker.get_daily_usage(u) for u in users},
            "top":   self._tracker.get_top_users(5),
        }

    def model_breakdown(self) -> dict[str, dict]:
        """按模型汇总用量。"""
        breakdown: dict[str, dict] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "calls": 0}
        )
        for rec in self._tracker._records:
            b = breakdown[rec.model]
            b["prompt_tokens"]     += rec.prompt_tokens
            b["completion_tokens"] += rec.completion_tokens
            b["cost_usd"]          += rec.cost_usd
            b["calls"]             += 1
        return dict(breakdown)
