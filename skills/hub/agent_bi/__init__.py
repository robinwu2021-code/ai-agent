"""
skills/hub/agent_bi/__init__.py — 智能报表 BI 查询 Skill

接口文档: POST https://fnb-qrcode.neargo.ai/v1/report/agent_bi
请求体:
  - braId      : 门店ID（可选，不传查全部）
  - metrics    : 指标字段列表
  - aggregation: 聚合函数 SUM/AVG/MAX/MIN/COUNT
  - rangeStart : 13位毫秒时间戳，每日凌晨 00:00:00
  - rangeEnd   : 13位毫秒时间戳，每日凌晨 00:00:00

环境变量:
  AGENT_BI_API_URL — 覆盖默认接口地址
  AGENT_BI_API_KEY — Bearer Token（可选，接口需要鉴权时设置）
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from core.models import PermissionLevel, ToolDescriptor


# ── 指标中文标签映射 ──────────────────────────────────────────────────────────

_METRIC_LABELS: dict[str, str] = {
    "turnover":              "总销售额",
    "actual_income":         "实际收入（含小费）",
    "actual_income_no_tips": "实际收入（不含小费）",
    "tips_amount":           "小费总额",
    "order_quantity":        "订单总数",
    "average_order_price":   "平均订单金额",
    "customer_number":       "顾客总数",
    "dining_room_amount":    "堂食销售额",
    "togo_amount":           "外带销售额",
    "store_delivery_amount": "门店配送销售额",
    "discount_amount":       "折扣总额",
    "discount_ratio":        "折扣比率",
    "member_quantity":       "会员订单数",
    "new_member_number":     "新注册会员数",
    "first_member_number":   "首次消费顾客数",
    "second_member_number":  "回头客数",
    "regular_member_number": "常客数",
    "refund_amount":         "退款总额",
    "refunded_quantity":     "退款订单数",
    "pay_by_cash":           "现金支付",
    "pay_by_card":           "卡支付",
    "pay_by_online":         "在线支付",
    "pay_by_savings":        "储值支付",
    "pos_amount":            "POS卡支付",
    "amount_not_taxed":      "不含税销售额",
    "vat":                   "增值税",
}

_VALID_METRICS      = frozenset(_METRIC_LABELS.keys())
_VALID_AGGREGATIONS = frozenset({"SUM", "AVG", "MAX", "MIN", "COUNT"})
_DEFAULT_API_URL    = "https://fnb-qrcode.neargo.ai/v1/report/agent_bi"


def _default_bra_id() -> str | None:
    """返回配置文件中的默认门店ID（AGENT_BI_DEFAULT_BRA_ID 环境变量）。"""
    return os.environ.get("AGENT_BI_DEFAULT_BRA_ID") or None


# ── 时间范围工具函数 ──────────────────────────────────────────────────────────

def _midnight_ms(dt: datetime) -> int:
    """返回某天凌晨 00:00:00 的 13 位毫秒时间戳。"""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)


def today_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    """今天的时间范围：今日 00:00 ~ 明日 00:00（毫秒）。"""
    now   = datetime.now(tz)
    start = _midnight_ms(now)
    end   = _midnight_ms(now + timedelta(days=1))
    return start, end


def yesterday_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    """昨天的时间范围：昨日 00:00 ~ 今日 00:00（毫秒）。"""
    now   = datetime.now(tz)
    start = _midnight_ms(now - timedelta(days=1))
    end   = _midnight_ms(now)
    return start, end


def week_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    """本周时间范围：本周一 00:00 ~ 下周一 00:00（毫秒）。"""
    now    = datetime.now(tz)
    monday = now - timedelta(days=now.weekday())
    start  = _midnight_ms(monday)
    end    = _midnight_ms(monday + timedelta(weeks=1))
    return start, end


def last_week_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    """上周时间范围：上周一 00:00 ~ 本周一 00:00（毫秒）。"""
    now          = datetime.now(tz)
    this_monday  = now - timedelta(days=now.weekday())
    last_monday  = this_monday - timedelta(weeks=1)
    start        = _midnight_ms(last_monday)
    end          = _midnight_ms(this_monday)
    return start, end


def month_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    """本月时间范围：本月1日 00:00 ~ 下月1日 00:00（毫秒）。"""
    now       = datetime.now(tz)
    first_day = now.replace(day=1)
    start     = _midnight_ms(first_day)
    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1)
    else:
        next_month = now.replace(month=now.month + 1, day=1)
    end = _midnight_ms(next_month)
    return start, end


# ── Skill 主类 ───────────────────────────────────────────────────────────────

class AgentBiSkill:
    """商家 BI 报表查询 Skill。

    通过对话框调用，支持按门店、时间范围、聚合函数查询多维度业务指标。
    接口地址: https://fnb-qrcode.neargo.ai/v1/report/agent_bi
    """

    descriptor = ToolDescriptor(
        name="agent_bi",
        description=(
            "查询商家BI报表数据，支持销售额、订单数、顾客数等多维度业务指标统计分析。"
            "可按门店(bra_id)和时间范围(range_start/range_end 毫秒时间戳)过滤，"
            "支持 SUM/AVG/MAX/MIN/COUNT 聚合。结果附带中文标签。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["query"],
                    "description": "操作类型，目前支持 query",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(_VALID_METRICS),
                    },
                    "minItems": 1,
                    "description": (
                        "要查询的指标列表（至少一个），可选值: "
                        + ", ".join(sorted(_VALID_METRICS))
                    ),
                },
                "bra_id": {
                    "type": "string",
                    "description": "门店ID，不传则查询所有门店",
                },
                "aggregation": {
                    "type": "string",
                    "enum": sorted(_VALID_AGGREGATIONS),
                    "default": "SUM",
                    "description": "聚合函数（默认 SUM）",
                },
                "range_start": {
                    "type": "integer",
                    "description": "统计开始时间（13位毫秒时间戳，精确到每日凌晨00:00:00）",
                },
                "range_end": {
                    "type": "integer",
                    "description": "统计结束时间（13位毫秒时间戳，精确到每日凌晨00:00:00）",
                },
            },
            "required": ["action", "metrics"],
        },
        source="skill",
        permission=PermissionLevel.NETWORK,
        timeout_ms=15_000,
        tags=["report", "bi", "analytics", "business", "sales"],
    )

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_url = api_url or os.environ.get("AGENT_BI_API_URL", _DEFAULT_API_URL)
        self._api_key = api_key or os.environ.get("AGENT_BI_API_KEY", "")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Skill 统一入口，由 SkillRegistry 调用。"""
        action = arguments.get("action", "query")
        if action == "query":
            return await self._query(
                metrics=arguments.get("metrics", []),
                bra_id=arguments.get("bra_id"),
                aggregation=arguments.get("aggregation", "SUM"),
                range_start=arguments.get("range_start"),
                range_end=arguments.get("range_end"),
            )
        return {
            "error": f"不支持的操作: {action!r}，目前仅支持: query",
            "type": "InvalidAction",
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _query(
        self,
        metrics: list[str],
        bra_id: str | None = None,
        aggregation: str = "SUM",
        range_start: int | None = None,
        range_end: int | None = None,
    ) -> dict[str, Any]:
        # ── 默认门店回落 ──────────────────────────────────────────────
        # 前端未传入 bra_id 时，自动使用 AGENT_BI_DEFAULT_BRA_ID 配置
        if not bra_id:
            bra_id = _default_bra_id()

        # ── 参数校验 ─────────────────────────────────────────────────
        if not metrics:
            return {
                "error": "metrics 不能为空，请至少指定一个指标",
                "valid_metrics": sorted(_VALID_METRICS),
            }

        invalid_metrics = [m for m in metrics if m not in _VALID_METRICS]
        if invalid_metrics:
            return {
                "error": f"无效指标: {invalid_metrics}",
                "valid_metrics": sorted(_VALID_METRICS),
            }

        if aggregation not in _VALID_AGGREGATIONS:
            return {
                "error": f"无效聚合函数: {aggregation!r}",
                "valid_aggregations": sorted(_VALID_AGGREGATIONS),
            }

        # ── 构建请求体 ────────────────────────────────────────────────
        payload: dict[str, Any] = {
            "metrics":     metrics,
            "aggregation": aggregation,
        }
        if bra_id:
            payload["braId"] = bra_id
        if range_start is not None:
            payload["rangeStart"] = range_start
        if range_end is not None:
            payload["rangeEnd"] = range_end

        # ── HTTP 请求 ─────────────────────────────────────────────────
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    self._api_url,
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()

        except httpx.TimeoutException:
            return {
                "error": "请求超时，BI 接口响应时间过长，请稍后重试",
                "type": "TimeoutError",
            }
        except httpx.HTTPStatusError as exc:
            return {
                "error": f"HTTP 错误: {exc.response.status_code}",
                "type": "HTTPError",
                "status_code": exc.response.status_code,
            }
        except httpx.RequestError as exc:
            return {
                "error": f"网络错误: {exc}",
                "type": "NetworkError",
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "type": type(exc).__name__}

        # ── 处理 API 错误码 ────────────────────────────────────────────
        if result.get("code") != 0:
            return {
                "error": result.get("message", "API 返回错误"),
                "code":  result.get("code"),
                "type":  "APIError",
            }

        # ── 解析并丰富响应 ─────────────────────────────────────────────
        raw_metrics: list[dict[str, Any]] = (
            result.get("data") or {}
        ).get("metrics", [])

        labeled: list[dict[str, Any]] = []
        for row in raw_metrics:
            labeled.append({
                k: {"value": v, "label": _METRIC_LABELS.get(k, k)}
                for k, v in row.items()
            })

        return {
            "success":     True,
            "aggregation": aggregation,
            "bra_id":      bra_id,
            "range_start": range_start,
            "range_end":   range_end,
            "metrics":     labeled,
            "raw_metrics": raw_metrics,
        }
