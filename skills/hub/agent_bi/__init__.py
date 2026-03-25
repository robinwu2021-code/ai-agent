"""
skills/hub/agent_bi/__init__.py — 智能报表 BI 查询 Skill

接口文档: POST https://fnb-qrcode.neargo.ai/v1/report/agent_bi
请求体:
  - braId      : 门店ID（可选，不传查全部）
  - metrics    : 指标字段列表
  - aggregation: 聚合函数 SUM/AVG/MAX/MIN/COUNT
  - rangeStart : 13位毫秒时间戳，每日凌晨 00:00:00
  - rangeEnd   : 13位毫秒时间戳，每日凌晨 00:00:00

配置（utils/config.py / 环境变量）:
  AGENT_BI_API_URL        — 覆盖默认接口地址（默认 https://fnb-qrcode.neargo.ai/v1/report/agent_bi）
  AGENT_BI_API_KEY        — Bearer Token（可选，接口需要鉴权时设置）
  AGENT_BI_DEFAULT_BRA_ID — 默认门店ID，前端未传入时自动使用（默认 B17612377308779358）
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from core.models import PermissionLevel, ToolDescriptor

log = structlog.get_logger("agent_bi")


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


def _default_bra_id() -> str:
    """返回配置的默认门店ID。
    优先读取 utils.config.get_settings()，回落到内置默认值 B17612377308779358。
    可通过 AGENT_BI_DEFAULT_BRA_ID 环境变量覆盖。
    """
    try:
        from utils.config import get_settings
        return get_settings().agent_bi_default_bra_id
    except Exception:
        return os.environ.get("AGENT_BI_DEFAULT_BRA_ID", "B17612377308779358")


# ── 时间范围工具函数 ──────────────────────────────────────────────────────────

def _midnight_ms(dt: datetime) -> int:
    """返回某天凌晨 00:00:00 的 13 位毫秒时间戳。"""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)


def today_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now   = datetime.now(tz)
    start = _midnight_ms(now)
    end   = _midnight_ms(now + timedelta(days=1))
    return start, end


def yesterday_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now   = datetime.now(tz)
    start = _midnight_ms(now - timedelta(days=1))
    end   = _midnight_ms(now)
    return start, end


def week_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now    = datetime.now(tz)
    monday = now - timedelta(days=now.weekday())
    start  = _midnight_ms(monday)
    end    = _midnight_ms(monday + timedelta(weeks=1))
    return start, end


def last_week_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now          = datetime.now(tz)
    this_monday  = now - timedelta(days=now.weekday())
    last_monday  = this_monday - timedelta(weeks=1)
    start        = _midnight_ms(last_monday)
    end          = _midnight_ms(this_monday)
    return start, end


def month_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now       = datetime.now(tz)
    first_day = now.replace(day=1)
    start     = _midnight_ms(first_day)
    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1)
    else:
        next_month = now.replace(month=now.month + 1, day=1)
    end = _midnight_ms(next_month)
    return start, end


def _yoy_period(start_ms: int, end_ms: int) -> tuple[int, int]:
    """同比：返回去年同期的时间范围（精确到天）。"""
    def shift_year(ms: int) -> int:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        try:
            shifted = dt.replace(year=dt.year - 1)
        except ValueError:          # Feb 29 → Feb 28
            shifted = dt.replace(year=dt.year - 1, day=28)
        return int(shifted.timestamp() * 1000)

    return shift_year(start_ms), shift_year(end_ms)


def _pop_period(start_ms: int, end_ms: int) -> tuple[int, int]:
    """环比：返回紧前同等时长的时间范围。"""
    duration = end_ms - start_ms
    return start_ms - duration, start_ms


# ── Skill 主类 ───────────────────────────────────────────────────────────────

class AgentBiSkill:
    """商家 BI 报表查询 Skill。

    通过对话框调用，支持按门店、时间范围、聚合函数查询多维度业务指标。
    支持同比（yoy）和环比（pop）对比分析。
    接口地址: https://fnb-qrcode.neargo.ai/v1/report/agent_bi
    """

    descriptor = ToolDescriptor(
        name="agent_bi",
        description=(
            # 中文：让 LLM 在中文对话中也能识别
            "查询门店BI报表数据（销售额/营业额、订单数、顾客数、支付方式、会员、退款等）。"
            "当用户询问销售数据、营业额、订单量、客流量等门店经营指标时，必须调用此工具。"
            "支持同比（yoy，与去年同期对比）和环比（pop，与上一时段对比）分析。\n"
            # English: for multi-language support
            "Query merchant BI report data: turnover/sales, orders, customers, payments, members, refunds. "
            "Call this tool whenever the user asks about store sales, revenue, order count, or any business metrics. "
            "Supports year-over-year (yoy/同比) and period-over-period (pop/环比) comparison. "
            # Parameter guidance
            "bra_id: copy from [BI store context] in system prompt, or omit to use the configured default store. "
            "range_start/range_end: use the ms-timestamp pairs from [BI date context] in the user message. "
            "comparison: set to 'yoy' for 同比 (vs. same period last year), 'pop' for 环比 (vs. immediately preceding period). "
            "Respond in the same language the user used."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["query"],
                    "description": "Operation type, currently only 'query' is supported.",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(_VALID_METRICS),
                    },
                    "minItems": 1,
                    "description": (
                        "List of metric fields to query (at least one). "
                        "Available values: " + ", ".join(sorted(_VALID_METRICS))
                    ),
                },
                "bra_id": {
                    "type": "string",
                    "description": (
                        "Store ID. Copy the value from '[BI store context] Current store ID: <value>' "
                        "in the system prompt. Omit this field if no store context is present."
                    ),
                },
                "aggregation": {
                    "type": "string",
                    "enum": sorted(_VALID_AGGREGATIONS),
                    "default": "SUM",
                    "description": "Aggregation function (default SUM).",
                },
                "range_start": {
                    "type": "integer",
                    "description": (
                        "Range start: 13-digit ms timestamp at midnight 00:00:00. "
                        "Use the rangeStart value from the matching entry in [BI date context]. "
                        "For a specific date not in the context: convert that date to its midnight ms timestamp."
                    ),
                },
                "range_end": {
                    "type": "integer",
                    "description": (
                        "Range end: 13-digit ms timestamp at midnight 00:00:00. "
                        "RULE — for a single day: range_end = range_start + 86400000 (exactly +24 h). "
                        "Use the rangeEnd value from [BI date context], or compute: rangeStart + 86400000. "
                        "Never set range_end equal to range_start (that is a zero-length window)."
                    ),
                },
                "comparison": {
                    "type": "string",
                    "enum": ["yoy", "pop"],
                    "description": (
                        "Comparison mode (optional). "
                        "'yoy' = 同比: compare current period vs same period last year. "
                        "'pop' = 环比: compare current period vs immediately preceding period of same length. "
                        "Only set when the user explicitly asks for 同比/环比 analysis. "
                        "The response will include both current and previous period data plus delta percentages."
                    ),
                },
            },
            "required": ["action", "metrics"],
        },
        source="skill",
        permission=PermissionLevel.NETWORK,
        timeout_ms=30_000,
        tags=["report", "bi", "analytics", "business", "sales", "comparison"],
    )

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from utils.config import get_settings
            cfg = get_settings()
            default_url = cfg.agent_bi_api_url
            default_key = cfg.agent_bi_api_key
        except Exception:
            default_url = "https://fnb-qrcode.neargo.ai/v1/report/agent_bi"
            default_key = ""
        self._api_url = api_url or default_url
        self._api_key = api_key or default_key

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
                comparison=arguments.get("comparison"),
            )
        return {
            "error": f"不支持的操作: {action!r}，目前仅支持: query",
            "type": "InvalidAction",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _call_api_raw(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        label: str = "current",
    ) -> dict[str, Any] | None:
        """发起单次 API 请求并返回原始 result dict；失败返回 None。"""
        log.info("agent_bi.request", label=label, url=self._api_url, payload=payload)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self._api_url, json=payload, headers=headers)
                log.info("agent_bi.response", label=label,
                         http_status=resp.status_code,
                         body_preview=resp.text[:500])
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException:
            log.error("agent_bi.timeout", label=label, url=self._api_url)
            return None
        except httpx.HTTPStatusError as exc:
            log.error("agent_bi.http_error", label=label,
                      status=exc.response.status_code, body=exc.response.text[:300])
            return None
        except httpx.RequestError as exc:
            log.error("agent_bi.network_error", label=label, error=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            log.error("agent_bi.unexpected_error", label=label, error=str(exc))
            return None

    @staticmethod
    def _extract_rows(
        result: dict[str, Any] | None,
        range_start: int | None = None,
    ) -> list[dict[str, Any]]:
        """从 API 响应中提取 metrics 列表，过滤掉 null 行。

        当返回多行（逐日数据）时，自动为每行注入 period_label（如 "12/14(周一)"），
        供前端折线图 X 轴使用。
        """
        if not result:
            return []
        _SUCCESS_CODES = {0, 200}
        api_code = result.get("code")
        if api_code not in _SUCCESS_CODES:
            return []
        data_block = result.get("data") or {}
        raw: list = data_block.get("metrics", [])
        rows = [r for r in raw if isinstance(r, dict)]

        # 多行时注入 period_label —— 让前端图表有可读的 X 轴标签
        _WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        if len(rows) > 1 and range_start is not None:
            for i, row in enumerate(rows):
                if "period_label" not in row:
                    day_ts = range_start + i * 86_400_000
                    dt = datetime.fromtimestamp(day_ts / 1000, tz=timezone.utc)
                    weekday = _WEEKDAY_ZH[dt.weekday()]   # Mon=0
                    row["period_label"] = f"{dt.month}/{dt.day}({weekday})"

        return rows

    @staticmethod
    def _label_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """为每个指标值附加中文 label。"""
        return [
            {k: {"value": v, "label": _METRIC_LABELS.get(k, k)} for k, v in row.items()}
            for row in raw_rows
        ]

    @staticmethod
    def _compute_deltas(
        current_rows: list[dict[str, Any]],
        prev_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """计算同比/环比变化量和变化率。"""
        if not current_rows or not prev_rows:
            return {}
        curr = current_rows[0]
        prev = prev_rows[0]
        deltas: dict[str, dict[str, Any]] = {}
        for k in curr:
            cv = curr[k]
            pv = prev.get(k)
            curr_val = float(cv) if isinstance(cv, (int, float)) else 0.0
            prev_val = float(pv) if isinstance(pv, (int, float)) else 0.0
            delta = curr_val - prev_val
            delta_pct = round(delta / prev_val * 100, 2) if prev_val != 0 else None
            deltas[k] = {"delta": round(delta, 4), "delta_pct": delta_pct}
        return deltas

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    async def _query(
        self,
        metrics: list[str],
        bra_id: str | None = None,
        aggregation: str = "SUM",
        range_start: int | None = None,
        range_end: int | None = None,
        comparison: str | None = None,   # 'yoy' | 'pop'
    ) -> dict[str, Any]:
        # ── 默认门店回落 ──────────────────────────────────────────────
        if not bra_id:
            bra_id = _default_bra_id()

        # ── 参数校验 ─────────────────────────────────────────────────
        if not metrics:
            return {"error": "metrics 不能为空，请至少指定一个指标",
                    "valid_metrics": sorted(_VALID_METRICS)}

        invalid_metrics = [m for m in metrics if m not in _VALID_METRICS]
        if invalid_metrics:
            return {"error": f"无效指标: {invalid_metrics}",
                    "valid_metrics": sorted(_VALID_METRICS)}

        if aggregation not in _VALID_AGGREGATIONS:
            return {"error": f"无效聚合函数: {aggregation!r}",
                    "valid_aggregations": sorted(_VALID_AGGREGATIONS)}

        if comparison and comparison not in ("yoy", "pop"):
            return {"error": f"无效对比类型: {comparison!r}，支持: yoy（同比）、pop（环比）",
                    "type": "InvalidComparison"}

        # ── 单日时间范围自动修复 ──────────────────────────────────────
        _ONE_DAY_MS = 86_400_000
        if range_start is not None:
            if range_end is None or range_end <= range_start:
                range_end = range_start + _ONE_DAY_MS
                log.info("agent_bi.range_end_auto_fixed",
                         range_start=range_start, range_end=range_end)

        # ── 构建当期请求体 ────────────────────────────────────────────
        payload: dict[str, Any] = {"metrics": metrics, "aggregation": aggregation}
        if bra_id:
            payload["braId"] = bra_id
        if range_start is not None:
            payload["rangeStart"] = range_start
        if range_end is not None:
            payload["rangeEnd"] = range_end

        headers = self._build_headers()

        # ── 发起请求（含同比/环比并发） ───────────────────────────────
        prev_start: int | None = None
        prev_end:   int | None = None

        if comparison and range_start is not None and range_end is not None:
            if comparison == "yoy":
                prev_start, prev_end = _yoy_period(range_start, range_end)
            else:  # pop
                prev_start, prev_end = _pop_period(range_start, range_end)

            prev_payload = {**payload, "rangeStart": prev_start, "rangeEnd": prev_end}
            log.info("agent_bi.comparison_periods",
                     comparison=comparison,
                     current=(range_start, range_end),
                     previous=(prev_start, prev_end))

            curr_result, prev_result = await asyncio.gather(
                self._call_api_raw(payload, headers, label="current"),
                self._call_api_raw(prev_payload, headers, label="previous"),
            )
        else:
            curr_result = await self._call_api_raw(payload, headers, label="current")
            prev_result = None

        # ── 检查当期状态码 ────────────────────────────────────────────
        if curr_result is None:
            return {"error": "API 请求失败，请检查网络或稍后重试", "type": "NetworkError"}

        api_code = curr_result.get("code")
        api_msg  = curr_result.get("message", "")
        if api_code not in {0, 200}:
            log.warning("agent_bi.api_error", code=api_code, message=api_msg)
            return {"error": api_msg or "API 返回错误", "code": api_code,
                    "type": "APIError", "detail": curr_result}

        # ── 解析当期数据 ──────────────────────────────────────────────
        raw_metrics  = self._extract_rows(curr_result, range_start=range_start)
        labeled      = self._label_rows(raw_metrics)

        log.info("agent_bi.success",
                 bra_id=bra_id, aggregation=aggregation,
                 range_start=range_start, range_end=range_end,
                 row_count=len(raw_metrics), comparison=comparison)

        response: dict[str, Any] = {
            "success":     True,
            "aggregation": aggregation,
            "bra_id":      bra_id,
            "range_start": range_start,
            "range_end":   range_end,
            "metrics":     labeled,
            "raw_metrics": raw_metrics,
        }

        # ── 追加对比期数据 ────────────────────────────────────────────
        if comparison and prev_start is not None:
            prev_raw     = self._extract_rows(prev_result, range_start=prev_start)
            prev_labeled = self._label_rows(prev_raw)
            deltas       = self._compute_deltas(raw_metrics, prev_raw)

            response.update({
                "comparison_type":      comparison,
                "previous_range_start": prev_start,
                "previous_range_end":   prev_end,
                "previous_metrics":     prev_labeled,
                "previous_raw_metrics": prev_raw,
                "deltas":               deltas,
            })

            log.info("agent_bi.comparison_done",
                     comparison=comparison,
                     prev_rows=len(prev_raw),
                     delta_keys=list(deltas.keys()))

        return response
