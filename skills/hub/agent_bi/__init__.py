"""
skills/hub/agent_bi/__init__.py — 智能报表 BI 查询 Skill

接口文档: POST https://fnb-qrcode.neargo.ai/v1/report/agent_bi

支持六种查询模式（由 dimensions / timeGranularity / metrics 自动路由）：
  DEFAULT    — report_daily_statistics（默认汇总）
  GRANULAR   — timeGranularity=DAY/MONTH/YEAR（按粒度分组，返回 period 字段）
  DISH_TOP   — order_detail（菜品销量排行）
  LOW_STOCK  — product（库存不足商品列表）
  STORED_VALUE — recharge_record（储值统计）
  MEMBER_INFO  — recharge_user_info（新会员统计）

配置（utils/config.py / 环境变量）:
  AGENT_BI_API_URL        — 覆盖默认接口地址
  AGENT_BI_API_KEY        — Bearer Token（可选）
  AGENT_BI_DEFAULT_BRA_ID — 默认门店ID
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


# ── 指标标签映射（含兼容模式特有字段）─────────────────────────────────────────

_METRIC_LABELS: dict[str, str] = {
    # ── 默认模式：report_daily_statistics ──────────────────────────────
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
    # ── 菜品排行模式：order_detail ──────────────────────────────────────
    "productName":           "菜品名称",
    "productQuantity":       "销售数量",
    # ── 库存不足模式：product ───────────────────────────────────────────
    "productId":             "商品ID",
    "stockNum":              "库存数量",
    # ── 储值统计模式：recharge_record ───────────────────────────────────
    "memberCreditActualAmount": "实际储值金额",
    "memberCreditGiftAmount":   "赠送储值金额",
    "residualActualAmount":     "剩余实际储值",
    "residualGiftAmount":       "剩余赠送储值",
    # ── 会员信息模式：recharge_user_info ───────────────────────────────
    "newMemberNumber":       "新增储值会员数",
}

# 默认模式白名单（report_daily_statistics 表字段）
_VALID_METRICS = frozenset({
    "turnover", "actual_income", "actual_income_no_tips", "tips_amount",
    "order_quantity", "average_order_price", "customer_number",
    "dining_room_amount", "togo_amount", "store_delivery_amount",
    "discount_amount", "discount_ratio", "member_quantity",
    "new_member_number", "first_member_number", "second_member_number",
    "regular_member_number", "refund_amount", "refunded_quantity",
    "pay_by_cash", "pay_by_card", "pay_by_online", "pay_by_savings",
    "pos_amount", "amount_not_taxed", "vat",
})

# 兼容模式可用的触发 metrics
_DISH_TOP_METRICS   = frozenset({"productName", "productQuantity", "product_name", "product_quantity"})
_LOW_STOCK_METRICS  = frozenset({"stockNum", "stock_num"})
_STORED_VAL_METRICS = frozenset({"memberCreditActualAmount", "memberCreditGiftAmount",
                                   "residualActualAmount", "residualGiftAmount"})
_MEMBER_METRICS     = frozenset({"newMemberNumber"})

_VALID_AGGREGATIONS = frozenset({"SUM", "AVG", "MAX", "MIN", "COUNT"})

# 非数值字段（不应包裹 {value, label}，保持原始字符串）
_NON_NUMERIC_FIELDS = frozenset({
    "period", "period_label", "date",
    "productName", "productId",
})


def _default_bra_id() -> str:
    try:
        from utils.config import get_settings
        return get_settings().agent_bi_default_bra_id
    except Exception:
        return os.environ.get("AGENT_BI_DEFAULT_BRA_ID", "B17612377308779358")


# ── 时间范围工具 ───────────────────────────────────────────────────────────────

def _midnight_ms(dt: datetime) -> int:
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)


def today_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now = datetime.now(tz)
    return _midnight_ms(now), _midnight_ms(now + timedelta(days=1))


def yesterday_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now = datetime.now(tz)
    return _midnight_ms(now - timedelta(days=1)), _midnight_ms(now)


def week_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now    = datetime.now(tz)
    monday = now - timedelta(days=now.weekday())
    return _midnight_ms(monday), _midnight_ms(monday + timedelta(weeks=1))


def last_week_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now         = datetime.now(tz)
    this_monday = now - timedelta(days=now.weekday())
    last_monday = this_monday - timedelta(weeks=1)
    return _midnight_ms(last_monday), _midnight_ms(this_monday)


def month_range(tz: timezone = timezone.utc) -> tuple[int, int]:
    now       = datetime.now(tz)
    first_day = now.replace(day=1)
    next_month = (now.replace(month=now.month + 1, day=1)
                  if now.month < 12 else now.replace(year=now.year + 1, month=1, day=1))
    return _midnight_ms(first_day), _midnight_ms(next_month)


def _yoy_period(start_ms: int, end_ms: int) -> tuple[int, int]:
    def shift_year(ms: int) -> int:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        try:
            shifted = dt.replace(year=dt.year - 1)
        except ValueError:
            shifted = dt.replace(year=dt.year - 1, day=28)
        return int(shifted.timestamp() * 1000)
    return shift_year(start_ms), shift_year(end_ms)


def _pop_period(start_ms: int, end_ms: int) -> tuple[int, int]:
    duration = end_ms - start_ms
    return start_ms - duration, start_ms


# ── 查询模式检测 ──────────────────────────────────────────────────────────────

def _detect_mode(
    dimensions: str | None,
    metrics: list[str],
    time_granularity: str | None,
) -> str:
    """
    根据 dimensions / metrics / timeGranularity 检测 API 路由模式。

    返回值：
      'dish_top'     — 菜品销量排行（order_detail）
      'low_stock'    — 库存不足商品（product）
      'stored_value' — 储值统计（recharge_record）
      'member_info'  — 会员信息统计（recharge_user_info）
      'granular'     — 按粒度分组（report_daily_statistics + timeGranularity）
      'default'      — 默认汇总（report_daily_statistics）
    """
    dim = (dimensions or "").lower()

    # 菜品排行
    if dim in {"product_top", "producttop", "dish_top", "dishtop"}:
        return "dish_top"
    if any(m in _DISH_TOP_METRICS for m in metrics):
        return "dish_top"

    # 库存不足
    if dim in {"low_stock", "lowstock", "inventory_low", "inventorylow",
               "stock_below", "stockbelow"}:
        return "low_stock"
    if any(m in _LOW_STOCK_METRICS for m in metrics):
        return "low_stock"

    # 储值统计
    if any(x in dim for x in ("stored_value", "storedvalue", "member_credit", "membercredit")):
        return "stored_value"
    if any(m in _STORED_VAL_METRICS for m in metrics):
        return "stored_value"

    # 会员信息
    if any(x in dim for x in ("member_info", "memberinfo", "new_member", "newmember")):
        return "member_info"
    if any(m in _MEMBER_METRICS for m in metrics):
        return "member_info"

    # 粒度模式
    if time_granularity in {"DAY", "MONTH", "YEAR"}:
        return "granular"

    return "default"


# ── Skill 主类 ────────────────────────────────────────────────────────────────

class AgentBiSkill:
    """商家 BI 报表查询 Skill。

    支持六种查询模式（自动路由）：
      • 默认汇总        — 营业额/订单/顾客等 26 项指标
      • 粒度分组        — 按天/月/年展开，返回 period 字段
      • 菜品销量排行    — 按销量倒序列出菜品
      • 库存不足商品    — 列出库存低于阈值的商品
      • 储值统计        — 储值充值与余额按天或汇总
      • 会员信息        — 新增储值会员数
    """

    descriptor = ToolDescriptor(
        name="agent_bi",
        description=(
            "查询门店BI报表数据。支持：销售额/营业额、订单数、顾客数、支付方式、会员、退款等26项指标（默认汇总）；"
            "按天/月/年粒度展开（传 time_granularity）；菜品销量排行（dimensions=dish_top）；"
            "库存不足商品列表（dimensions=low_stock）；储值充值与余额统计（dimensions=stored_value_time_range）；"
            "新会员统计（dimensions=member_info_time_range）；同比(yoy)/环比(pop)对比分析。\n"
            "Query store BI report: sales/revenue, orders, customers, payments, members, refunds; "
            "dish ranking; low-stock items; stored-value stats; new member stats; yoy/pop comparison. "
            "bra_id: from [BI store context]; range_start/range_end: from [BI date context]; "
            "For trend queries use time_granularity=DAY/MONTH/YEAR with aggregation=SUM."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["query"],
                    "description": "操作类型，固定为 'query'",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(_VALID_METRICS)},
                    "minItems": 1,
                    "description": (
                        "默认模式下的指标字段列表（至少一个）。"
                        "菜品/库存/储值/会员兼容模式下可省略。"
                        "可用值: " + ", ".join(sorted(_VALID_METRICS))
                    ),
                },
                "bra_id": {
                    "type": "string",
                    "description": "门店ID，从系统提示的 [BI store context] 中复制；未传则使用配置默认值。",
                },
                "aggregation": {
                    "type": "string",
                    "enum": sorted(_VALID_AGGREGATIONS),
                    "default": "SUM",
                    "description": "聚合函数（默认 SUM）。粒度模式必传，菜品/库存模式无效。",
                },
                "dimensions": {
                    "type": "string",
                    "enum": [
                        "trend",
                        "product_top", "dish_top",
                        "low_stock", "inventory_low",
                        "stored_value_time_range", "member_credit_time_range",
                        "member_info_time_range", "new_member_time_range",
                    ],
                    "description": (
                        "维度参数，控制查询模式：\n"
                        "  trend / time_range        → 时间维度（适用于储值/会员按天展开）\n"
                        "  product_top / dish_top    → 菜品销量排行\n"
                        "  low_stock / inventory_low → 库存不足商品列表\n"
                        "  stored_value_time_range   → 储值统计（按天展开）\n"
                        "  member_info_time_range    → 新会员统计（按天展开）"
                    ),
                },
                "time_granularity": {
                    "type": "string",
                    "enum": ["DAY", "MONTH", "YEAR"],
                    "description": (
                        "查询粒度（优先级高于 dimensions）。"
                        "DAY=按自然日分组，MONTH=按月分组，YEAR=按年分组。"
                        "使用此参数时 aggregation 必传且必须合法，返回结果含 period 字段。"
                        "示例：查询近一个月每天的营业额趋势，传 time_granularity=DAY + aggregation=SUM。"
                    ),
                },
                "stock_threshold": {
                    "type": "integer",
                    "description": "库存阈值（仅 low_stock 模式有效），库存 ≤ 此值的商品会被列出，默认 5。",
                },
                "range_start": {
                    "type": "integer",
                    "description": (
                        "统计开始时间（13位毫秒时间戳，每日凌晨00:00:00）。"
                        "使用 [BI date context] 中的 rangeStart 值。"
                    ),
                },
                "range_end": {
                    "type": "integer",
                    "description": (
                        "统计结束时间（13位毫秒时间戳，每日凌晨00:00:00）。"
                        "单日查询：range_end = range_start + 86400000。"
                        "不得等于 range_start（零长度窗口无意义）。"
                    ),
                },
                "comparison": {
                    "type": "string",
                    "enum": ["yoy", "pop"],
                    "description": (
                        "对比模式（可选）：yoy=同比（与去年同期），pop=环比（与上一时段）。"
                        "仅当用户明确要求同比/环比时设置。"
                        "不与 dish_top / low_stock 模式同时使用。"
                    ),
                },
            },
            "required": ["action"],
        },
        source="skill",
        permission=PermissionLevel.NETWORK,
        timeout_ms=30_000,
        tags=["report", "bi", "analytics", "business", "sales", "comparison"],
    )

    def __init__(self, api_url: str | None = None, api_key: str | None = None) -> None:
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

    # ── 公开入口 ─────────────────────────────────────────────────────────────

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = arguments.get("action", "query")
        if action == "query":
            return await self._query(
                metrics=arguments.get("metrics") or [],
                bra_id=arguments.get("bra_id"),
                aggregation=arguments.get("aggregation", "SUM"),
                dimensions=arguments.get("dimensions"),
                time_granularity=arguments.get("time_granularity"),
                stock_threshold=arguments.get("stock_threshold"),
                range_start=arguments.get("range_start"),
                range_end=arguments.get("range_end"),
                comparison=arguments.get("comparison"),
            )
        return {"error": f"不支持的操作: {action!r}，目前仅支持: query", "type": "InvalidAction"}

    # ── 内部工具 ─────────────────────────────────────────────────────────────

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
        except Exception as exc:
            log.error("agent_bi.unexpected_error", label=label, error=str(exc))
            return None

    @staticmethod
    def _extract_rows(
        result: dict[str, Any] | None,
        range_start: int | None = None,
    ) -> list[dict[str, Any]]:
        """从 API 响应中提取 metrics 列表，过滤空行。

        多行数据处理：
          1. 优先使用 API 返回的 period / date 字段作为 period_label（含周几中文标注）。
          2. 若 API 未返回该字段且 range_start 已知，则按 i*86400000 推算（兜底）。
        """
        if not result:
            return []
        if result.get("code") not in {0, 200}:
            return []

        data_block = result.get("data") or {}
        raw: list = data_block.get("metrics", [])
        rows = [r for r in raw if isinstance(r, dict)]

        _WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

        if len(rows) > 1:
            for i, row in enumerate(rows):
                if "period_label" in row:
                    continue  # 已有，跳过

                # 优先用 API 返回的 period / date 字段
                raw_period = row.get("period") or row.get("date")
                if raw_period:
                    # 尝试解析日期字符串并附加周几
                    try:
                        # 支持 "yyyy-MM-dd" / "yyyy-MM" / "yyyy"
                        parts = str(raw_period).split("-")
                        if len(parts) == 3:   # yyyy-MM-dd
                            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                                          tzinfo=timezone.utc)
                            weekday = _WEEKDAY_ZH[dt.weekday()]
                            row["period_label"] = f"{dt.month}/{dt.day}({weekday})"
                        else:
                            row["period_label"] = raw_period   # yyyy-MM / yyyy 直接使用
                    except Exception:
                        row["period_label"] = raw_period
                elif range_start is not None:
                    # 兜底：按偏移量推算
                    day_ts = range_start + i * 86_400_000
                    dt = datetime.fromtimestamp(day_ts / 1000, tz=timezone.utc)
                    weekday = _WEEKDAY_ZH[dt.weekday()]
                    row["period_label"] = f"{dt.month}/{dt.day}({weekday})"

        return rows

    @staticmethod
    def _label_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """为数值类型指标附加中文 label；非数值字段（period/date/productName 等）保持原始值。"""
        result = []
        for row in raw_rows:
            labeled: dict[str, Any] = {}
            for k, v in row.items():
                if k in _NON_NUMERIC_FIELDS or not isinstance(v, (int, float)):
                    # 字符串/日期/ID 字段：直接保留，不套 {value, label}
                    labeled[k] = v
                else:
                    labeled[k] = {"value": v, "label": _METRIC_LABELS.get(k, k)}
            result.append(labeled)
        return result

    @staticmethod
    def _compute_deltas(
        current_rows: list[dict[str, Any]],
        prev_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """计算同比/环比变化量和变化率（仅对首行数值字段计算）。"""
        if not current_rows or not prev_rows:
            return {}
        curr = current_rows[0]
        prev = prev_rows[0]
        deltas: dict[str, dict[str, Any]] = {}
        for k in curr:
            if k in _NON_NUMERIC_FIELDS:
                continue
            cv = curr[k]
            pv = prev.get(k)
            curr_val = float(cv) if isinstance(cv, (int, float)) else 0.0
            prev_val = float(pv) if isinstance(pv, (int, float)) else 0.0
            delta     = curr_val - prev_val
            delta_pct = round(delta / prev_val * 100, 2) if prev_val != 0 else None
            deltas[k] = {"delta": round(delta, 4), "delta_pct": delta_pct}
        return deltas

    # ── 核心查询 ─────────────────────────────────────────────────────────────

    async def _query(
        self,
        metrics: list[str],
        bra_id: str | None = None,
        aggregation: str = "SUM",
        dimensions: str | None = None,
        time_granularity: str | None = None,
        stock_threshold: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        comparison: str | None = None,
    ) -> dict[str, Any]:

        # ── 默认门店回落 ──────────────────────────────────────────────────────
        if not bra_id:
            bra_id = _default_bra_id()

        # ── 检测查询模式 ──────────────────────────────────────────────────────
        mode = _detect_mode(dimensions, metrics, time_granularity)

        # ── 参数校验（仅默认/粒度模式需校验 metrics 白名单）───────────────────
        if mode in ("default", "granular"):
            if not metrics:
                return {"error": "metrics 不能为空，默认模式下请至少指定一个指标",
                        "valid_metrics": sorted(_VALID_METRICS)}
            invalid = [m for m in metrics if m not in _VALID_METRICS]
            if invalid:
                return {"error": f"无效指标: {invalid}，默认模式仅支持标准字段",
                        "valid_metrics": sorted(_VALID_METRICS)}

        if mode == "granular" and aggregation not in _VALID_AGGREGATIONS:
            return {"error": f"粒度模式必须传合法的 aggregation，当前值: {aggregation!r}",
                    "valid_aggregations": sorted(_VALID_AGGREGATIONS)}

        if aggregation and aggregation not in _VALID_AGGREGATIONS:
            # 非法聚合函数：按文档行为，不聚合，但打 warning 便于排查
            log.warning("agent_bi.invalid_aggregation_fallback", aggregation=aggregation)
            aggregation = "SUM"

        if comparison and comparison not in ("yoy", "pop"):
            return {"error": f"无效对比类型: {comparison!r}，支持: yoy（同比）、pop（环比）",
                    "type": "InvalidComparison"}

        # dish_top / low_stock 无意义做同比/环比，静默忽略
        if mode in ("dish_top", "low_stock"):
            comparison = None

        # ── 时间范围自动修复（单日零窗口）───────────────────────────────────
        _ONE_DAY_MS = 86_400_000
        if range_start is not None and (range_end is None or range_end <= range_start):
            range_end = range_start + _ONE_DAY_MS
            log.info("agent_bi.range_end_auto_fixed",
                     range_start=range_start, range_end=range_end)

        # ── 构建 API 请求体 ──────────────────────────────────────────────────
        payload: dict[str, Any] = {}

        # 公共字段
        if bra_id:
            payload["braId"] = bra_id
        if range_start is not None:
            payload["rangeStart"] = range_start
        if range_end is not None:
            payload["rangeEnd"] = range_end

        if mode == "dish_top":
            # 菜品排行：dimensions 触发，aggregation 不参与
            payload["dimensions"] = dimensions or "product_top"

        elif mode == "low_stock":
            # 库存不足：dimensions 触发，stockThreshold 可选
            payload["dimensions"] = dimensions or "low_stock"
            if stock_threshold is not None:
                payload["stockThreshold"] = stock_threshold

        elif mode == "stored_value":
            # 储值统计：metrics 触发；dimensions 可选（含 trend 时按天展开）
            payload["metrics"] = metrics or list(_STORED_VAL_METRICS)
            if dimensions:
                payload["dimensions"] = dimensions

        elif mode == "member_info":
            # 会员信息：dimensions 触发；metrics 可选
            if dimensions:
                payload["dimensions"] = dimensions
            if metrics:
                payload["metrics"] = metrics

        elif mode == "granular":
            # 粒度分组：timeGranularity 优先
            payload["metrics"] = metrics
            payload["aggregation"] = aggregation
            payload["timeGranularity"] = time_granularity
            # dimensions 中的 day/month/year 关键词也保留（兜底）
            if dimensions:
                payload["dimensions"] = dimensions

        else:  # default
            payload["metrics"] = metrics
            payload["aggregation"] = aggregation
            if dimensions:
                payload["dimensions"] = dimensions

        headers = self._build_headers()

        log.info("agent_bi.mode_detected", mode=mode, dimensions=dimensions,
                 time_granularity=time_granularity, metrics=metrics)

        # ── 发起请求（含同比/环比并发）──────────────────────────────────────
        prev_start: int | None = None
        prev_end:   int | None = None

        if comparison and range_start is not None and range_end is not None:
            if comparison == "yoy":
                prev_start, prev_end = _yoy_period(range_start, range_end)
            else:
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

        # ── 检查状态码 ───────────────────────────────────────────────────────
        if curr_result is None:
            return {"error": "API 请求失败，请检查网络或稍后重试", "type": "NetworkError"}

        api_code = curr_result.get("code")
        api_msg  = curr_result.get("message", "")
        if api_code not in {0, 200}:
            log.warning("agent_bi.api_error", code=api_code, message=api_msg)
            return {"error": api_msg or "API 返回错误", "code": api_code,
                    "type": "APIError", "detail": curr_result}

        # ── 解析响应 ─────────────────────────────────────────────────────────
        raw_metrics = self._extract_rows(curr_result, range_start=range_start)
        labeled     = self._label_rows(raw_metrics)

        log.info("agent_bi.success",
                 mode=mode, bra_id=bra_id,
                 range_start=range_start, range_end=range_end,
                 row_count=len(raw_metrics), comparison=comparison)

        # ── 空数据：明确告知 LLM，禁止重试 ───────────────────────────────────
        if not raw_metrics:
            log.info("agent_bi.no_data", mode=mode, bra_id=bra_id,
                     range_start=range_start, range_end=range_end)
            return {
                "success":   True,
                "no_data":   True,
                "mode":      mode,
                "bra_id":    bra_id,
                "range_start": range_start,
                "range_end":   range_end,
                "metrics":   [],
                "message":   "该时间段内暂无数据，请直接告知用户没有相关数据，勿重试。",
            }

        response: dict[str, Any] = {
            "success":          True,
            "mode":             mode,
            "aggregation":      aggregation,
            "bra_id":           bra_id,
            "range_start":      range_start,
            "range_end":        range_end,
            "metrics":          labeled,
            "raw_metrics":      raw_metrics,
        }

        # ── 追加对比期数据 ───────────────────────────────────────────────────
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
