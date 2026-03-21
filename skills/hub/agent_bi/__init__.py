"""
skills/hub/agent_bi/__init__.py — 门店 BI 数据查询 Skill

通过 /v1/report/agent_bi 接口查询门店经营指标，支持：
- 25+ 项业务指标（销售额/订单/会员/支付/退款等）
- 日期快捷词（today/yesterday/last_7_days/this_month 等）
- 多指标并查 + 聚合函数

配置：
  环境变量 AGENT_BI_BASE_URL — 接口基础地址，默认 http://localhost:8080
  环境变量 AGENT_BI_TOKEN   — 可选 Bearer Token
"""
from __future__ import annotations

import os
from datetime import date, timedelta, datetime, timezone
from typing import Any

from core.models import PermissionLevel, ToolDescriptor

# ── 默认指标（用户未指定时返回的核心指标）────────────────────────────
_DEFAULT_METRICS = [
    "turnover",
    "order_quantity",
    "average_order_price",
    "customer_number",
]

# ── 合法指标集合（用于校验）──────────────────────────────────────────
_VALID_METRICS = {
    "turnover", "actual_income", "actual_income_no_tips", "tips_amount",
    "order_quantity", "average_order_price", "customer_number",
    "dining_room_amount", "togo_amount", "store_delivery_amount",
    "discount_amount", "discount_ratio",
    "member_quantity", "new_member_number", "first_member_number",
    "second_member_number", "regular_member_number",
    "refund_amount", "refunded_quantity",
    "pay_by_cash", "pay_by_card", "pay_by_online", "pay_by_savings", "pos_amount",
    "amount_not_taxed", "vat",
}

# ── 指标中文说明（用于结果标注）─────────────────────────────────────
_METRIC_LABELS: dict[str, str] = {
    "turnover":                "总销售额",
    "actual_income":           "实际收入（含小费）",
    "actual_income_no_tips":   "实际收入（不含小费）",
    "tips_amount":             "小费总额",
    "order_quantity":          "订单总数",
    "average_order_price":     "平均订单金额",
    "customer_number":         "顾客总数",
    "dining_room_amount":      "堂食销售额",
    "togo_amount":             "外带销售额",
    "store_delivery_amount":   "门店配送销售额",
    "discount_amount":         "折扣总额",
    "discount_ratio":          "折扣比率",
    "member_quantity":         "会员订单数",
    "new_member_number":       "新注册会员数",
    "first_member_number":     "首次消费顾客数",
    "second_member_number":    "回头客数",
    "regular_member_number":   "常客数",
    "refund_amount":           "退款总额",
    "refunded_quantity":       "退款订单数",
    "pay_by_cash":             "现金支付金额",
    "pay_by_card":             "卡支付金额",
    "pay_by_online":           "在线支付金额",
    "pay_by_savings":          "储值支付金额",
    "pos_amount":              "POS卡支付金额",
    "amount_not_taxed":        "不含税销售额",
    "vat":                     "增值税",
}


# ── 日期解析工具 ─────────────────────────────────────────────────────

def _day_to_midnight_ms(d: date) -> int:
    """将 date 转为当天 00:00:00 UTC 的 13 位毫秒时间戳。"""
    dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_date(s: str) -> date:
    """
    解析日期字符串，支持：
    - ISO 格式：2024-03-01
    - 快捷词：today / yesterday / last_7_days / last_30_days /
              this_month / last_month
    返回 (start_date, end_date) 二元组；快捷词会返回一个范围。
    """
    s = s.strip().lower()
    today = date.today()

    shortcuts = {
        "today":       (today, today),
        "yesterday":   (today - timedelta(days=1), today - timedelta(days=1)),
        "last_7_days": (today - timedelta(days=6), today),
        "last_30_days":(today - timedelta(days=29), today),
        "this_month":  (today.replace(day=1), today),
        "last_month":  (
            (today.replace(day=1) - timedelta(days=1)).replace(day=1),
            today.replace(day=1) - timedelta(days=1),
        ),
    }
    if s in shortcuts:
        return shortcuts[s]  # type: ignore[return-value]

    # ISO 格式
    try:
        d = date.fromisoformat(s)
        return d, d
    except ValueError:
        raise ValueError(f"无法解析日期 '{s}'，支持格式：YYYY-MM-DD 或 today/yesterday/last_7_days/last_30_days/this_month/last_month")


def _resolve_range(date_start: str | None, date_end: str | None) -> tuple[int, int]:
    """返回 (range_start_ms, range_end_ms)，均为 13 位时间戳（当天凌晨）。"""
    today = date.today()

    if date_start:
        s_start, s_end = _parse_date(date_start)
    else:
        s_start = s_end = today  # 默认今天

    if date_end:
        _, e_end = _parse_date(date_end)
    else:
        e_end = s_end  # date_end 缺省时与 date_start 快捷词展开结果的 end 相同

    # rangeEnd 按 API 要求也取凌晨时间戳（表示"到该天结束前"）
    return _day_to_midnight_ms(s_start), _day_to_midnight_ms(e_end)


# ── Skill 主体 ───────────────────────────────────────────────────────

class AgentBISkill:
    descriptor = ToolDescriptor(
        name="agent_bi",
        description=(
            "门店 BI 数据查询工具。根据门店 ID、日期范围和指标列表，"
            "调用报表接口返回销售额、订单量、客单价、会员数等经营数据。"
            "日期支持 today/yesterday/last_7_days/this_month 等快捷词，"
            "也支持 YYYY-MM-DD 格式。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "string",
                    "description": "门店 ID（braId），不传则查所有门店汇总",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "指标列表，如 [\"turnover\", \"order_quantity\"]。"
                        "为空时默认返回：turnover/order_quantity/average_order_price/customer_number"
                    ),
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["SUM", "AVG", "MAX", "MIN", "COUNT"],
                    "description": "聚合函数，默认 SUM",
                },
                "date_start": {
                    "type": "string",
                    "description": (
                        "查询开始日期，如 \"2024-03-01\" 或 today/yesterday/"
                        "last_7_days/last_30_days/this_month/last_month"
                    ),
                },
                "date_end": {
                    "type": "string",
                    "description": "查询结束日期（含），省略时与 date_start 相同",
                },
            },
            "required": [],
        },
        source="skill",
        permission=PermissionLevel.READ,
        timeout_ms=10_000,
        tags=["bi", "report", "analytics", "sales", "store"],
    )

    def __init__(self) -> None:
        self._base_url = os.environ.get("AGENT_BI_BASE_URL", "http://localhost:8080").rstrip("/")
        self._token    = os.environ.get("AGENT_BI_TOKEN", "")

    async def execute(self, arguments: dict) -> dict[str, Any]:
        store_id    = arguments.get("store_id") or None
        raw_metrics = arguments.get("metrics") or []
        aggregation = (arguments.get("aggregation") or "SUM").upper()
        date_start  = arguments.get("date_start") or None
        date_end    = arguments.get("date_end") or None

        # 校验并过滤指标
        if raw_metrics:
            invalid = [m for m in raw_metrics if m not in _VALID_METRICS]
            if invalid:
                return {"error": f"无效指标: {invalid}，请从支持列表中选择"}
            metrics = raw_metrics
        else:
            metrics = _DEFAULT_METRICS

        # 解析时间范围
        try:
            range_start, range_end = _resolve_range(date_start, date_end)
        except ValueError as e:
            return {"error": str(e)}

        # 构建请求体
        payload: dict[str, Any] = {
            "metrics":     metrics,
            "aggregation": aggregation,
            "rangeStart":  range_start,
            "rangeEnd":    range_end,
        }
        if store_id:
            payload["braId"] = store_id

        # 发起 HTTP 请求
        try:
            result = await self._call_api(payload)
        except Exception as e:
            return {"error": f"API 请求失败: {e}"}

        # 解析响应
        code = result.get("code", -1)
        if code != 0:
            return {
                "error":   result.get("message", "API 返回错误"),
                "code":    code,
                "payload": payload,
            }

        raw_metrics_list: list[dict] = (result.get("data") or {}).get("metrics") or []

        # 附加中文标签，方便 LLM 直接呈现
        labeled: list[dict] = []
        for row in raw_metrics_list:
            labeled_row = {
                k: {"value": v, "label": _METRIC_LABELS.get(k, k)}
                for k, v in row.items()
            }
            labeled.append(labeled_row)

        return {
            "store_id":   store_id or "（全部门店）",
            "date_range": {
                "start": date_start or "today",
                "end":   date_end or date_start or "today",
            },
            "aggregation": aggregation,
            "metrics":     metrics,
            "data":        labeled,
            "raw":         raw_metrics_list,
        }

    async def _call_api(self, payload: dict) -> dict:
        """向 /v1/report/agent_bi 发起 POST 请求，返回响应 JSON。"""
        import json
        import urllib.request

        url     = f"{self._base_url}/v1/report/agent_bi"
        body    = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        # timeout=9s（略低于 skill timeout_ms=10000）
        with urllib.request.urlopen(req, timeout=9) as resp:
            return json.loads(resp.read().decode("utf-8"))
