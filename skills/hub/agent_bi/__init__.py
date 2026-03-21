"""
AgentBiSkill — 调用 BI 报表接口获取商家业务指标数据。

API 文档: /v1/report/agent_bi
- braId: 门店ID（可选）
- metrics: 指标字段列表
- aggregation: 聚合函数 SUM/AVG/MAX/MIN/COUNT
- rangeStart / rangeEnd: 13位毫秒时间戳

使用 AGENT_BI_API_URL 环境变量指定接口地址（默认 http://localhost:8080/v1/report/agent_bi）。
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


_METRIC_LABELS: dict[str, str] = {
    "turnover": "总销售额",
    "actual_income": "实际收入（含小费）",
    "actual_income_no_tips": "实际收入（不含小费）",
    "tips_amount": "小费总额",
    "order_quantity": "订单总数",
    "average_order_price": "平均订单金额",
    "customer_number": "顾客总数",
    "dining_room_amount": "堂食销售额",
    "togo_amount": "外带销售额",
    "store_delivery_amount": "门店配送销售额",
    "discount_amount": "折扣总额",
    "discount_ratio": "折扣比率",
    "member_quantity": "会员订单数",
    "new_member_number": "新注册会员数",
    "first_member_number": "首次消费顾客数",
    "second_member_number": "回头客数",
    "regular_member_number": "常客数",
    "refund_amount": "退款总额",
    "refunded_quantity": "退款订单数",
    "pay_by_cash": "现金支付",
    "pay_by_card": "卡支付",
    "pay_by_online": "在线支付",
    "pay_by_savings": "储值支付",
    "pos_amount": "POS卡支付",
    "amount_not_taxed": "不含税销售额",
    "vat": "增值税",
}

_VALID_METRICS = set(_METRIC_LABELS.keys())
_VALID_AGGREGATIONS = {"SUM", "AVG", "MAX", "MIN", "COUNT"}
_DEFAULT_API_URL = "http://localhost:8080/v1/report/agent_bi"


class AgentBiSkill:
    name = "agent_bi"

    def execute(self, action: str, **kwargs: Any) -> dict[str, Any]:
        if action == "query":
            return self._query(**kwargs)
        return {"error": f"Unknown action: {action!r}. Supported: query"}

    def _query(
        self,
        metrics: list[str],
        bra_id: str | None = None,
        aggregation: str = "SUM",
        range_start: int | None = None,
        range_end: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        # Validate
        invalid = [m for m in metrics if m not in _VALID_METRICS]
        if invalid:
            return {
                "error": f"无效指标: {invalid}",
                "valid_metrics": sorted(_VALID_METRICS),
            }
        if aggregation not in _VALID_AGGREGATIONS:
            return {
                "error": f"无效聚合函数: {aggregation!r}",
                "valid": sorted(_VALID_AGGREGATIONS),
            }

        payload: dict[str, Any] = {"metrics": metrics, "aggregation": aggregation}
        if bra_id:
            payload["braId"] = bra_id
        if range_start is not None:
            payload["rangeStart"] = range_start
        if range_end is not None:
            payload["rangeEnd"] = range_end

        try:
            api_url = os.environ.get("AGENT_BI_API_URL", _DEFAULT_API_URL)
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                api_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result: dict[str, Any] = json.loads(resp.read())

            if result.get("code") != 0:
                return {
                    "error": result.get("message", "API 返回错误"),
                    "code": result.get("code"),
                }

            raw_metrics: list[dict[str, Any]] = (
                result.get("data") or {}
            ).get("metrics", [])

            # Enrich with Chinese labels
            labeled: list[dict[str, Any]] = []
            for row in raw_metrics:
                labeled.append(
                    {
                        k: {"value": v, "label": _METRIC_LABELS.get(k, k)}
                        for k, v in row.items()
                    }
                )

            return {
                "success": True,
                "aggregation": aggregation,
                "bra_id": bra_id,
                "range_start": range_start,
                "range_end": range_end,
                "metrics": labeled,
                "raw_metrics": raw_metrics,
            }
        except urllib.error.URLError as exc:
            return {"error": f"网络错误: {exc.reason}", "type": "NetworkError"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "type": type(exc).__name__}
