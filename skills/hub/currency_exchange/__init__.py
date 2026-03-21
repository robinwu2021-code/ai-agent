"""
skills/hub/currency_exchange/__init__.py — 汇率查询与货币换算 Skill

数据源：Frankfurter API（欧洲央行汇率，每日更新，免费无需 Key）
"""
from __future__ import annotations

import httpx

from core.models import PermissionLevel, ToolDescriptor

_API = "https://api.frankfurter.app/latest"
_CURRENCY_NAMES = {
    "USD": "美元", "EUR": "欧元", "CNY": "人民币", "JPY": "日元",
    "GBP": "英镑", "HKD": "港元", "KRW": "韩元", "AUD": "澳元",
    "CAD": "加元", "SGD": "新加坡元", "CHF": "瑞士法郎", "SEK": "瑞典克朗",
    "NOK": "挪威克朗", "MXN": "墨西哥比索", "INR": "印度卢比",
    "BRL": "巴西雷亚尔", "RUB": "俄罗斯卢布", "ZAR": "南非兰特",
}


class CurrencyExchangeSkill:
    descriptor = ToolDescriptor(
        name="currency_exchange",
        description=(
            "查询实时汇率并进行货币换算。数据来源：Frankfurter API（欧洲央行汇率，每日更新）。"
            "支持 USD、EUR、CNY、JPY、GBP、HKD 等主流货币。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "要换算的金额，默认 1",
                },
                "from_currency": {
                    "type": "string",
                    "description": "源货币代码，如 USD、EUR、CNY",
                },
                "to_currency": {
                    "type": "string",
                    "description": "目标货币代码，支持逗号分隔多个，如 CNY,JPY,EUR",
                },
            },
            "required": ["from_currency", "to_currency"],
        },
        source="skill",
        permission=PermissionLevel.READ,
        timeout_ms=10_000,
        tags=["currency", "exchange", "finance", "money"],
    )

    async def execute(self, arguments: dict) -> dict:
        amount       = float(arguments.get("amount", 1))
        from_cur     = arguments["from_currency"].upper().strip()
        to_raw       = arguments["to_currency"].upper().strip()
        to_currencies = [c.strip() for c in to_raw.split(",") if c.strip()]

        params: dict = {"from": from_cur, "to": ",".join(to_currencies)}
        if amount != 1:
            params["amount"] = amount

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(_API, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            return {"error": f"汇率 API 请求失败: {e}"}

        rates  = data.get("rates", {})
        date   = data.get("date", "")
        result = {
            "date":          date,
            "from":          from_cur,
            "from_name":     _CURRENCY_NAMES.get(from_cur, from_cur),
            "amount":        amount,
            "rates":         {},
        }
        for cur, rate in rates.items():
            result["rates"][cur] = {
                "rate":    round(rate / amount, 6),          # 单位汇率
                "result":  round(rate, 4),                   # 换算结果
                "name":    _CURRENCY_NAMES.get(cur, cur),
                "display": f"{amount} {from_cur} = {round(rate, 4)} {cur}",
            }
        return result
