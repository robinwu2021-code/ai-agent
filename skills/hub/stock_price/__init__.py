"""
skills/hub/stock_price/__init__.py — 股票行情查询 Skill

数据源：Yahoo Finance 公共 API（免费，无需 Key）
A 股：上交所 600036.SS / 深交所 000001.SZ
港股：0700.HK
美股：AAPL、TSLA、MSFT
"""
from __future__ import annotations

import httpx

from core.models import PermissionLevel, ToolDescriptor

_YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_YF_QUOTE = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
_HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class StockPriceSkill:
    descriptor = ToolDescriptor(
        name="stock_price",
        description=(
            "查询股票/ETF 实时或当日行情，数据来源 Yahoo Finance（免费）。"
            "A 股代码加后缀：上交所 .SS（如 600036.SS），深交所 .SZ（如 000001.SZ）；"
            "港股后缀 .HK（如 0700.HK）；美股直接输入代码（如 AAPL）。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，如 AAPL、TSLA、0700.HK、600036.SS、000001.SZ",
                },
                "info": {
                    "type": "string",
                    "enum": ["price", "detail"],
                    "description": "price=仅价格（快速），detail=详细信息，默认 price",
                },
            },
            "required": ["symbol"],
        },
        source="skill",
        permission=PermissionLevel.READ,
        timeout_ms=15_000,
        tags=["stock", "finance", "market", "price"],
    )

    async def execute(self, arguments: dict) -> dict:
        symbol = arguments["symbol"].upper().strip()
        mode   = arguments.get("info", "price")

        url = _YF_CHART.format(symbol=symbol)
        params = {"range": "1d", "interval": "1m", "includePrePost": "false"}

        try:
            async with httpx.AsyncClient(timeout=12.0, headers=_HEADERS) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            return {"error": f"行情请求失败: {e}"}

        try:
            result_data = data["chart"]["result"]
            if not result_data:
                err = data["chart"].get("error", {})
                return {"error": f"未找到股票 {symbol}：{err.get('description', '无数据')}"}

            meta = result_data[0]["meta"]
            cur_price  = meta.get("regularMarketPrice")
            prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
            currency   = meta.get("currency", "")
            long_name  = meta.get("longName") or meta.get("shortName", symbol)
            exch       = meta.get("exchangeName", "")

            change     = round(cur_price - prev_close, 4) if cur_price and prev_close else None
            change_pct = round(change / prev_close * 100, 2) if change and prev_close else None
            trend      = ("↑" if change > 0 else "↓" if change < 0 else "—") if change is not None else "—"

            result = {
                "symbol":       symbol,
                "name":         long_name,
                "exchange":     exch,
                "currency":     currency,
                "price":        cur_price,
                "prev_close":   prev_close,
                "change":       change,
                "change_pct":   f"{change_pct:+.2f}%" if change_pct is not None else None,
                "trend":        trend,
                "display":      f"{long_name}（{symbol}）  {cur_price} {currency}  {trend} {change_pct:+.2f}%"
                                if change_pct is not None else f"{long_name}  {cur_price} {currency}",
            }

            if mode == "detail":
                result.update({
                    "day_high":    meta.get("regularMarketDayHigh"),
                    "day_low":     meta.get("regularMarketDayLow"),
                    "open":        meta.get("regularMarketOpen"),
                    "volume":      meta.get("regularMarketVolume"),
                    "market_cap":  meta.get("marketCap"),
                    "52w_high":    meta.get("fiftyTwoWeekHigh"),
                    "52w_low":     meta.get("fiftyTwoWeekLow"),
                    "timezone":    meta.get("timezone"),
                })

            return result

        except (KeyError, IndexError, TypeError) as e:
            return {"error": f"解析行情数据失败: {e}，原始: {str(data)[:200]}"}
