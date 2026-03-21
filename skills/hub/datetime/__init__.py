"""
skills/hub/datetime/__init__.py — 日期时间工具 Skill
"""
from __future__ import annotations

import datetime
import zoneinfo

from core.models import PermissionLevel, ToolDescriptor

_DEFAULT_TZ = "Asia/Shanghai"


class DatetimeSkill:
    descriptor = ToolDescriptor(
        name="datetime_tool",
        description=(
            "查询当前日期时间、进行日期加减运算、计算两个日期之差、查询某天是星期几。"
            "支持时区转换，默认使用 Asia/Shanghai。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["now", "diff", "add", "weekday"],
                    "description": "now=当前时间, diff=两日期相差天数, add=日期加减, weekday=查询星期",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA 时区名，如 Asia/Shanghai、UTC、America/New_York",
                },
                "date1": {"type": "string", "description": "日期 YYYY-MM-DD（diff/weekday 用）"},
                "date2": {"type": "string", "description": "第二个日期 YYYY-MM-DD（diff 用）"},
                "days":  {"type": "integer", "description": "要加减的天数（add 用，负数为减）"},
                "base_date": {"type": "string", "description": "基准日期 YYYY-MM-DD（add 用，默认今天）"},
            },
            "required": ["action"],
        },
        source="skill",
        permission=PermissionLevel.READ,
        timeout_ms=3_000,
        tags=["datetime", "time", "timezone", "date"],
    )

    _WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    _WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    async def execute(self, arguments: dict) -> dict:
        action   = arguments["action"]
        tz_name  = arguments.get("timezone", _DEFAULT_TZ)

        try:
            tz = zoneinfo.ZoneInfo(tz_name)
        except zoneinfo.ZoneInfoNotFoundError:
            return {"error": f"未知时区: {tz_name}"}

        if action == "now":
            now = datetime.datetime.now(tz)
            return {
                "datetime":  now.strftime("%Y-%m-%d %H:%M:%S"),
                "date":      now.strftime("%Y-%m-%d"),
                "time":      now.strftime("%H:%M:%S"),
                "weekday_zh": self._WEEKDAYS_ZH[now.weekday()],
                "weekday_en": self._WEEKDAYS_EN[now.weekday()],
                "timezone":  tz_name,
                "timestamp": int(now.timestamp()),
            }

        if action == "diff":
            d1 = datetime.date.fromisoformat(arguments["date1"])
            d2 = datetime.date.fromisoformat(arguments["date2"])
            delta = (d2 - d1).days
            return {
                "date1": str(d1),
                "date2": str(d2),
                "days":  delta,
                "description": f"{d1} 到 {d2} 相差 {abs(delta)} 天"
                               + ("（date2 在 date1 之后）" if delta >= 0 else "（date2 在 date1 之前）"),
            }

        if action == "add":
            days      = arguments.get("days", 0)
            base_str  = arguments.get("base_date")
            base      = datetime.date.fromisoformat(base_str) if base_str else datetime.date.today()
            result    = base + datetime.timedelta(days=days)
            return {
                "base_date":   str(base),
                "days_added":  days,
                "result_date": str(result),
                "weekday_zh":  self._WEEKDAYS_ZH[result.weekday()],
                "weekday_en":  self._WEEKDAYS_EN[result.weekday()],
            }

        if action == "weekday":
            d = datetime.date.fromisoformat(arguments["date1"])
            return {
                "date":       str(d),
                "weekday_zh": self._WEEKDAYS_ZH[d.weekday()],
                "weekday_en": self._WEEKDAYS_EN[d.weekday()],
                "weekday_num": d.weekday() + 1,  # 1=周一 … 7=周日
            }

        return {"error": f"未知 action: {action}"}
