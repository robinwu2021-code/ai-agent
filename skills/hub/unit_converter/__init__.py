"""
skills/hub/unit_converter/__init__.py — 单位换算 Skill（本地计算，无需网络）
"""
from __future__ import annotations

from core.models import PermissionLevel, ToolDescriptor

# ── 换算表：所有单位统一换算到 SI 基准单位 ─────────────────────────
# key: 单位别名（小写），value: (换算到基准单位的倍数, 类别, 显示名)

_UNITS: dict[str, tuple[float, str, str]] = {
    # 长度 → 米 (m)
    "m": (1, "length", "米"), "meter": (1, "length", "米"), "meters": (1, "length", "米"),
    "km": (1000, "length", "千米"), "kilometer": (1000, "length", "千米"),
    "cm": (0.01, "length", "厘米"), "mm": (0.001, "length", "毫米"),
    "mile": (1609.344, "length", "英里"), "miles": (1609.344, "length", "英里"),
    "yard": (0.9144, "length", "码"), "yards": (0.9144, "length", "码"),
    "ft": (0.3048, "length", "英尺"), "foot": (0.3048, "length", "英尺"), "feet": (0.3048, "length", "英尺"),
    "inch": (0.0254, "length", "英寸"), "inches": (0.0254, "length", "英寸"), "in": (0.0254, "length", "英寸"),
    "nm": (1e-9, "length", "纳米"), "um": (1e-6, "length", "微米"),
    "nautical_mile": (1852, "length", "海里"), "nmi": (1852, "length", "海里"),

    # 重量 → 千克 (kg)
    "kg": (1, "weight", "千克"), "kilogram": (1, "weight", "千克"),
    "g": (0.001, "weight", "克"), "gram": (0.001, "weight", "克"),
    "mg": (1e-6, "weight", "毫克"), "t": (1000, "weight", "吨"), "ton": (1000, "weight", "吨"),
    "lb": (0.453592, "weight", "磅"), "lbs": (0.453592, "weight", "磅"), "pound": (0.453592, "weight", "磅"),
    "oz": (0.0283495, "weight", "盎司"), "ounce": (0.0283495, "weight", "盎司"),
    "jin": (0.5, "weight", "斤"), "liang": (0.05, "weight", "两"),

    # 面积 → 平方米 (m²)
    "m2": (1, "area", "平方米"), "sqm": (1, "area", "平方米"),
    "km2": (1e6, "area", "平方千米"), "ha": (10000, "area", "公顷"),
    "cm2": (1e-4, "area", "平方厘米"),
    "ft2": (0.092903, "area", "平方英尺"), "sqft": (0.092903, "area", "平方英尺"),
    "acre": (4046.86, "area", "英亩"),
    "mu": (666.667, "area", "亩"),

    # 体积 → 升 (L)
    "l": (1, "volume", "升"), "liter": (1, "volume", "升"), "litre": (1, "volume", "升"),
    "ml": (0.001, "volume", "毫升"), "cl": (0.01, "volume", "厘升"),
    "m3": (1000, "volume", "立方米"), "cm3": (0.001, "volume", "立方厘米"),
    "gallon": (3.78541, "volume", "加仑（美）"), "gal": (3.78541, "volume", "加仑（美）"),
    "fl_oz": (0.0295735, "volume", "液盎司"), "cup": (0.236588, "volume", "杯"),
    "pint": (0.473176, "volume", "品脱"), "quart": (0.946353, "volume", "夸脱"),

    # 速度 → 米/秒 (m/s)
    "ms": (1, "speed", "米/秒"), "m/s": (1, "speed", "米/秒"),
    "kmh": (1/3.6, "speed", "千米/时"), "km/h": (1/3.6, "speed", "千米/时"),
    "mph": (0.44704, "speed", "英里/时"), "knot": (0.514444, "speed", "节"),
}

# 温度单独处理（非线性）
_TEMP_UNITS = {"celsius", "c", "°c", "fahrenheit", "f", "°f", "kelvin", "k"}


def _convert_temp(value: float, from_u: str, to_u: str) -> float:
    # normalize to celsius first
    fu = from_u.lower().lstrip("°")
    tu = to_u.lower().lstrip("°")
    if fu in ("c", "celsius"):
        celsius = value
    elif fu in ("f", "fahrenheit"):
        celsius = (value - 32) * 5 / 9
    elif fu in ("k", "kelvin"):
        celsius = value - 273.15
    else:
        raise ValueError(f"未知温度单位: {from_u}")

    if tu in ("c", "celsius"):
        return celsius
    elif tu in ("f", "fahrenheit"):
        return celsius * 9 / 5 + 32
    elif tu in ("k", "kelvin"):
        return celsius + 273.15
    else:
        raise ValueError(f"未知温度单位: {to_u}")


class UnitConverterSkill:
    descriptor = ToolDescriptor(
        name="unit_converter",
        description=(
            "单位换算工具。支持长度、重量、温度、面积、体积、速度六大类别，本地计算，无需网络。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "value":     {"type": "number", "description": "要换算的数值"},
                "from_unit": {"type": "string", "description": "源单位，如 km、kg、celsius、m2、liter、kmh"},
                "to_unit":   {"type": "string", "description": "目标单位，如 mile、lb、fahrenheit、ft2、gallon、mph"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        source="skill",
        permission=PermissionLevel.READ,
        timeout_ms=1_000,
        tags=["unit", "convert", "math", "measurement"],
    )

    async def execute(self, arguments: dict) -> dict:
        value    = float(arguments["value"])
        from_raw = arguments["from_unit"].lower().strip()
        to_raw   = arguments["to_unit"].lower().strip()

        # 温度
        if from_raw.lstrip("°") in ("c", "celsius", "f", "fahrenheit", "k", "kelvin"):
            try:
                result = _convert_temp(value, from_raw, to_raw)
                return {
                    "input":  f"{value} {from_raw}",
                    "output": f"{round(result, 4)} {to_raw}",
                    "value":  round(result, 4),
                    "category": "temperature",
                }
            except ValueError as e:
                return {"error": str(e)}

        from_info = _UNITS.get(from_raw)
        to_info   = _UNITS.get(to_raw)

        if not from_info:
            return {"error": f"未知单位: {from_raw}，支持: {', '.join(sorted(_UNITS))}"}
        if not to_info:
            return {"error": f"未知单位: {to_raw}，支持: {', '.join(sorted(_UNITS))}"}

        from_factor, from_cat, from_name = from_info
        to_factor,   to_cat,   to_name   = to_info

        if from_cat != to_cat:
            return {"error": f"单位类别不匹配: {from_raw}({from_cat}) ↔ {to_raw}({to_cat})"}

        si_value = value * from_factor
        result   = si_value / to_factor

        return {
            "input":    f"{value} {from_name}",
            "output":   f"{round(result, 6)} {to_name}",
            "value":    round(result, 6),
            "category": from_cat,
        }
