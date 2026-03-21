"""
skills/hub/text_translate/__init__.py — 文本翻译 Skill

数据源：MyMemory 免费翻译 API（无需 API Key，每天 5000 词限额）
备用：LibreTranslate 公共实例
"""
from __future__ import annotations

import urllib.parse

import httpx

from core.models import PermissionLevel, ToolDescriptor

_MYMEMORY_API = "https://api.mymemory.translated.net/get"

_LANG_NAMES = {
    "zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文",
    "fr": "法文", "de": "德文", "es": "西班牙文", "pt": "葡萄牙文",
    "it": "意大利文", "ru": "俄文", "ar": "阿拉伯文", "th": "泰文",
    "vi": "越南文", "id": "印度尼西亚文", "nl": "荷兰文",
}


class TextTranslateSkill:
    descriptor = ToolDescriptor(
        name="text_translate",
        description=(
            "文本翻译工具。支持中英日韩法德西等主流语言互译，"
            "source_lang 留空时自动检测源语言。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要翻译的文本",
                },
                "target_lang": {
                    "type": "string",
                    "description": (
                        "目标语言代码，如 zh（中文）、en（英文）、"
                        "ja（日文）、ko（韩文）、fr（法文）、de（德文）"
                    ),
                },
                "source_lang": {
                    "type": "string",
                    "description": "源语言代码，留空自动检测",
                },
            },
            "required": ["text", "target_lang"],
        },
        source="skill",
        permission=PermissionLevel.READ,
        timeout_ms=15_000,
        tags=["translate", "language", "text", "nlp"],
    )

    async def execute(self, arguments: dict) -> dict:
        text        = arguments["text"].strip()
        target_lang = arguments["target_lang"].lower().strip()
        source_lang = arguments.get("source_lang", "").lower().strip() or "autodetect"

        langpair = f"{source_lang}|{target_lang}"

        params = {
            "q":        text,
            "langpair": langpair,
        }

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.get(_MYMEMORY_API, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            return {"error": f"翻译 API 请求失败: {e}"}

        status = data.get("responseStatus")
        if status != 200:
            return {"error": f"翻译失败（状态 {status}）: {data.get('responseDetails', '')}"}

        translated = data["responseData"]["translatedText"]
        detected   = data["responseData"].get("detectedLanguage", source_lang)

        return {
            "original":      text,
            "translated":    translated,
            "source_lang":   detected or source_lang,
            "target_lang":   target_lang,
            "target_name":   _LANG_NAMES.get(target_lang, target_lang),
            "source_name":   _LANG_NAMES.get((detected or source_lang).split("-")[0], detected or source_lang),
        }
