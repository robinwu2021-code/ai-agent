"""
skills/builtins.py — 扩展内置 Skill 集合

包含：
  - HttpRequestSkill       发起 HTTP 请求
  - JsonProcessorSkill     JSON 数据提取与转换
  - SummarizerSkill        使用 LLM 压缩长文本
  - CalculatorSkill        安全数学表达式计算
"""
from __future__ import annotations

import ast
import json
import operator
from typing import Any

from core.models import PermissionLevel, ToolDescriptor


class HttpRequestSkill:
    """发起 HTTP GET/POST 请求，返回响应体。"""

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="http_request",
            description="发起 HTTP 请求到指定 URL，返回响应内容。适合调用 REST API 或抓取网页内容。",
            input_schema={
                "type": "object",
                "properties": {
                    "url":     {"type": "string"},
                    "method":  {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
                    "headers": {"type": "object", "default": {}},
                    "body":    {"type": "object", "description": "POST/PUT 请求体（JSON）"},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["url"],
            },
            source="skill",
            permission=PermissionLevel.NETWORK,
            timeout_ms=20_000,
            tags=["http", "api", "network"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("请安装 httpx：pip install httpx")

        url     = arguments["url"]
        method  = arguments.get("method", "GET").upper()
        headers = arguments.get("headers", {})
        body    = arguments.get("body")
        timeout = int(arguments.get("timeout", 10))

        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "POST":
                resp = await client.post(url, json=body, headers=headers)
            elif method == "PUT":
                resp = await client.put(url, json=body, headers=headers)
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

        # 尝试解析 JSON，否则返回文本
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_data = resp.json()
            except Exception:
                body_data = resp.text
        else:
            body_data = resp.text[:5000]  # 截断超长 HTML

        return {
            "status_code": resp.status_code,
            "body":        body_data,
            "headers":     dict(resp.headers),
        }


class JsonProcessorSkill:
    """
    用 jq 风格路径提取或转换 JSON 数据。
    支持简单点路径（a.b.c）和列表索引（items[0].name）。
    """

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="json_extract",
            description="从 JSON 数据中提取字段或做简单转换。支持点路径（a.b.c）和列表索引（items[0]）。",
            input_schema={
                "type": "object",
                "properties": {
                    "data":  {"description": "JSON 数据（对象或数组）"},
                    "path":  {"type": "string", "description": "提取路径，如 'user.profile.name' 或 'results[0].score'"},
                    "keys":  {"type": "array",  "items": {"type": "string"}, "description": "提取多个顶层 key"},
                },
                "required": ["data"],
            },
            source="skill",
            permission=PermissionLevel.READ,
            timeout_ms=3_000,
            tags=["json", "transform", "data"],
        )

    async def execute(self, arguments: dict[str, Any]) -> Any:
        data = arguments["data"]
        if isinstance(data, str):
            data = json.loads(data)

        # 多 key 提取
        if keys := arguments.get("keys"):
            return {k: data.get(k) for k in keys if isinstance(data, dict)}

        # 路径提取
        path = arguments.get("path", "")
        if not path:
            return data

        current = data
        for part in path.replace("][", ".").replace("[", ".").replace("]", "").split("."):
            if not part:
                continue
            try:
                if isinstance(current, list):
                    current = current[int(part)]
                elif isinstance(current, dict):
                    current = current[part]
                else:
                    raise KeyError(f"Cannot navigate into {type(current).__name__} with key '{part}'")
            except (KeyError, IndexError, ValueError) as e:
                raise ValueError(f"Path '{path}' failed at '{part}': {e}")
        return current


class CalculatorSkill:
    """
    安全数学表达式求值，支持基本运算和常用数学函数。
    使用 AST 解析，不执行任意代码。
    """

    _SAFE_OPS = {
        ast.Add:  operator.add,
        ast.Sub:  operator.sub,
        ast.Mult: operator.mul,
        ast.Div:  operator.truediv,
        ast.Pow:  operator.pow,
        ast.Mod:  operator.mod,
        ast.USub: operator.neg,
    }
    _SAFE_FUNCS = {
        "abs": abs, "round": round, "min": min, "max": max,
        "int": int, "float": float,
    }

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="calculator",
            description="安全地计算数学表达式，支持 +/-/*/÷/^ 和 abs/round/min/max 函数。",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '2 ** 10 + round(3.14159, 2)'"},
                },
                "required": ["expression"],
            },
            source="skill",
            permission=PermissionLevel.READ,
            timeout_ms=1_000,
            tags=["math", "calculate"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expr = arguments["expression"]
        try:
            tree   = ast.parse(expr, mode="eval")
            result = self._eval_node(tree.body)
            return {"expression": expr, "result": result}
        except Exception as e:
            raise ValueError(f"计算失败: {e}")

    def _eval_node(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op_fn = self._SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = self._SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator")
            return op_fn(self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            fn = self._SAFE_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"Function '{node.func.id}' not allowed")
            args = [self._eval_node(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class SummarizerSkill:
    """
    使用 LLM 对长文本进行摘要。
    需要注入 LLM Engine。
    """

    def __init__(self, llm_engine: Any) -> None:
        self._llm = llm_engine

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="summarize_text",
            description="将较长的文本摘要压缩为简洁的要点。适合处理文章、报告、日志等长内容。",
            input_schema={
                "type": "object",
                "properties": {
                    "text":       {"type": "string", "description": "需要摘要的原文"},
                    "max_tokens": {"type": "integer", "default": 300, "description": "目标摘要长度（token）"},
                    "style":      {"type": "string", "enum": ["bullet", "paragraph"], "default": "paragraph"},
                },
                "required": ["text"],
            },
            source="skill",
            permission=PermissionLevel.NETWORK,
            timeout_ms=30_000,
            tags=["summarize", "text", "llm"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text       = arguments["text"]
        max_tokens = int(arguments.get("max_tokens", 300))
        style      = arguments.get("style", "paragraph")

        prompt = (
            f"请将以下内容摘要为{'要点列表' if style == 'bullet' else '简洁段落'}，"
            f"控制在 {max_tokens} token 以内：\n\n{text}"
        )
        summary = await self._llm.summarize(prompt, max_tokens=max_tokens)
        return {"summary": summary, "original_length": len(text)}
