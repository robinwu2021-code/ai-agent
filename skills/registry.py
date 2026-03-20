"""
skills/registry.py — Skill 注册表与内置技能

包含：
  - LocalSkillRegistry   本地注册表实现
  - 内置 Skill 集合
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any

import structlog

from core.models import (
    PermissionLevel, ToolDescriptor, ToolResult,
)

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

class LocalSkillRegistry:
    """本地 Skill 注册表，支持动态注册和调用。"""

    def __init__(self, timeout_multiplier: float = 1.0) -> None:
        self._skills: dict[str, Any] = {}
        self._timeout_multiplier = timeout_multiplier

    def register(self, skill: Any) -> None:
        name = skill.descriptor.name
        self._skills[name] = skill
        log.info("skill.registered", name=name)

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Any | None:
        return self._skills.get(name)

    def list_descriptors(self) -> list[ToolDescriptor]:
        return [s.descriptor for s in self._skills.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        skill = self._skills.get(name)
        if not skill:
            return ToolResult(
                tool_call_id="",
                tool_name=name,
                content=None,
                error=f"Skill '{name}' not found",
            )
        timeout = skill.descriptor.timeout_ms / 1000 * self._timeout_multiplier
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(skill.execute(arguments), timeout=timeout)
            duration = int((time.monotonic() - start) * 1000)
            log.info("skill.call.success", name=name, duration_ms=duration)
            return ToolResult(
                tool_call_id="",
                tool_name=name,
                content=result,
                duration_ms=duration,
            )
        except asyncio.TimeoutError:
            return ToolResult(tool_call_id="", tool_name=name, content=None,
                              error=f"Skill timeout after {timeout}s")
        except Exception as e:
            log.error("skill.call.error", name=name, error=str(e))
            return ToolResult(tool_call_id="", tool_name=name, content=None, error=str(e))


# ─────────────────────────────────────────────
# Built-in Skills
# ─────────────────────────────────────────────

class PythonExecutorSkill:
    """在子进程中安全执行 Python 代码片段。"""

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="execute_python",
            description="在沙箱子进程中执行 Python 代码，返回 stdout 和 stderr。适合数据处理、计算、格式转换。",
            input_schema={
                "type": "object",
                "properties": {
                    "code":    {"type": "string", "description": "Python 代码"},
                    "timeout": {"type": "integer", "description": "超时秒数", "default": 10},
                },
                "required": ["code"],
            },
            source="skill",
            permission=PermissionLevel.EXEC,
            timeout_ms=30_000,
            tags=["code", "python", "compute"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, str]:
        code    = arguments["code"]
        timeout = int(arguments.get("timeout", 10))
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            raise RuntimeError(f"Code execution timeout after {timeout}s")


class FileReadSkill:
    """读取本地文本文件内容。"""

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        self._allowed = [Path(d).resolve() for d in (allowed_dirs or ["/tmp"])]

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="read_file",
            description="读取文件内容并返回文本。支持 txt/md/json/csv 等文本格式。",
            input_schema={
                "type": "object",
                "properties": {
                    "path":       {"type": "string", "description": "文件路径"},
                    "max_chars":  {"type": "integer", "description": "最大字符数", "default": 10000},
                },
                "required": ["path"],
            },
            source="skill",
            permission=PermissionLevel.READ,
            timeout_ms=5_000,
            tags=["file", "read"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = Path(arguments["path"]).resolve()
        max_chars = int(arguments.get("max_chars", 10000))
        # 权限检查
        if not any(str(path).startswith(str(d)) for d in self._allowed):
            raise PermissionError(f"Access denied: {path}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = path.read_text(errors="replace")
        truncated = len(content) > max_chars
        return {
            "content":   content[:max_chars],
            "truncated": truncated,
            "size":      path.stat().st_size,
        }


class FileWriteSkill:
    """向文件写入内容（覆盖或追加）。"""

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        self._allowed = [Path(d).resolve() for d in (allowed_dirs or ["/tmp"])]

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="write_file",
            description="向文件写入文本内容，可选覆盖或追加模式。",
            input_schema={
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                    "mode":    {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
                },
                "required": ["path", "content"],
            },
            source="skill",
            permission=PermissionLevel.WRITE,
            timeout_ms=5_000,
            tags=["file", "write"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = Path(arguments["path"]).resolve()
        if not any(str(path).startswith(str(d)) for d in self._allowed):
            raise PermissionError(f"Access denied: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if arguments.get("mode") == "append" else "w"
        path.write_text(arguments["content"]) if mode == "w" else \
            open(path, "a").write(arguments["content"])
        return {"path": str(path), "bytes_written": len(arguments["content"].encode())}


class WebSearchSkill:
    """通过搜索 API 获取网页摘要（stub，生产替换为真实搜索 API）。"""

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="web_search",
            description="搜索互联网获取当前信息。返回最相关的搜索结果摘要。",
            input_schema={
                "type": "object",
                "properties": {
                    "query":    {"type": "string", "description": "搜索关键词"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            source="skill",
            permission=PermissionLevel.NETWORK,
            timeout_ms=15_000,
            tags=["search", "web", "internet"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # Stub：生产环境接入 Tavily / SerpAPI / Bing Search
        query = arguments["query"]
        log.info("web_search.stub", query=query)
        return {
            "query":   query,
            "results": [
                {"title": f"搜索结果 {i+1}: {query}", "snippet": "（生产环境请接入真实搜索 API）", "url": f"https://example.com/{i}"}
                for i in range(arguments.get("max_results", 3))
            ],
        }


class MemoryQuerySkill:
    """允许 LLM 主动查询长期记忆。"""

    def __init__(self, long_term_memory: Any, user_id_getter: Any) -> None:
        self._ltm = long_term_memory
        self._get_user_id = user_id_getter  # 运行时从 task context 获取

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="query_memory",
            description="查询用户的长期记忆，获取历史经验、偏好和已知事实。",
            input_schema={
                "type": "object",
                "properties": {
                    "query":   {"type": "string", "description": "查询内容"},
                    "user_id": {"type": "string", "description": "用户 ID"},
                    "top_k":   {"type": "integer", "default": 3},
                },
                "required": ["query", "user_id"],
            },
            source="skill",
            permission=PermissionLevel.READ,
            timeout_ms=5_000,
            tags=["memory", "recall"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        results = await self._ltm.search(
            user_id=arguments["user_id"],
            query=arguments["query"],
            top_k=int(arguments.get("top_k", 3)),
        )
        return {
            "memories": [
                {"text": e.text, "type": e.type.value, "importance": e.importance}
                for e in results
            ]
        }
