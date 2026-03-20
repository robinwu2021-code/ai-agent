"""
mcp/hub.py — MCP 连接器 Hub 与 SSE/Stdio 实现

包含：
  - DefaultMCPHub      多 Server 管理中心
  - SSEMCPConnector    HTTP SSE 传输（生产用）
  - MockMCPConnector   测试用 Mock
"""
from __future__ import annotations

import json
import time
from typing import Any

import structlog

from core.models import PermissionLevel, ToolDescriptor, ToolResult

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Hub
# ─────────────────────────────────────────────

class DefaultMCPHub:
    """
    管理多个 MCP Connector，提供统一调用入口。
    负责会话池（每个 Connector 按需 connect）、权限检查和调用审计。
    """

    def __init__(self) -> None:
        self._connectors: dict[str, Any] = {}
        self._audit_log:  list[dict]     = []  # 生产替换为持久化审计日志

    def register_connector(self, connector: Any) -> None:
        self._connectors[connector.server_name] = connector
        log.info("mcp.connector.registered", server=connector.server_name)

    def list_descriptors(self) -> list[ToolDescriptor]:
        """收集所有已连接 Server 的工具列表（懒加载）。"""
        result = []
        for conn in self._connectors.values():
            try:
                # 同步调用已缓存的描述符
                if hasattr(conn, "_cached_tools"):
                    result.extend(conn._cached_tools)
            except Exception as e:
                log.warning("mcp.list_tools.error", server=conn.server_name, error=str(e))
        return result

    async def refresh_all(self) -> None:
        """主动刷新所有 Connector 的工具列表。"""
        for conn in self._connectors.values():
            try:
                await conn.connect()
                tools = await conn.list_tools()
                conn._cached_tools = tools
                log.info("mcp.tools.refreshed", server=conn.server_name, count=len(tools))
            except Exception as e:
                log.error("mcp.refresh.error", server=conn.server_name, error=str(e))

    async def call(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        conn = self._connectors.get(server_name)
        if not conn:
            return ToolResult(tool_call_id="", tool_name=tool_name,
                              content=None, error=f"MCP Server '{server_name}' not registered")
        start = time.monotonic()
        try:
            result = await conn.call_tool(tool_name, arguments)
        except Exception as e:
            result = ToolResult(tool_call_id="", tool_name=tool_name,
                                content=None, error=str(e))
        result.duration_ms = int((time.monotonic() - start) * 1000)
        # 审计
        self._audit_log.append({
            "server": server_name, "tool": tool_name,
            "args": arguments, "error": result.error,
            "duration_ms": result.duration_ms,
        })
        return result

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)


# ─────────────────────────────────────────────
# SSE MCP Connector
# ─────────────────────────────────────────────

class SSEMCPConnector:
    """
    通过 HTTP SSE 协议对接 MCP Server。
    参考：https://modelcontextprotocol.io/docs/concepts/transports
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self._name          = name
        self._url           = url
        self._headers       = headers or {}
        self._allowed_tools = set(allowed_tools) if allowed_tools else None
        self._cached_tools: list[ToolDescriptor] = []
        self._session       = None

    @property
    def server_name(self) -> str:
        return self._name

    async def connect(self) -> None:
        try:
            import httpx
            self._session = httpx.AsyncClient(headers=self._headers, timeout=30)
            log.info("mcp.sse.connected", server=self._name, url=self._url)
        except ImportError:
            raise RuntimeError("请安装 httpx：pip install httpx")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None

    async def list_tools(self) -> list[ToolDescriptor]:
        if not self._session:
            await self.connect()
        resp = await self._session.post(
            f"{self._url}/tools/list", json={}
        )
        resp.raise_for_status()
        data = resp.json()
        tools = []
        for t in data.get("tools", []):
            if self._allowed_tools and t["name"] not in self._allowed_tools:
                continue
            tools.append(ToolDescriptor(
                name=f"{self._name}__{t['name']}",  # 命名空间前缀
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object"}),
                source="mcp",
                timeout_ms=30_000,
                tags=["mcp", self._name],
            ))
        self._cached_tools = tools
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        if not self._session:
            await self.connect()
        # 去掉命名空间前缀
        real_name = tool_name.removeprefix(f"{self._name}__")
        if self._allowed_tools and real_name not in self._allowed_tools:
            return ToolResult(tool_call_id="", tool_name=tool_name,
                              content=None, error="Tool not in allowlist")
        resp = await self._session.post(
            f"{self._url}/tools/call",
            json={"name": real_name, "arguments": arguments},
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        # 提取文本内容
        text_parts = [c["text"] for c in content if c.get("type") == "text"]
        return ToolResult(
            tool_call_id="",
            tool_name=tool_name,
            content="\n".join(text_parts) if text_parts else data,
        )


# ─────────────────────────────────────────────
# Mock MCP Connector（测试）
# ─────────────────────────────────────────────

class MockMCPConnector:
    """预设响应的 Mock Connector，用于单元测试。"""

    def __init__(self, name: str, tools: list[ToolDescriptor], responses: dict[str, Any] | None = None) -> None:
        self._name     = name
        self._tools    = tools
        self._responses = responses or {}
        self._cached_tools = tools
        self.call_history: list[dict] = []

    @property
    def server_name(self) -> str:
        return self._name

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def list_tools(self) -> list[ToolDescriptor]:
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        self.call_history.append({"tool": tool_name, "args": arguments})
        content = self._responses.get(tool_name, {"mock": True, "tool": tool_name})
        return ToolResult(tool_call_id="", tool_name=tool_name, content=content)
