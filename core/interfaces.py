"""
core/interfaces.py — 所有可替换模块的抽象接口（Protocol）

设计原则：
  - 每个接口是一个 Python Protocol，不继承任何具体类
  - 实现类只需结构匹配（duck typing），无需显式继承
  - 便于测试时用 Mock 替换任意模块
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from core.models import (
    AgentConfig, AgentTask, LLMResponse,
    MemoryEntry, MemoryType,
    Message, ToolDescriptor, ToolResult,
)


# ─────────────────────────────────────────────
# LLM Engine
# ─────────────────────────────────────────────

@runtime_checkable
class LLMEngine(Protocol):
    """
    语言模型推理引擎。
    可替换为：AnthropicEngine / OpenAIEngine / OllamaEngine / MockEngine
    """

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> LLMResponse:
        """单次同步推理，返回完整响应。"""
        ...

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> AsyncIterator[str]:
        """流式推理，yield 文本 delta。"""
        ...

    async def embed(self, text: str) -> list[float]:
        """文本向量化（记忆系统使用）。"""
        ...

    async def summarize(self, text: str, max_tokens: int) -> str:
        """将长文本压缩到 max_tokens 以内（上下文裁剪使用）。"""
        ...


# ─────────────────────────────────────────────
# Memory Store
# ─────────────────────────────────────────────

@runtime_checkable
class ShortTermMemory(Protocol):
    """
    短期记忆（工作记忆）。
    可替换为：RedisShortTermMemory / InMemoryShortTermMemory
    """

    async def save_task(self, task: AgentTask) -> None: ...
    async def load_task(self, task_id: str) -> AgentTask | None: ...
    async def delete_task(self, task_id: str) -> None: ...

    async def append_message(self, task_id: str, msg: Message) -> None: ...
    async def get_messages(self, task_id: str) -> list[Message]: ...

    async def set_scratchpad(self, task_id: str, key: str, value: Any) -> None: ...
    async def get_scratchpad(self, task_id: str, key: str) -> Any: ...


@runtime_checkable
class LongTermMemory(Protocol):
    """
    长期记忆（语义/情节/用户画像）。
    可替换为：QdrantLongTermMemory / InMemoryLongTermMemory
    """

    async def write(self, entry: MemoryEntry) -> None: ...

    async def search(
        self,
        user_id: str,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]: ...

    async def get_profile(self, user_id: str) -> dict[str, Any]: ...
    async def update_profile(self, user_id: str, data: dict[str, Any]) -> None: ...

    async def prune(self, user_id: str, max_items: int, score_threshold: float) -> int:
        """删除重要性低于阈值的旧记忆，返回删除条数。"""
        ...


# ─────────────────────────────────────────────
# Skill
# ─────────────────────────────────────────────

@runtime_checkable
class Skill(Protocol):
    """
    单个技能的执行接口。
    实现类需提供 descriptor 属性和 execute 方法。
    """

    @property
    def descriptor(self) -> ToolDescriptor: ...

    async def execute(self, arguments: dict[str, Any]) -> Any:
        """执行技能，返回任意可序列化结果。失败应抛出异常。"""
        ...


@runtime_checkable
class SkillRegistry(Protocol):
    """
    技能注册表，管理 Skill 的注册、发现和调用。
    可替换为：LocalSkillRegistry / RemoteSkillRegistry
    """

    def register(self, skill: Skill) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> Skill | None: ...
    def list_descriptors(self) -> list[ToolDescriptor]: ...

    async def call(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolResult: ...


# ─────────────────────────────────────────────
# MCP Connector
# ─────────────────────────────────────────────

@runtime_checkable
class MCPConnector(Protocol):
    """
    MCP 协议连接器，对接单个 MCP Server。
    可替换为：SSEMCPConnector / StdioMCPConnector / MockMCPConnector
    """

    @property
    def server_name(self) -> str: ...

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    async def list_tools(self) -> list[ToolDescriptor]: ...

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult: ...


@runtime_checkable
class MCPHub(Protocol):
    """
    MCP Hub，管理多个 MCPConnector。
    可替换为：DefaultMCPHub / MockMCPHub
    """

    def register_connector(self, connector: MCPConnector) -> None: ...
    def list_descriptors(self) -> list[ToolDescriptor]: ...

    async def call(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult: ...


# ─────────────────────────────────────────────
# Context Manager
# ─────────────────────────────────────────────

@runtime_checkable
class ContextManager(Protocol):
    """
    上下文管理器，负责在 token 预算内组装最优 Prompt。
    可替换为：PriorityContextManager / SimpleContextManager
    """

    async def build(
        self,
        task: AgentTask,
        tools: list[ToolDescriptor],
        long_term_memory: LongTermMemory,
        budget: int,
    ) -> list[Message]: ...


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

@runtime_checkable
class Orchestrator(Protocol):
    """
    编排引擎，驱动整个 Run Loop。
    可替换为：ReactOrchestrator / PlanExecuteOrchestrator
    """

    async def run(
        self, task: AgentTask, config: AgentConfig
    ) -> AsyncIterator[dict[str, Any]]:
        """
        执行任务，yield 进度事件（流式）。
        事件格式：
          {"type": "step",  "step": 1, "tool": "...", "status": "running"}
          {"type": "delta", "text": "..."}
          {"type": "done",  "task_id": "...", "usage": {...}}
          {"type": "error", "message": "..."}
        """
        ...
