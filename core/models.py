"""
core/models.py — 核心领域模型
所有模块共享的数据结构，无任何业务逻辑依赖。
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TaskStatus(str, Enum):
    PLANNING  = "planning"
    RUNNING   = "running"
    DONE      = "done"
    ERROR     = "error"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"

class MemoryType(str, Enum):
    SEMANTIC  = "semantic"   # 向量化语义记忆
    EPISODIC  = "episodic"   # 历史任务摘要
    PROFILE   = "profile"    # 用户画像/偏好

class MessageRole(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
    TOOL      = "tool"


# ─────────────────────────────────────────────
# Message & Tool Call
# ─────────────────────────────────────────────

class ToolCall(BaseModel):
    id:        str = Field(default_factory=lambda: f"tc_{uuid.uuid4().hex[:8]}")
    tool_name: str
    arguments: dict[str, Any]

class ToolResult(BaseModel):
    tool_call_id: str
    tool_name:    str
    content:      Any
    error:        str | None = None
    duration_ms:  int = 0

class Message(BaseModel):
    role:        MessageRole
    content:     str | None = None
    tool_calls:  list[ToolCall]  = Field(default_factory=list)
    tool_result: ToolResult | None = None
    created_at:  datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Task & Plan
# ─────────────────────────────────────────────

class AgentInput(BaseModel):
    text:  str
    files: list[dict[str, str]] = Field(default_factory=list)  # [{name, url}]
    metadata: dict[str, Any]   = Field(default_factory=dict)

class TaskStep(BaseModel):
    id:        str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:6]}")
    goal:      str
    tool_hint: str | None = None   # 预期工具（可选，LLM 可忽略）
    depends:   list[str]  = Field(default_factory=list)
    result:    Any        = None
    error:     str | None = None
    status:    StepStatus = StepStatus.PENDING

class AgentTask(BaseModel):
    id:         str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    session_id: str
    user_id:    str
    input:      AgentInput
    plan:       list[TaskStep] = Field(default_factory=list)
    step_idx:   int = 0
    history:    list[Message]  = Field(default_factory=list)
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    status:     TaskStatus     = TaskStatus.PLANNING
    retries:    int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def current_step(self) -> TaskStep | None:
        if self.step_idx < len(self.plan):
            return self.plan[self.step_idx]
        return None

    def add_message(self, msg: Message) -> None:
        self.history.append(msg)
        self.updated_at = datetime.utcnow()


# ─────────────────────────────────────────────
# Skill / Tool Descriptor
# ─────────────────────────────────────────────

class PermissionLevel(str, Enum):
    READ    = "read"
    WRITE   = "write"
    NETWORK = "network"
    EXEC    = "exec"

class ToolDescriptor(BaseModel):
    """Skill 和 MCP Tool 的统一抽象描述，LLM 直接消费。"""
    name:         str
    description:  str
    input_schema: dict[str, Any]   # JSON Schema
    source:       Literal["skill", "mcp"]
    permission:   PermissionLevel = PermissionLevel.READ
    timeout_ms:   int = 30_000
    tags:         list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────

class MemoryEntry(BaseModel):
    id:         str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    user_id:    str
    type:       MemoryType
    text:       str
    metadata:   dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5        # 0-1，固化时由 LLM 打分
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# LLM Response
# ─────────────────────────────────────────────

class LLMResponse(BaseModel):
    content:    str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage:      dict[str, int] = Field(default_factory=dict)  # {prompt_tokens, completion_tokens}
    model:      str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ─────────────────────────────────────────────
# Agent Config
# ─────────────────────────────────────────────

class AgentConfig(BaseModel):
    max_steps:       int         = 20
    timeout_ms:      int         = 120_000
    max_retries:     int         = 3
    token_budget:    int         = 12_000
    stream:          bool        = True
    # 单引擎模式：直接指定模型名（传给引擎的 default_model）
    model:           str         = ""
    # 多模型路由模式：指定 ModelRegistry 中的别名（优先级高于 model）
    model_alias:     str | None  = None
    # 工作流节点标识：用于节点级路由覆盖（TaskRouter.node_overrides）
    node_id:         str | None  = None
