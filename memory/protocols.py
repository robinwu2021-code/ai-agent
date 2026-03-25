"""
memory/protocols.py — 记忆系统接口定义

所有 Memory 实现必须满足以下 Protocol，使调用方可以无感知地切换后端。
接口分三层：
  ShortTermStore   短期记忆（任务状态 + 消息历史 + Scratchpad）
  LongTermStore    长期记忆（语义存储 + 检索 + 用户画像）
  WorkingMemory    工作记忆（当前会话的 context 组装 + 历史压缩）

所有方法均为 async，无 sync 版本（FastAPI 生产环境保持一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.models import AgentTask, Message


# ─────────────────────────────────────────────────────────────────
# 公共数据结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class ScoredMemory:
    """长期记忆检索结果，携带 relevance score。"""
    id:          str
    user_id:     str
    text:        str
    score:       float        = 0.0
    importance:  float        = 0.5
    memory_type: str          = "semantic"   # semantic | episodic | profile
    scope:       str          = "personal"   # personal | project | workspace
    metadata:    dict         = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# ShortTermStore Protocol
# ─────────────────────────────────────────────────────────────────

@runtime_checkable
class ShortTermStore(Protocol):
    """
    短期记忆接口。
    保存任务执行状态，跨进程用 Redis，单进程测试用 InMemory。
    """

    async def save_task(self, task: AgentTask) -> None: ...
    async def load_task(self, task_id: str) -> AgentTask | None: ...
    async def delete_task(self, task_id: str) -> None: ...

    async def append_message(self, task_id: str, msg: Message) -> None: ...
    async def get_messages(self, task_id: str) -> list[Message]: ...

    async def set_scratchpad(self, task_id: str, key: str, value: Any) -> None: ...
    async def get_scratchpad(self, task_id: str, key: str) -> Any: ...


# ─────────────────────────────────────────────────────────────────
# LongTermStore Protocol
# ─────────────────────────────────────────────────────────────────

@runtime_checkable
class LongTermStore(Protocol):
    """
    长期记忆接口。
    核心：write / search / prune。
    workspace 扩展方法（write_scoped / search_mixed）可选实现。
    """

    async def write(self, entry: Any) -> None: ...

    async def search(
        self,
        user_id:     str,
        query:       str,
        top_k:       int  = 5,
        filters:     dict = {},
    ) -> list[ScoredMemory]: ...

    async def get_profile(
        self, user_id: str, workspace_id: str = ""
    ) -> dict[str, Any]: ...

    async def update_profile(
        self, user_id: str, data: dict[str, Any]
    ) -> None: ...

    async def prune(
        self, user_id: str, max_items: int, score_threshold: float
    ) -> int: ...

    # ── 可选：workspace 多租户扩展 ──────────────────────────────
    # 实现方可选覆盖，调用方通过 hasattr 检测
    #
    # async def write_scoped(scope, author_id, text, workspace_id, ...) -> Any
    # async def search_mixed(query, user_id, workspace_id, project_id, ...) -> list


# ─────────────────────────────────────────────────────────────────
# WorkingMemory Protocol
# ─────────────────────────────────────────────────────────────────

@runtime_checkable
class WorkingMemory(Protocol):
    """
    工作记忆接口：当前会话的活跃 context 组装与历史压缩。
    替换原 PriorityContextManager，承担上下文构建的全部逻辑。
    """

    async def build_context(
        self,
        task:         AgentTask,
        tools:        list[Any],
        ltm:          Any,                    # LongTermStore
        budget:       int,
        workspace_id: str | None = None,
        project_id:   str | None = None,
        system_prompt_override: str | None = None,
    ) -> list[Message]: ...

    async def compress_history(
        self,
        messages: list[Message],
        budget:   int,
    ) -> list[Message]: ...


# ─────────────────────────────────────────────────────────────────
# MemorySystem — 统一门面（持有三层实例）
# ─────────────────────────────────────────────────────────────────

@dataclass
class MemorySystem:
    """
    Memory 门面，组合三层实现。

    通过 MemoryFactory.build_from_yaml() 构建，
    外部只依赖此对象，不直接依赖具体实现类。
    """
    stm:     Any   # ShortTermStore
    ltm:     Any   # LongTermStore
    working: Any   # WorkingMemory
    consolidator: Any  # MemoryConsolidator

    def is_workspace_aware(self) -> bool:
        """LTM 是否支持 workspace 多租户检索。"""
        return hasattr(self.ltm, "search_mixed")
