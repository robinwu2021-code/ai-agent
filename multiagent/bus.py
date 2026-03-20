"""
multiagent/bus.py — 多 Agent 协作系统

架构：
  - AgentBus        Agent 间消息总线（发布/订阅）
  - SubAgentPool    子 Agent 生命周期管理（创建/复用/销毁）
  - RouterAgent     负责将任务路由到最合适的 Sub-Agent
  - AgentMessage    Agent 间通信消息格式

使用方式：
    bus  = AgentBus()
    pool = SubAgentPool(container_factory=make_container)
    router = RouterAgent(bus, pool, llm_engine)

    result = await router.dispatch(
        task="分析这份财报并生成摘要",
        context={"user_id": "u1", "session_id": "s1"},
    )
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

import structlog

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────

@dataclass
class AgentMessage:
    id:          str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender:      str = ""
    recipient:   str = ""          # "" = broadcast
    msg_type:    str = "task"      # task | result | error | status | cancel
    payload:     dict = field(default_factory=dict)
    reply_to:    str | None = None
    created_at:  str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class SubAgentSpec:
    """子 Agent 的能力描述，用于路由决策。"""
    name:        str
    description: str              # 面向 LLM 的能力描述
    skills:      list[str]        # 擅长的技能标签
    max_steps:   int = 10


# ─────────────────────────────────────────────
# Agent Bus
# ─────────────────────────────────────────────

class AgentBus:
    """
    异步消息总线。
    Agents 通过 subscribe/publish 实现解耦通信。
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._history:     list[AgentMessage]             = []

    def subscribe(self, agent_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._subscribers.setdefault(agent_id, []).append(q)
        return q

    def unsubscribe(self, agent_id: str, queue: asyncio.Queue) -> None:
        queues = self._subscribers.get(agent_id, [])
        if queue in queues:
            queues.remove(queue)

    async def publish(self, msg: AgentMessage) -> None:
        self._history.append(msg)
        targets = (
            self._subscribers.get(msg.recipient, [])
            if msg.recipient
            else [q for qs in self._subscribers.values() for q in qs]
        )
        for q in targets:
            await q.put(msg)
        log.debug("bus.publish", type=msg.msg_type,
                  sender=msg.sender, recipient=msg.recipient or "broadcast")

    async def request(
        self, sender: str, recipient: str,
        payload: dict, timeout: float = 60.0
    ) -> AgentMessage:
        """发送请求并等待回复（RPC 模式）。"""
        msg_id = uuid.uuid4().hex[:8]
        reply_q = asyncio.Queue()
        self._subscribers.setdefault(f"__reply_{msg_id}", []).append(reply_q)

        req = AgentMessage(
            id=msg_id, sender=sender, recipient=recipient,
            msg_type="task", payload=payload, reply_to=f"__reply_{msg_id}",
        )
        await self.publish(req)

        try:
            reply = await asyncio.wait_for(reply_q.get(), timeout=timeout)
            return reply
        except asyncio.TimeoutError:
            return AgentMessage(
                sender=recipient, recipient=sender,
                msg_type="error",
                payload={"error": f"Agent '{recipient}' timed out after {timeout}s"},
                reply_to=msg_id,
            )
        finally:
            self._subscribers.pop(f"__reply_{msg_id}", None)

    def get_history(self, limit: int = 50) -> list[AgentMessage]:
        return list(self._history[-limit:])


# ─────────────────────────────────────────────
# Sub Agent Pool
# ─────────────────────────────────────────────

class SubAgentPool:
    """
    子 Agent 生命周期管理。
    container_factory: (spec: SubAgentSpec) -> AgentContainer
    """

    def __init__(self, container_factory: Callable) -> None:
        self._factory     = container_factory
        self._specs:      dict[str, SubAgentSpec] = {}
        self._containers: dict[str, Any]          = {}

    def register(self, spec: SubAgentSpec) -> None:
        self._specs[spec.name] = spec
        log.info("subagent.registered", name=spec.name)

    def get_agent(self, name: str) -> Any | None:
        """获取或创建 Sub-Agent 实例。"""
        if name not in self._specs:
            return None
        if name not in self._containers:
            spec = self._specs[name]
            self._containers[name] = self._factory(spec)
            log.info("subagent.created", name=name)
        return self._containers[name]

    def list_specs(self) -> list[SubAgentSpec]:
        return list(self._specs.values())

    def describe_for_llm(self) -> str:
        """返回所有 Sub-Agent 的能力描述，供路由 LLM 使用。"""
        lines = []
        for spec in self._specs.values():
            lines.append(
                f"- {spec.name}: {spec.description} "
                f"(技能: {', '.join(spec.skills)})"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Router Agent
# ─────────────────────────────────────────────

ROUTE_PROMPT = """你是任务路由器。根据任务描述，选择最合适的 Agent 来处理。

可用 Agent：
{agents}

任务：{task}

请直接返回 Agent 名称（只需名称，无需解释）。
如果没有合适的 Agent，返回 "orchestrator"。"""


class RouterAgent:
    """
    将任务路由到最合适的 Sub-Agent。
    也可并行分发子任务并聚合结果。
    """

    def __init__(
        self,
        bus: AgentBus,
        pool: SubAgentPool,
        llm_engine: Any,
    ) -> None:
        self._bus  = bus
        self._pool = pool
        self._llm  = llm_engine

    async def route(self, task: str, context: dict) -> str:
        """返回最合适的 Sub-Agent 名称。"""
        agents = self._pool.describe_for_llm()
        if not agents:
            return "orchestrator"

        prompt = ROUTE_PROMPT.format(agents=agents, task=task[:500])
        route_fn = getattr(self._llm, "route_decision", None)
        name = await route_fn(prompt, max_tokens=30) if route_fn else await self._llm.summarize(prompt, max_tokens=30)
        name   = name.strip().split()[0].lower()

        if name in {s.name for s in self._pool.list_specs()}:
            log.info("router.decision", task=task[:80], agent=name)
            return name
        return "orchestrator"

    async def dispatch(
        self,
        task: str,
        context: dict,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """路由并执行，返回结果。"""
        agent_name = await self.route(task, context)
        container  = self._pool.get_agent(agent_name)

        if container is None:
            return {"agent": "orchestrator", "result": None,
                    "error": f"Agent '{agent_name}' not found"}

        from core.models import AgentConfig
        agent  = container.agent()
        config = AgentConfig(
            max_steps=self._pool._specs.get(agent_name, SubAgentSpec("", "", [])).max_steps
        )

        events = []
        async for event in agent.run(
            user_id=context.get("user_id", "multi_agent"),
            session_id=context.get("session_id", f"sub_{uuid.uuid4().hex[:6]}"),
            text=task,
            config=config,
        ):
            events.append(event)

        # 提取最终文本
        text = next(
            (e["text"] for e in reversed(events) if e.get("type") == "delta"),
            ""
        )
        done = next((e for e in reversed(events) if e.get("type") == "done"), {})

        return {
            "agent":  agent_name,
            "result": text,
            "status": done.get("status", "unknown"),
            "usage":  done.get("usage", {}),
        }

    async def dispatch_parallel(
        self,
        tasks: list[str],
        context: dict,
    ) -> list[dict[str, Any]]:
        """并行分发多个子任务，等待全部完成。"""
        coros   = [self.dispatch(t, context) for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return [
            r if isinstance(r, dict) else {"error": str(r), "result": None}
            for r in results
        ]
