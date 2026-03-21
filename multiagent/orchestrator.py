"""
multiagent/orchestrator.py — 多 Agent 协作编排系统

═══════════════════════════════════════════════════════════════
  架构概览
═══════════════════════════════════════════════════════════════

                         用户请求
                            │
                    OrchestratorAgent
                  ┌──────── │ ────────┐
                  │  1.分解任务+分配   │
                  │  2.构建依赖图     │
                  └──────── │ ────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ResearcherAgent   AnalystAgent      WriterAgent
    (web_search,       (agent_bi,       (无工具，
     datetime)         python_exec)     纯生成)
    {"role":"研究员"}  {"role":"分析师"} {"role":"撰稿人"}
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                     结果聚合 + 综合回答

═══════════════════════════════════════════════════════════════
  子 Agent 定义方式
═══════════════════════════════════════════════════════════════

  spec = AgentSpec(
      name        = "researcher",
      description = "负责信息搜索和资料收集，擅长网络检索",
      skills      = ["web_search", "datetime"],
      system_prompt = "你是一位专业研究员，专注于收集准确信息",
      max_steps   = 8,
  )

═══════════════════════════════════════════════════════════════
  编排事件流
═══════════════════════════════════════════════════════════════

{"type":"orchestrating","agents":["researcher","analyst","writer"]}
{"type":"subtask_assign","agent":"researcher","goal":"搜索竞品信息"}
{"type":"subtask_assign","agent":"analyst","goal":"分析数据","depends":["researcher"]}
{"type":"agent_start","agent":"researcher","subtask_id":"st_xxx"}
{"type":"agent_event","agent":"researcher","event":{...}}  ← 子 Agent 的原始事件
{"type":"agent_done","agent":"researcher","subtask_id":"st_xxx","tokens":N}
{"type":"agent_start","agent":"analyst","subtask_id":"st_yyy"}
...
{"type":"delta","text":"最终综合回答"}
{"type":"done","task_id":"...","status":"done","usage":{...}}

═══════════════════════════════════════════════════════════════
  快速使用
═══════════════════════════════════════════════════════════════

  specs = [
      AgentSpec("researcher", "搜索和资料收集", ["web_search"]),
      AgentSpec("analyst",    "数据分析",       ["agent_bi","python_exec"]),
      AgentSpec("writer",     "内容撰写",       []),
  ]
  orch = MultiAgentOrchestrator(container, specs)

  # 通过 AgentContainer.orchestrator 使用
  container.orchestrator = orch
  async for ev in container.agent().run("帮我做竞品分析报告"):
      print(ev)
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

from core.models import (
    AgentConfig,
    AgentInput,
    AgentTask,
    Message,
    MessageRole,
    StepStatus,
    TaskStatus,
    TaskStep,
    ToolCall,
    ToolResult,
)

log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 子 Agent 定义
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    """
    描述一个专属子 Agent 的能力。

    name:          唯一标识，LLM 在分配任务时使用此名称
    description:   面向 LLM 的能力描述（越精确越好）
    skills:        该 Agent 可以使用的 Skill 名称列表（空=全部）
    system_prompt: 子 Agent 的角色系统提示词（覆盖默认 prompt）
    max_steps:     子 Agent 单次任务最大工具调用轮数
    """
    name:          str
    description:   str
    skills:        list[str]      = field(default_factory=list)
    system_prompt: str            = ""
    max_steps:     int            = 8


@dataclass
class SubTask:
    """编排器分配给某个子 Agent 的子任务。"""
    id:         str         = field(default_factory=lambda: f"st_{uuid.uuid4().hex[:8]}")
    agent_name: str         = ""
    goal:       str         = ""
    depends:    list[str]   = field(default_factory=list)  # 依赖的 SubTask.id
    result:     str         = ""
    error:      str         = ""
    status:     StepStatus  = StepStatus.PENDING


# ──────────────────────────────────────────────────────────────────────
# 分解提示词
# ──────────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """你是任务编排系统。请将以下任务分配给合适的 Agent 执行。

任务：{goal}

可用 Agents：
{agents_desc}

请将任务分解，为每个子任务指定：
  - id:         子任务唯一标识（如 "t1","t2"）
  - agent:      执行此子任务的 Agent 名称（必须是上面列出的名称之一）
  - goal:       子任务目标（一句话，清晰具体）
  - depends:    依赖的子任务 id 列表（无依赖则为 []，有依赖则等上游完成后才执行）

注意：
  - 无依赖关系的子任务将并行执行（尽量让更多任务并行）
  - 若某子任务需要另一子任务的结果，必须在 depends 中声明
  - 每个 Agent 只分配一个子任务（如有必要可用同一 Agent 多次）
  - 总子任务数：2-6 个

直接返回 JSON 数组，不要其他内容。"""

_SYNTH_PROMPT = """请根据多个 Agent 的执行结果，生成最终综合回答。

原始任务：{goal}

各 Agent 执行结果：
{results}

请整合所有信息，给出完整、清晰的最终回答。"""


# ──────────────────────────────────────────────────────────────────────
# MultiAgentOrchestrator
# ──────────────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    多 Agent 协作编排引擎。

    内部使用 DAG 调度：子任务按依赖关系并行/串行执行。
    每个子任务由对应 AgentSpec 配置的专属 Agent 执行（独立的 skill 过滤和 system_prompt）。
    """

    def __init__(
        self,
        llm_engine:        Any,
        skill_registry:    Any,
        mcp_hub:           Any,
        context_manager:   Any,
        short_term_memory: Any,
        long_term_memory:  Any,
        agent_specs:       list[AgentSpec],
        max_parallel:      int = 4,
    ) -> None:
        self._llm    = llm_engine
        self._skills = skill_registry
        self._mcp    = mcp_hub
        self._ctx    = context_manager
        self._stm    = short_term_memory
        self._ltm    = long_term_memory
        self._specs  = {s.name: s for s in agent_specs}
        self._max_parallel = max_parallel

    # ══════════════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════════════

    async def run(
        self, task: AgentTask, config: AgentConfig
    ) -> AsyncIterator[dict[str, Any]]:
        task.status = TaskStatus.RUNNING
        await self._stm.save_task(task)
        total_tokens = 0

        try:
            # ── Phase 1: 任务分解与 Agent 分配 ────────────────────
            yield {
                "type":   "orchestrating",
                "agents": list(self._specs.keys()),
            }

            subtasks = await self._decompose(task, config)
            if not subtasks:
                # fallback: 单 Agent 直接执行
                subtasks = [SubTask(
                    agent_name = next(iter(self._specs)),
                    goal       = task.input.text,
                )]

            for st in subtasks:
                yield {
                    "type":       "subtask_assign",
                    "subtask_id": st.id,
                    "agent":      st.agent_name,
                    "goal":       st.goal,
                    "depends":    st.depends,
                }

            # ── Phase 2: DAG 并行执行子任务 ───────────────────────
            completed: dict[str, str]  = {}   # subtask_id → result
            failed:    set[str]        = set()
            in_flight: dict[str, asyncio.Task] = {}
            event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

            async def run_subtask(st: SubTask) -> None:
                await event_queue.put({
                    "type":       "agent_start",
                    "agent":      st.agent_name,
                    "subtask_id": st.id,
                    "goal":       st.goal,
                })
                try:
                    result, tokens = await self._run_subtask(
                        st, task, config, completed
                    )
                    st.result = result
                    st.status = StepStatus.DONE
                    completed[st.id] = result
                    await event_queue.put({
                        "type":       "agent_done",
                        "agent":      st.agent_name,
                        "subtask_id": st.id,
                        "tokens":     tokens,
                        "result":     result[:300],
                    })
                except Exception as exc:
                    log.exception("multiagent.subtask_failed",
                                  subtask_id=st.id, agent=st.agent_name)
                    st.status = StepStatus.FAILED
                    st.error  = str(exc)
                    failed.add(st.id)
                    await event_queue.put({
                        "type":       "agent_error",
                        "agent":      st.agent_name,
                        "subtask_id": st.id,
                        "error":      str(exc),
                    })
                finally:
                    await event_queue.put(None)  # 完成信号

            # DAG 调度
            while True:
                ready = [
                    st for st in subtasks
                    if st.id not in completed
                    and st.id not in in_flight
                    and st.id not in failed
                    and all(dep in completed for dep in st.depends)
                    and not any(dep in failed for dep in st.depends)
                ]
                if not ready and not in_flight:
                    break

                # 启动新的就绪子任务
                to_start = ready[: self._max_parallel - len(in_flight)]
                for st in to_start:
                    t = asyncio.create_task(run_subtask(st))
                    in_flight[st.id] = t

                # 消费事件队列
                while not event_queue.empty():
                    ev = event_queue.get_nowait()
                    if ev is None:
                        done_ids = [sid for sid, t in in_flight.items() if t.done()]
                        for sid in done_ids:
                            del in_flight[sid]
                    else:
                        yield ev

                # 等待至少一个子任务完成
                if in_flight:
                    await asyncio.wait(
                        list(in_flight.values()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # 清空队列
                    while not event_queue.empty():
                        ev = await event_queue.get()
                        if ev is None:
                            done_ids = [sid for sid, t in in_flight.items() if t.done()]
                            for sid in done_ids:
                                del in_flight[sid]
                        else:
                            yield ev

            # ── Phase 3: 综合所有子 Agent 结果 ────────────────────
            if completed:
                results_lines = []
                for st in subtasks:
                    if st.id in completed:
                        results_lines.append(
                            f"[{st.agent_name} - {st.goal}]\n{completed[st.id]}"
                        )
                results_text = "\n\n".join(results_lines)
                synth_prompt = _SYNTH_PROMPT.format(
                    goal=task.input.text, results=results_text
                )
                sum_fn = getattr(self._llm, "summarize", None)
                answer = await sum_fn(synth_prompt, max_tokens=2000) if sum_fn else results_text
                yield {"type": "delta", "text": answer}

            task.status = TaskStatus.DONE
            await self._stm.save_task(task)
            yield {
                "type":    "done",
                "task_id": task.id,
                "status":  "done",
                "usage":   {"total_tokens": total_tokens},
            }

        except Exception as exc:
            log.exception("multiagent_orchestrator.error", task_id=task.id)
            task.status = TaskStatus.ERROR
            await self._stm.save_task(task)
            yield {"type": "error", "message": str(exc)}
            yield {"type": "done", "task_id": task.id,
                   "status": "error", "usage": {}}

    # ══════════════════════════════════════════════════════════════════
    # 任务分解（LLM 分配给各 Agent）
    # ══════════════════════════════════════════════════════════════════

    async def _decompose(
        self, task: AgentTask, config: AgentConfig
    ) -> list[SubTask]:
        agents_desc = "\n".join(
            f'  - name: "{s.name}"\n    description: "{s.description}"\n    skills: {s.skills}'
            for s in self._specs.values()
        )
        prompt = _DECOMPOSE_PROMPT.format(
            goal=task.input.text,
            agents_desc=agents_desc,
        )
        sum_fn  = getattr(self._llm, "summarize", None)
        plan_fn = getattr(self._llm, "plan",      None)

        if plan_fn:
            messages = [Message(role=MessageRole.USER, content=prompt)]
            resp = await plan_fn(messages, config)
            raw  = resp.content or ""
        elif sum_fn:
            raw = await sum_fn(prompt, max_tokens=1000)
        else:
            return []

        return self._parse_subtasks(raw)

    def _parse_subtasks(self, raw: str) -> list[SubTask]:
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            items = json.loads(text)
        except Exception:
            log.warning("multiagent.decompose_parse_failed", raw=raw[:200])
            return []

        subtasks:  list[SubTask] = []
        id_map:    dict[str, str] = {}   # plan-local id → SubTask.id

        for item in items:
            local_id   = str(item.get("id", ""))
            agent_name = item.get("agent", "")
            # 若 agent 不在 specs 中，分配给第一个
            if agent_name not in self._specs and self._specs:
                agent_name = next(iter(self._specs))
            st = SubTask(
                agent_name = agent_name,
                goal       = item.get("goal", "执行任务"),
            )
            if local_id:
                id_map[local_id] = st.id
            subtasks.append(st)

        # 替换依赖 id
        for item, st in zip(items, subtasks):
            st.depends = [
                id_map[d] for d in item.get("depends", []) if d in id_map
            ]

        return subtasks

    # ══════════════════════════════════════════════════════════════════
    # 子任务执行（每个子 Agent 独立的 skill 过滤 + system_prompt）
    # ══════════════════════════════════════════════════════════════════

    async def _run_subtask(
        self,
        st:        SubTask,
        parent:    AgentTask,
        config:    AgentConfig,
        completed: dict[str, str],
    ) -> tuple[str, int]:
        spec = self._specs.get(st.agent_name)
        if not spec:
            raise ValueError(f"Agent spec not found: {st.agent_name!r}")

        # 过滤 skill（若 spec.skills 非空）
        if spec.skills:
            from skills.loader import FilteredSkillRegistry
            skill_reg = FilteredSkillRegistry(self._skills, set(spec.skills))
        else:
            skill_reg = self._skills

        tools = skill_reg.list_descriptors() + self._mcp.list_descriptors()

        # 构建子任务消息历史
        history: list[Message] = []

        # 1. Agent 角色 system prompt
        if spec.system_prompt:
            history.append(Message(role=MessageRole.SYSTEM, content=spec.system_prompt))

        # 2. 上游结果注入
        upstream = _build_upstream_str(st, completed)
        if upstream:
            history.append(Message(
                role    = MessageRole.SYSTEM,
                content = f"上游 Agent 执行结果（供参考）：\n{upstream}",
            ))

        # 3. 子任务目标
        history.append(Message(role=MessageRole.USER, content=st.goal))

        total_tokens = 0
        for _ in range(spec.max_steps):
            resp = await self._llm.chat(history, tools, config)
            total_tokens += resp.usage.get("total_tokens", 0)

            if not resp.tool_calls:
                return resp.content or "", total_tokens

            history.append(Message(
                role       = MessageRole.ASSISTANT,
                content    = resp.content,
                tool_calls = resp.tool_calls,
            ))

            for tc in resp.tool_calls:
                result = await self._dispatch_tool(tc, skill_reg)
                history.append(Message(role=MessageRole.TOOL, tool_result=result))

        # 超出步数限制
        last = next(
            (m.content for m in reversed(history)
             if m.role == MessageRole.ASSISTANT and m.content), ""
        )
        return last, total_tokens

    async def _dispatch_tool(self, tc: ToolCall, skill_reg: Any) -> ToolResult:
        t0 = time.monotonic()
        try:
            if "__" in tc.tool_name:
                srv, tool = tc.tool_name.split("__", 1)
                content = await self._mcp.call(srv, tool, tc.arguments)
            else:
                content = await skill_reg.call(tc.tool_name, tc.arguments)
            return ToolResult(
                tool_call_id=tc.id, tool_name=tc.tool_name, content=content,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tc.id, tool_name=tc.tool_name,
                content=None, error=str(exc),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )


def _build_upstream_str(st: SubTask, completed: dict[str, str]) -> str:
    if not st.depends:
        return ""
    lines = [f"- 子任务[{dep_id}]：{completed.get(dep_id, '（无结果）')[:400]}"
             for dep_id in st.depends]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# 工厂函数：从 AgentContainer 快速创建多 Agent 编排器
# ──────────────────────────────────────────────────────────────────────

def create_multi_agent_orchestrator(
    container:   Any,
    agent_specs: list[AgentSpec],
    max_parallel: int = 4,
) -> MultiAgentOrchestrator:
    """
    从现有 AgentContainer 创建 MultiAgentOrchestrator，
    复用 LLM、Skill、Memory 等所有组件。

    使用示例：
        from multiagent.orchestrator import AgentSpec, create_multi_agent_orchestrator

        specs = [
            AgentSpec("researcher", "信息搜索与资料收集", ["web_search","datetime"]),
            AgentSpec("analyst",    "数据分析与可视化",    ["agent_bi","python_exec"]),
            AgentSpec("writer",     "内容撰写与报告输出",   []),
        ]
        container.orchestrator = create_multi_agent_orchestrator(container, specs)
    """
    router = container._effective_router()
    return MultiAgentOrchestrator(
        llm_engine        = router,
        skill_registry    = container.skill_registry,
        mcp_hub           = container.mcp_hub,
        context_manager   = container.context_manager,
        short_term_memory = container.short_term_memory,
        long_term_memory  = container.long_term_memory,
        agent_specs       = agent_specs,
        max_parallel      = max_parallel,
    )
