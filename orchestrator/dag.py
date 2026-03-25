"""
orchestrator/dag.py — DAG 任务拆分编排引擎

═══════════════════════════════════════════════════════════════
  核心思想
═══════════════════════════════════════════════════════════════

把"计划"从有序列表升级为有向无环图（DAG）：

  顺序模式（PlanExecute）:
    step1 → step2 → step3 → step4

  DAG 模式（本文件）:
    step_search ─────────────────────────────────┐
                                                   ├─→ step_report
    step_fetch ─→ step_analyze ──────────────────┘
    （search / fetch 并行，analyze 等 fetch，report 等全部完成）

═══════════════════════════════════════════════════════════════
  LLM 生成计划时需额外返回 depends 字段
═══════════════════════════════════════════════════════════════

LLM 返回 JSON 示例：
[
  { "id": "s1", "goal": "搜索竞品信息",     "tool_hint": "web_search", "depends": [] },
  { "id": "s2", "goal": "获取销售数据",     "tool_hint": "agent_bi",   "depends": [] },
  { "id": "s3", "goal": "分析数据对比竞品", "tool_hint": null,         "depends": ["s1","s2"] },
  { "id": "s4", "goal": "生成报告",         "tool_hint": null,         "depends": ["s3"] }
]

s1、s2 无依赖 → 并行执行
s3 等 s1、s2 → 串行
s4 等 s3 → 串行

═══════════════════════════════════════════════════════════════
  事件流
═══════════════════════════════════════════════════════════════

{"type":"planning"}
{"type":"plan","steps":[{"id":"s1","goal":"...","depends":[]}, ...]}
{"type":"parallel_start","step_ids":["s1","s2"]}
{"type":"step_start","step_id":"s1","step":1,"goal":"..."}
{"type":"step_start","step_id":"s2","step":2,"goal":"..."}
{"type":"step_done","step_id":"s1","step":1,"result":"..."}
{"type":"step_done","step_id":"s2","step":2,"result":"..."}
{"type":"step_start","step_id":"s3","step":3,"goal":"..."}
...
{"type":"delta","text":"最终回答"}
{"type":"done","task_id":"...","status":"done","usage":{...}}
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
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

# ── Plan 生成 Prompt ──────────────────────────────────────────────────

_PLAN_PROMPT = """请分析以下任务并制定执行计划。

任务：{goal}

将任务分解为 2-8 个步骤，识别哪些步骤可以并行执行。
返回 JSON 数组，每项包含：
  - id:        步骤唯一标识（如 "s1","s2"，你自己分配）
  - goal:      步骤目标（简洁一句话）
  - tool_hint: 建议使用的工具名（可选，若不确定则设为 null）
  - depends:   依赖的步骤 id 列表（空数组 = 可立即执行）

依赖规则：
  - 若步骤 B 需要步骤 A 的结果，则 B.depends = ["A的id"]
  - 无依赖关系的步骤应该并行执行（depends = []）
  - 避免不必要的依赖（让更多步骤能并行）

直接返回 JSON 数组，不要其他内容。"""

_SYNTH_PROMPT = """请根据以下任务执行结果生成最终综合回答。

原始任务：{goal}

各步骤执行结果：
{results}

请用清晰、结构化的方式总结答案。"""


class DAGOrchestrator:
    """
    DAG 并行任务拆分编排引擎。

    与 PlanExecuteOrchestrator 的区别：
      - 步骤带 depends 字段，建立依赖图
      - 无依赖或依赖已完成的步骤并行执行
      - 每个步骤独立运行 mini-ReAct（最多 max_step_steps 轮）
      - 步骤结果写入 task.scratchpad[step_id]，供下游步骤参考
    """

    def __init__(
        self,
        llm_engine: Any,
        skill_registry: Any,
        mcp_hub: Any,
        context_manager: Any,
        short_term_memory: Any,
        long_term_memory: Any,
        max_parallel: int = 4,   # 最大并行步骤数
        max_step_steps: int = 5, # 每步 ReAct 最大轮数
    ) -> None:
        self._llm    = llm_engine
        self._skills = skill_registry
        self._mcp    = mcp_hub
        self._ctx    = context_manager
        self._stm    = short_term_memory
        self._ltm    = long_term_memory
        self._max_parallel  = max_parallel
        self._max_step_steps = max_step_steps

    # ══════════════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════════════

    async def run(
        self, task: AgentTask, config: AgentConfig
    ) -> AsyncIterator[dict[str, Any]]:
        task.status = TaskStatus.RUNNING
        await self._stm.save_task(task)

        event_queue: asyncio.Queue[dict] = asyncio.Queue()
        total_tokens = 0

        try:
            # ── Phase 1: 生成带依赖的计划 ──────────────────────────
            yield {"type": "planning"}
            steps = await self._generate_plan(task, config)
            task.plan = steps
            await self._stm.save_task(task)

            yield {
                "type":  "plan",
                "steps": [
                    {"id": s.id, "goal": s.goal,
                     "tool_hint": s.tool_hint, "depends": s.depends}
                    for s in steps
                ],
            }

            # ── Phase 2: DAG 并行执行 ───────────────────────────────
            completed: dict[str, Any] = {}   # step_id → result text
            failed:    set[str]       = set()
            step_map   = {s.id: s for s in steps}

            # 每个步骤的事件通过独立队列传输到主流
            async def run_step_emit(step: TaskStep, step_num: int) -> None:
                """执行单步，把事件发到队列。"""
                await event_queue.put({
                    "type": "step_start",
                    "step_id": step.id,
                    "step":    step_num,
                    "goal":    step.goal,
                })
                try:
                    result, tokens = await self._execute_step(
                        step, task, config, completed
                    )
                    step.result = result
                    step.status = StepStatus.DONE
                    completed[step.id] = result
                    total_tokens.__iadd__ if False else None  # 只是占位，tokens 后面聚合
                    await event_queue.put({
                        "type":    "step_done",
                        "step_id": step.id,
                        "step":    step_num,
                        "result":  str(result)[:300],
                    })
                except Exception as exc:
                    step.status = StepStatus.FAILED
                    step.error  = str(exc)
                    failed.add(step.id)
                    await event_queue.put({
                        "type":    "step_failed",
                        "step_id": step.id,
                        "step":    step_num,
                        "error":   str(exc),
                    })
                await event_queue.put(None)  # 步骤完成信号

            # DAG 调度主循环
            step_num   = 0
            in_flight:  set[str] = set()
            tasks_map: dict[str, asyncio.Task] = {}
            _sentinel = object()

            while True:
                # 找出可以立即执行的步骤
                ready = [
                    s for s in steps
                    if s.id not in completed
                    and s.id not in in_flight
                    and s.id not in failed
                    # 所有依赖已完成
                    and all(dep in completed for dep in s.depends)
                    # 上游没有失败（跳过依赖失败的步骤）
                    and not any(dep in failed for dep in s.depends)
                ]

                if not ready and not in_flight:
                    break   # 所有步骤已完成或无法继续

                # 批量启动，控制并发数
                to_start = ready[: self._max_parallel - len(in_flight)]
                if to_start:
                    yield {
                        "type":     "parallel_start",
                        "step_ids": [s.id for s in to_start],
                        "goals":    [s.goal for s in to_start],
                    }
                    for s in to_start:
                        step_num += 1
                        in_flight.add(s.id)
                        t = asyncio.create_task(run_step_emit(s, step_num))
                        tasks_map[s.id] = t

                # 消费事件队列（非阻塞 poll）
                drained = False
                while not drained:
                    try:
                        ev = event_queue.get_nowait()
                        if ev is None:
                            # 某步骤完成，移出 in_flight
                            done_ids = [
                                sid for sid, t in tasks_map.items()
                                if t.done() and sid in in_flight
                            ]
                            for sid in done_ids:
                                in_flight.discard(sid)
                        else:
                            yield ev
                    except asyncio.QueueEmpty:
                        drained = True
                        if in_flight:
                            await asyncio.sleep(0.05)  # 等待步骤完成

                # 等待至少一个步骤完成，然后重新检查 ready 集合
                if in_flight:
                    done, _ = await asyncio.wait(
                        [tasks_map[sid] for sid in in_flight],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in done:
                        for sid, task_ in tasks_map.items():
                            if task_ is t:
                                in_flight.discard(sid)
                    # 清空队列中剩余的事件
                    while not event_queue.empty():
                        ev = await event_queue.get()
                        if ev is not None:
                            yield ev

            # ── Phase 3: 综合所有步骤结果生成最终回答 ─────────────
            if completed:
                results_text = "\n".join(
                    f"[步骤 {s.goal}]\n{completed.get(s.id, '（无结果）')}"
                    for s in steps if s.id in completed
                )
                synth_prompt = _SYNTH_PROMPT.format(
                    goal=task.input.text, results=results_text
                )
                sum_fn = getattr(self._llm, "summarize", None)
                if sum_fn:
                    answer = await sum_fn(synth_prompt, max_tokens=1500)
                else:
                    # fallback: 拼接所有结果
                    answer = results_text
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
            log.exception("dag_orchestrator.error", task_id=task.id)
            task.status = TaskStatus.ERROR
            await self._stm.save_task(task)
            yield {"type": "error", "message": str(exc)}
            yield {
                "type":    "done",
                "task_id": task.id,
                "status":  "error",
                "usage":   {"total_tokens": total_tokens},
            }

    # ══════════════════════════════════════════════════════════════════
    # 计划生成（LLM 返回带 depends 的 JSON）
    # ══════════════════════════════════════════════════════════════════

    async def _generate_plan(self, task: AgentTask, config: AgentConfig) -> list[TaskStep]:
        prompt  = _PLAN_PROMPT.format(goal=task.input.text)
        sum_fn  = getattr(self._llm, "summarize", None)
        plan_fn = getattr(self._llm, "plan",      None)

        if plan_fn:
            messages = [Message(role=MessageRole.USER, content=prompt)]
            resp = await plan_fn(messages, config)
            raw  = resp.content or ""
        elif sum_fn:
            raw = await sum_fn(prompt, max_tokens=800)
        else:
            raw = "[]"

        steps = self._parse_plan(raw)
        if not steps:
            steps = [TaskStep(goal=task.input.text)]
        return steps

    def _parse_plan(self, raw: str) -> list[TaskStep]:
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            items = json.loads(text)
        except Exception:
            log.warning("dag.plan_parse_failed", raw=raw[:200])
            return []

        steps: list[TaskStep] = []
        id_map: dict[str, str] = {}   # plan-local id → TaskStep.id

        for item in items:
            local_id = str(item.get("id", ""))
            s = TaskStep(
                goal      = item.get("goal", "执行任务"),
                tool_hint = item.get("tool_hint") or None,
            )
            if local_id:
                id_map[local_id] = s.id
            steps.append(s)

        # 第二遍：把 depends 里的 local_id 替换为 TaskStep.id
        for item, step in zip(items, steps):
            step.depends = [
                id_map[d] for d in item.get("depends", []) if d in id_map
            ]

        return steps

    # ══════════════════════════════════════════════════════════════════
    # 单步执行（mini-ReAct）
    # ══════════════════════════════════════════════════════════════════

    async def _execute_step(
        self,
        step:      TaskStep,
        task:      AgentTask,
        config:    AgentConfig,
        completed: dict[str, Any],
    ) -> tuple[str, int]:
        """
        在子任务上运行 mini-ReAct，最多 max_step_steps 轮。
        上游步骤结果通过 step_context 注入到 system 消息。
        """
        step.status = StepStatus.RUNNING
        total_tokens = 0

        # 构建子任务
        upstream_ctx = _build_upstream_context(step, task, completed)
        sub_task = AgentTask(
            session_id = f"{task.session_id}__sub__{step.id}",
            user_id    = task.user_id,
            input      = AgentInput(text=step.goal),
        )
        if upstream_ctx:
            sub_task.scratchpad["upstream"] = upstream_ctx

        tools   = self._skills.list_descriptors() + self._mcp.list_descriptors()
        history: list[Message] = []

        # 注入上游结果到 system 消息
        if upstream_ctx:
            history.append(Message(
                role    = MessageRole.SYSTEM,
                content = f"上游步骤结果（供本步骤参考）：\n{upstream_ctx}",
            ))

        history.append(Message(role=MessageRole.USER, content=step.goal))

        for _ in range(self._max_step_steps):
            resp = await self._llm.chat(history, tools, config)
            total_tokens += resp.usage.get("total_tokens", 0)

            if not resp.tool_calls:
                # 得到文本回答 → 步骤完成
                return resp.content or "", total_tokens

            # 执行工具调用
            assistant_msg = Message(
                role       = MessageRole.ASSISTANT,
                content    = resp.content,
                tool_calls = resp.tool_calls,
            )
            history.append(assistant_msg)

            for tc in resp.tool_calls:
                result = await self._dispatch_tool(tc)
                history.append(Message(
                    role        = MessageRole.TOOL,
                    tool_result = result,
                ))

        # 超过最大轮数：返回最后的 content
        last = next((m.content for m in reversed(history)
                     if m.role == MessageRole.ASSISTANT and m.content), "")
        return last, total_tokens

    async def _dispatch_tool(self, tc: ToolCall) -> ToolResult:
        t0 = time.monotonic()
        try:
            if "__" in tc.tool_name:
                srv, tool = tc.tool_name.split("__", 1)
                content = await self._mcp.call(srv, tool, tc.arguments)
            else:
                content = await self._skills.call(tc.tool_name, tc.arguments)
            return ToolResult(
                tool_call_id=tc.id, tool_name=tc.tool_name,
                content=content,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tc.id, tool_name=tc.tool_name,
                content=None, error=str(exc),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )


# ──────────────────────────────────────────────────────────────────────
# 辅助：构建上游步骤的上下文
# ──────────────────────────────────────────────────────────────────────

def _build_upstream_context(
    step:      TaskStep,
    task:      AgentTask,
    completed: dict[str, Any],
) -> str:
    """将 step.depends 里已完成步骤的结果格式化为文本。"""
    if not step.depends:
        return ""
    step_map = {s.id: s for s in task.plan}
    lines = []
    for dep_id in step.depends:
        dep_step = step_map.get(dep_id)
        result   = completed.get(dep_id, "（无结果）")
        goal     = dep_step.goal if dep_step else dep_id
        lines.append(f"- {goal}：{str(result)[:400]}")
    return "\n".join(lines)
