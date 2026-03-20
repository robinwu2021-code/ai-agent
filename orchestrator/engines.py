"""
orchestrator/engines.py — 编排引擎实现

包含：
  - ReactOrchestrator       单步 ReAct 循环，适合简单任务
  - PlanExecuteOrchestrator 先规划后执行，适合复杂多步任务
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import structlog

from core.models import (
    AgentConfig, AgentTask, Message, MessageRole,
    TaskStatus, TaskStep, StepStatus,
    ToolCall, ToolResult,
)

log = structlog.get_logger(__name__)

PLAN_PROMPT = """请分析以下任务并制定执行计划。

任务：{goal}

请将任务分解为 2-8 个具体步骤，每步骤应是一个独立的可执行操作。
返回 JSON 数组，每项包含：
  - goal: 步骤目标（简洁的一句话）
  - tool_hint: 建议使用的工具名（可选，若不确定则省略）

只返回 JSON，不要任何其他内容。"""


class ReactOrchestrator:
    """
    ReAct（Reason + Act）循环编排引擎。
    每轮：LLM 推理 → 工具调用 → 观察结果 → 下一轮推理
    适合简单至中等复杂度任务，无需预先规划。
    """

    def __init__(
        self,
        llm_engine: Any,
        skill_registry: Any,
        mcp_hub: Any,
        context_manager: Any,
        short_term_memory: Any,
        long_term_memory: Any,
    ) -> None:
        self._llm    = llm_engine
        self._skills = skill_registry
        self._mcp    = mcp_hub
        self._ctx    = context_manager
        self._stm    = short_term_memory
        self._ltm    = long_term_memory

    def _all_tools(self):
        return self._skills.list_descriptors() + self._mcp.list_descriptors()

    async def run(
        self, task: AgentTask, config: AgentConfig
    ) -> AsyncIterator[dict[str, Any]]:
        task.status = TaskStatus.RUNNING
        await self._stm.save_task(task)

        total_tokens = 0
        step_num     = 0

        try:
            for _ in range(config.max_steps):
                step_num += 1
                # 组装上下文
                messages = await self._ctx.build(
                    task, self._all_tools(), self._ltm, config.token_budget
                )

                # LLM 推理
                yield {"type": "thinking", "step": step_num}
                resp = await self._llm.chat(messages, self._all_tools(), config)
                total_tokens += sum(resp.usage.values())

                # 无工具调用 → 最终回答
                if not resp.has_tool_calls:
                    task.add_message(Message(
                        role=MessageRole.ASSISTANT,
                        content=resp.content,
                    ))
                    if resp.content:
                        yield {"type": "delta", "text": resp.content}
                    break

                # 有工具调用
                task.add_message(Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=resp.tool_calls,
                ))

                # 逐一执行工具
                for tc in resp.tool_calls:
                    yield {"type": "step", "step": step_num,
                           "tool": tc.tool_name, "status": "running"}

                    tool_result = await self._dispatch_tool(tc, task)

                    # 注入工具结果
                    task.add_message(Message(
                        role=MessageRole.TOOL,
                        tool_result=tool_result,
                    ))
                    await self._stm.append_message(task.id, task.history[-1])

                    yield {"type": "step", "step": step_num,
                           "tool": tc.tool_name, "status": "done",
                           "error": tool_result.error}

                await self._stm.save_task(task)

            task.status = TaskStatus.DONE

        except Exception as e:
            log.exception("orchestrator.error", task_id=task.id, error=str(e))
            task.status = TaskStatus.ERROR
            yield {"type": "error", "message": str(e)}

        finally:
            await self._stm.save_task(task)
            yield {"type": "done", "task_id": task.id,
                   "status": task.status.value,
                   "usage": {"total_tokens": total_tokens}}

    async def _dispatch_tool(self, tc: ToolCall, task: AgentTask) -> ToolResult:
        """根据工具名前缀路由到 Skill 或 MCP。"""
        # MCP 工具格式：{server_name}__{tool_name}
        if "__" in tc.tool_name:
            server_name, tool_name = tc.tool_name.split("__", 1)
            result = await self._mcp.call(server_name, tool_name, tc.arguments)
        else:
            result = await self._skills.call(tc.tool_name, tc.arguments)
        result.tool_call_id = tc.id
        return result


class PlanExecuteOrchestrator:
    """
    Plan-and-Execute 编排引擎。
    先让 LLM 生成完整执行计划，再逐步执行，支持重规划。
    适合复杂多步任务（文件处理、多服务集成等）。
    """

    def __init__(
        self,
        llm_engine: Any,
        skill_registry: Any,
        mcp_hub: Any,
        context_manager: Any,
        short_term_memory: Any,
        long_term_memory: Any,
    ) -> None:
        self._llm    = llm_engine
        self._skills = skill_registry
        self._mcp    = mcp_hub
        self._ctx    = context_manager
        self._stm    = short_term_memory
        self._ltm    = long_term_memory

    def _all_tools(self):
        return self._skills.list_descriptors() + self._mcp.list_descriptors()

    async def run(
        self, task: AgentTask, config: AgentConfig
    ) -> AsyncIterator[dict[str, Any]]:
        task.status = TaskStatus.PLANNING
        total_tokens = 0

        try:
            # ── Phase 1: 生成计划 ─────────────────────
            yield {"type": "planning"}
            plan = await self._generate_plan(task, config)
            task.plan = plan
            task.status = TaskStatus.RUNNING
            await self._stm.save_task(task)
            yield {"type": "plan", "steps": [{"id": s.id, "goal": s.goal} for s in plan]}

            # ── Phase 2: 逐步执行 ─────────────────────
            for i, step in enumerate(task.plan):
                task.step_idx = i
                step.status   = StepStatus.RUNNING
                yield {"type": "step_start", "step": i + 1, "goal": step.goal}

                step_result = await self._execute_step(step, task, config)
                total_tokens += step_result.get("tokens", 0)

                if step_result.get("error"):
                    step.status = StepStatus.FAILED
                    step.error  = step_result["error"]
                    # 触发重规划（最多 1 次）
                    if task.retries < config.max_retries:
                        task.retries += 1
                        yield {"type": "replanning", "step": i + 1, "reason": step.error}
                        new_steps = await self._replan_from(task, i, config)
                        task.plan[i:] = new_steps
                        continue
                    else:
                        yield {"type": "step_failed", "step": i + 1, "error": step.error}
                        break
                else:
                    step.status  = StepStatus.DONE
                    step.result  = step_result.get("result")
                    task.scratchpad[step.id] = step.result
                    yield {"type": "step_done", "step": i + 1, "result": str(step.result)[:200]}

                await self._stm.save_task(task)

            # ── Phase 3: 汇总最终答案 ─────────────────
            final = await self._synthesize(task, config)
            task.add_message(Message(role=MessageRole.ASSISTANT, content=final))
            yield {"type": "delta", "text": final}
            task.status = TaskStatus.DONE

        except Exception as e:
            log.exception("plan_execute.error", task_id=task.id)
            task.status = TaskStatus.ERROR
            yield {"type": "error", "message": str(e)}

        finally:
            await self._stm.save_task(task)
            yield {"type": "done", "task_id": task.id,
                   "status": task.status.value,
                   "usage": {"total_tokens": total_tokens}}

    async def _generate_plan(self, task: AgentTask, config: AgentConfig) -> list[TaskStep]:
        prompt = PLAN_PROMPT.format(goal=task.input.text)
        # 如果 router 支持 plan()，用规划专用路由；否则退化到 summarize
        if hasattr(self._llm, 'plan'):
            from core.models import Message, MessageRole
            plan_msg = [Message(role=MessageRole.USER, content=prompt)]
            resp = await self._llm.plan(plan_msg, config)
            summary_resp = resp.content or ""
        else:
            summary_resp = await self._llm.summarize(prompt, max_tokens=600)
        try:
            clean = summary_resp.strip().lstrip("```json").rstrip("```").strip()
            items = json.loads(clean)
            return [
                TaskStep(goal=item["goal"], tool_hint=item.get("tool_hint"))
                for item in items if "goal" in item
            ]
        except Exception:
            # 无法解析则退化为单步 ReAct
            return [TaskStep(goal=task.input.text)]

    async def _execute_step(
        self, step: TaskStep, task: AgentTask, config: AgentConfig
    ) -> dict[str, Any]:
        """用 ReAct 方式执行单个步骤（最多 5 轮循环）。"""
        react = ReactOrchestrator(
            self._llm, self._skills, self._mcp,
            self._ctx, self._stm, self._ltm,
        )
        # 为子步骤创建临时上下文
        sub_input = task.input.model_copy(update={"text": step.goal})
        sub_task  = AgentTask(
            session_id=task.session_id,
            user_id=task.user_id,
            input=sub_input,
            scratchpad=dict(task.scratchpad),
        )
        sub_config = config.model_copy(update={"max_steps": 5, "stream": False})
        tokens = 0
        last_text = None
        async for event in react.run(sub_task, sub_config):
            if event["type"] == "delta":
                last_text = event["text"]
            elif event["type"] == "done":
                tokens = event.get("usage", {}).get("total_tokens", 0)
            elif event["type"] == "error":
                return {"error": event["message"], "tokens": tokens}
        return {"result": last_text, "tokens": tokens}

    async def _replan_from(
        self, task: AgentTask, from_idx: int, config: AgentConfig
    ) -> list[TaskStep]:
        completed = [s.goal for s in task.plan[:from_idx] if s.status == StepStatus.DONE]
        failed_goal = task.plan[from_idx].goal
        prompt = (
            f"原任务：{task.input.text}\n"
            f"已完成步骤：{completed}\n"
            f"失败步骤：{failed_goal}\n"
            f"请生成替代的后续步骤（JSON 数组格式，同之前格式）。"
        )
        raw = await self._llm.summarize(prompt, max_tokens=400)
        try:
            items = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
            return [TaskStep(goal=item["goal"], tool_hint=item.get("tool_hint")) for item in items]
        except Exception:
            return [TaskStep(goal=f"尝试其他方式完成：{failed_goal}")]

    async def _synthesize(self, task: AgentTask, config: AgentConfig) -> str:
        results = {
            step.id: str(step.result)[:300]
            for step in task.plan if step.status == StepStatus.DONE and step.result
        }
        if not results:
            return "任务已执行，但未产生明确输出。"
        messages = await self._ctx.build(task, [], self._ltm, config.token_budget)
        messages.append(Message(
            role=MessageRole.USER,
            content=f"请根据以上执行结果，给出最终的综合回答。执行结果摘要：{json.dumps(results, ensure_ascii=False)[:500]}",
        ))
        resp = await self._llm.chat(messages, [], config)
        return resp.content or "任务已完成。"
