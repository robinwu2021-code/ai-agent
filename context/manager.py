"""
context/manager.py — 上下文管理器

按 P0-P5 优先级组装 Prompt，在 token 预算内最大化信息密度。
"""
from __future__ import annotations

from typing import Any

import structlog

from core.models import AgentTask, Message, MessageRole, ToolDescriptor

log = structlog.get_logger(__name__)

# 简单 token 估算：按中英文字符数 / 3（实际生产用 tiktoken）
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)

def _msg_tokens(msg: Message) -> int:
    parts = [msg.content or ""]
    if msg.tool_calls:
        import json
        for tc in msg.tool_calls:
            parts.append(tc.tool_name)
            parts.append(json.dumps(tc.arguments))
    if msg.tool_result:
        parts.append(str(msg.tool_result.content or ""))
    return sum(_estimate_tokens(p) for p in parts) + 4  # role overhead


SYSTEM_PROMPT_TEMPLATE = """你是一个高效的 AI Agent。

## 当前用户
用户 ID: {user_id}

## 行为规范
- 优先使用工具完成任务，不要凭空猜测
- 每次只调用一个工具，观察结果后再决定下一步
- 如果任务已完成，直接输出最终答案，不要再调用工具
- 遇到错误，分析原因后尝试替代方案
- 输出格式：简洁、结构化，优先使用 Markdown

## 当前任务
{task_summary}

## 用户画像
{profile}
"""


class PriorityContextManager:
    """
    按优先级组装上下文，超出 token 预算时从低优先级开始压缩。

    优先级：
      P0 System Prompt    (不可压缩)
      P1 任务状态摘要      (不可压缩)
      P2 长期记忆召回      (可压缩)
      P3 对话历史          (滑动窗口)
      P4 工具描述列表      (可截断)
      P5 用户当前输入      (不可压缩)
    """

    def __init__(self, llm_engine: Any | None = None) -> None:
        self._llm = llm_engine  # 用于压缩，可为 None（降级为截断）

    async def build(
        self,
        task: AgentTask,
        tools: list[ToolDescriptor],
        long_term_memory: Any,
        budget: int,
    ) -> list[Message]:
        used = 0
        messages: list[Message] = []

        # ── P0: System Prompt ─────────────────────────
        profile   = await long_term_memory.get_profile(task.user_id)
        sys_text  = SYSTEM_PROMPT_TEMPLATE.format(
            user_id=task.user_id,
            task_summary=self._task_summary(task),
            profile=self._format_profile(profile),
        )
        sys_tokens = _estimate_tokens(sys_text)
        if used + sys_tokens <= budget:
            messages.append(Message(role=MessageRole.SYSTEM, content=sys_text))
            used += sys_tokens

        # ── P2: 长期记忆 ──────────────────────────────
        mem_budget = min(2000, budget // 6)
        memories   = await long_term_memory.search(
            user_id=task.user_id, query=task.input.text, top_k=5
        )
        if memories:
            mem_text = "## 相关记忆\n" + "\n".join(f"- {m.text}" for m in memories)
            if _estimate_tokens(mem_text) > mem_budget:
                mem_text = mem_text[:mem_budget * 3]  # 简单截断
            messages.append(Message(role=MessageRole.SYSTEM, content=mem_text))
            used += _estimate_tokens(mem_text)

        # ── P3: 对话历史（滑动窗口）─────────────────
        history_budget = min(4000, budget - used - 2000)
        history_msgs   = self._sliding_window(task.history, history_budget)
        messages.extend(history_msgs)
        used += sum(_msg_tokens(m) for m in history_msgs)

        # ── P4: 工具描述 ──────────────────────────────
        # 工具通过 API 的 tools 参数传递，此处仅在无 tools 参数时注入文本
        # 实际已在 LLMEngine.chat() 的 tools 参数中传递，此处略过

        # ── P5: 用户当前输入（必须完整保留）─────────
        user_msg = Message(role=MessageRole.USER, content=task.input.text)
        # 仅当历史中最后一条不是同内容时追加
        if not task.history or task.history[-1].content != task.input.text:
            messages.append(user_msg)

        log.debug(
            "context.built",
            task_id=task.id,
            messages=len(messages),
            est_tokens=used,
            budget=budget,
        )
        return messages

    def _task_summary(self, task: AgentTask) -> str:
        lines = [f"目标：{task.input.text}"]
        if task.plan:
            lines.append(f"执行计划（共 {len(task.plan)} 步）：")
            for i, step in enumerate(task.plan):
                prefix = "✓" if step.status.value == "done" else ("→" if step.status.value == "running" else "○")
                lines.append(f"  {prefix} [{i+1}] {step.goal}")
        if task.scratchpad:
            import json
            lines.append(f"中间状态：{json.dumps(task.scratchpad, ensure_ascii=False)[:200]}")
        return "\n".join(lines)

    def _format_profile(self, profile: dict) -> str:
        if not profile:
            return "（暂无画像数据）"
        return "、".join(f"{k}={v}" for k, v in list(profile.items())[:5])

    def _sliding_window(self, history: list[Message], budget: int) -> list[Message]:
        """保留最近消息，超出预算时从中间删除，始终保留最新 3 条。"""
        if not history:
            return []
        # 始终保留最新 3 条
        tail = history[-3:]
        head = history[:-3]
        tail_tokens = sum(_msg_tokens(m) for m in tail)
        remaining = budget - tail_tokens
        if remaining <= 0:
            return tail
        # 从最新往旧填充 head
        selected = []
        for msg in reversed(head):
            t = _msg_tokens(msg)
            if remaining - t < 0:
                break
            selected.insert(0, msg)
            remaining -= t
        return selected + tail


class SimpleContextManager:
    """极简上下文管理器：直接拼接所有历史，不做压缩。适合短任务/测试。"""

    async def build(
        self,
        task: AgentTask,
        tools: list[ToolDescriptor],
        long_term_memory: Any,
        budget: int,
    ) -> list[Message]:
        msgs = [Message(role=MessageRole.SYSTEM,
                        content=f"你是 AI Agent。任务：{task.input.text}")]
        msgs.extend(task.history)
        if not task.history or task.history[-1].content != task.input.text:
            msgs.append(Message(role=MessageRole.USER, content=task.input.text))
        return msgs
