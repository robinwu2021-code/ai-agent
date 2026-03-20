"""
memory/consolidation.py — 记忆固化服务

在任务完成后异步提取重要信息写入长期记忆。
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from core.models import AgentTask, MemoryEntry, MemoryType

log = structlog.get_logger(__name__)


class MemoryConsolidator:
    """
    将已完成任务的短期记忆固化到长期记忆。
    依赖 LLMEngine 进行摘要和重要性打分。
    """

    EXTRACT_PROMPT = """从以下对话历史中提取值得长期记忆的信息。
返回 JSON 数组，每项包含：
  - text: 要记住的事实（一句话）
  - type: semantic（知识/事实）| episodic（任务经验）| profile（用户偏好）
  - importance: 0.0-1.0 的重要性评分

只提取真正有价值的信息，忽略临时指令和工具调用细节。
直接返回 JSON，不要其他内容。

对话历史：
{history}"""

    def __init__(self, llm_engine: Any, long_term_memory: Any) -> None:
        self._llm = llm_engine
        self._ltm = long_term_memory

    async def consolidate(self, task: AgentTask) -> list[MemoryEntry]:
        """提取并写入长期记忆，返回写入的条目列表。"""
        if not task.history:
            return []

        history_text = self._format_history(task)
        prompt = self.EXTRACT_PROMPT.format(history=history_text)

        # 轻量摘要提取
        # consolidate node_id routes to cheap model if configured
        kw = dict(node_id="consolidate") if hasattr(self._llm, "summarize") else {}
        sum_fn = getattr(self._llm, "summarize", None)
        raw = await sum_fn(prompt, max_tokens=800, **kw) if sum_fn else str(prompt)[:2400]
        facts = self._parse_facts(raw)

        entries: list[MemoryEntry] = []
        for fact in facts:
            entry = MemoryEntry(
                user_id=task.user_id,
                type=MemoryType(fact.get("type", "semantic")),
                text=fact["text"],
                importance=float(fact.get("importance", 0.5)),
                metadata={"task_id": task.id, "task_status": task.status.value},
            )
            await self._ltm.write(entry)
            entries.append(entry)

        # 情节记忆：任务结果摘要
        if task.status.value in ("done", "error"):
            episode = MemoryEntry(
                user_id=task.user_id,
                type=MemoryType.EPISODIC,
                text=f"任务[{task.input.text[:60]}]执行结果: {task.status.value}",
                importance=0.4,
                metadata={"task_id": task.id},
            )
            await self._ltm.write(episode)
            entries.append(episode)

        log.info("memory.consolidated", task_id=task.id, entries=len(entries))

        # 异步裁剪（控制记忆总量）
        await self._ltm.prune(task.user_id, max_items=5000, score_threshold=0.1)

        return entries

    def _format_history(self, task: AgentTask) -> str:
        lines = []
        for msg in task.history[-20:]:  # 最近 20 条
            if msg.role.value == "system":
                continue
            if msg.content:
                lines.append(f"[{msg.role.value}]: {msg.content[:200]}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"[tool_call]: {tc.tool_name}({json.dumps(tc.arguments, ensure_ascii=False)[:100]})")
            if msg.tool_result:
                lines.append(f"[tool_result]: {str(msg.tool_result.content)[:100]}")
        return "\n".join(lines)

    def _parse_facts(self, raw: str) -> list[dict]:
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean)
        except Exception:
            log.warning("consolidation.parse_failed", raw=raw[:200])
            return []
