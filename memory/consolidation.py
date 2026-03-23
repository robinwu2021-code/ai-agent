"""
memory/consolidation.py — 记忆固化服务（Workspace 感知版）

在任务完成后异步提取重要信息写入长期记忆。

优化点（vs 旧版）：
  1. 增量固化：每 INCREMENTAL_THRESHOLD 条新消息自动触发一次（不等任务结束）
  2. 全量历史摘要：不再只取 history[-20]，而是分段摘要后整合
  3. 健壮 JSON 解析：多种格式容错
  4. 外部可调用 consolidate_incremental() 做周期性固化
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog

from core.models import AgentTask, MemoryEntry, MemoryType

log = structlog.get_logger(__name__)

# 增量触发阈值：每增加 N 条消息就固化一次（防止任务超时/崩溃时全量丢失）
INCREMENTAL_THRESHOLD = 10

# 每段历史最多传给 LLM 的消息数
SEGMENT_SIZE = 20

# 旧版兼容 prompt（无工作区）
_PROMPT_PLAIN = """从以下对话历史中提取值得长期记忆的信息。
返回 JSON 数组，每项包含：
  - text: 要记住的事实（一句话，不超过50字）
  - type: semantic（知识/事实）| episodic（任务经验）| profile（用户偏好）
  - importance: 0.0-1.0 的重要性评分

只提取真正有价值的信息，忽略临时指令和工具调用细节。
直接返回 JSON 数组，不要 markdown 代码块，不要其他内容。

对话历史：
{history}"""

# workspace 版 prompt（含 scope 分类）
_PROMPT_WORKSPACE = """从以下对话历史中提取值得长期记忆的信息。
返回 JSON 数组，每项包含：
  - text:       要记住的事实（一句话，不超过50字）
  - type:       semantic（知识/事实）| episodic（任务经验）| profile（用户偏好）
  - importance: 0.0-1.0 的重要性评分
  - scope:      记忆范围
      personal  → 用户自身偏好、习惯、风格（仅用户本人可见）
      project   → 项目决策、活动结果、阶段状态（项目成员可见）
      workspace → 行业知识、品牌规范、产品事实（全工作区可见）

分类示例：
  "用户喜欢简洁的文案风格"                → scope=personal
  "上海门店国庆促销ROI达到3.2倍"          → scope=project
  "品牌配色规范：主色调为品牌橙#FF6B35"    → scope=workspace

只提取真正有价值的信息，不超过8条。
直接返回 JSON 数组，不要 markdown 代码块，不要其他内容。

项目上下文：{project_context}

对话历史：
{history}"""


class MemoryConsolidator:
    """
    将已完成/进行中任务的短期记忆固化到长期记忆。

    提供两种调用方式：
      - consolidate()：任务完成时全量固化（旧接口，向后兼容）
      - consolidate_incremental()：每 N 条消息触发的增量固化
    """

    def __init__(self, llm_engine: Any, long_term_memory: Any) -> None:
        self._llm = llm_engine
        self._ltm = long_term_memory
        # 记录每个 task_id 上次固化时处理到的消息索引
        self._last_consolidated: dict[str, int] = {}

    # ──────────────────────────────────────────────────────────────
    # 对外接口
    # ──────────────────────────────────────────────────────────────

    async def consolidate(
        self,
        task:         AgentTask,
        workspace_id: str | None = None,
        project_id:   str | None = None,
    ) -> list[Any]:
        """
        任务完成时的全量固化。
        处理从上次增量固化位置到结尾的所有消息，避免重复固化。
        """
        if not task.history:
            return []

        start_idx = self._last_consolidated.get(task.id, 0)
        new_msgs  = [m for m in task.history[start_idx:] if m.role.value != "system"]

        if not new_msgs:
            return []

        entries = await self._run_consolidation(
            task         = task,
            messages     = new_msgs,
            workspace_id = workspace_id,
            project_id   = project_id,
        )
        self._last_consolidated[task.id] = len(task.history)

        # 情节记忆
        if task.status.value in ("done", "error"):
            ep_entries = await self._write_episodic(task, workspace_id, project_id)
            entries.extend(ep_entries)
            self._last_consolidated.pop(task.id, None)  # 任务结束，清理状态

        log.info("memory.consolidated",
                 task_id=task.id, entries=len(entries),
                 workspace_id=workspace_id, project_id=project_id)

        await self._safe_prune(task.user_id)
        return entries

    async def consolidate_incremental(
        self,
        task:         AgentTask,
        workspace_id: str | None = None,
        project_id:   str | None = None,
    ) -> list[Any]:
        """
        增量固化：只处理上次固化后的新消息。
        由外部按消息数阈值调用（如 Agent.run() 中每 10 条消息触发）。

        调用示例（在 Agent.run() 的消息追加后）：
            if len(task.history) % INCREMENTAL_THRESHOLD == 0:
                await consolidator.consolidate_incremental(task, ...)
        """
        start_idx = self._last_consolidated.get(task.id, 0)
        new_msgs  = [
            m for m in task.history[start_idx:]
            if m.role.value != "system"
        ]
        if len(new_msgs) < INCREMENTAL_THRESHOLD:
            return []   # 还没到阈值

        entries = await self._run_consolidation(
            task         = task,
            messages     = new_msgs,
            workspace_id = workspace_id,
            project_id   = project_id,
        )
        self._last_consolidated[task.id] = len(task.history)

        log.info("memory.consolidated.incremental",
                 task_id=task.id, entries=len(entries),
                 new_msgs=len(new_msgs))
        return entries

    # ──────────────────────────────────────────────────────────────
    # 内部逻辑
    # ──────────────────────────────────────────────────────────────

    async def _run_consolidation(
        self,
        task:         AgentTask,
        messages:     list[Any],
        workspace_id: str | None,
        project_id:   str | None,
    ) -> list[Any]:
        """
        核心固化逻辑：
          1. 将 messages 分段（每段 SEGMENT_SIZE 条）
          2. 每段调用 LLM 提取 facts
          3. 合并 facts 并写入 LTM
        """
        all_facts: list[dict] = []

        # 分段处理（防止消息太多超出 LLM context）
        for i in range(0, len(messages), SEGMENT_SIZE):
            segment      = messages[i: i + SEGMENT_SIZE]
            history_text = self._format_messages(segment)
            is_workspace_mode = bool(workspace_id)

            if is_workspace_mode:
                project_ctx = (
                    f"workspace_id={workspace_id}, project_id={project_id}"
                    if project_id else f"workspace_id={workspace_id}"
                )
                prompt = _PROMPT_WORKSPACE.format(
                    history         = history_text,
                    project_context = project_ctx,
                )
            else:
                prompt = _PROMPT_PLAIN.format(history=history_text)

            sum_fn = getattr(self._llm, "summarize", None)
            kw     = dict(node_id="consolidate") if hasattr(self._llm, "chat") else {}
            if sum_fn:
                raw = await sum_fn(prompt, max_tokens=1200, **kw)
            else:
                raw = ""

            facts = self._parse_facts(raw)
            all_facts.extend(facts)

        # 写入 LTM
        is_workspace_ltm  = hasattr(self._ltm, "write_scoped")
        is_workspace_mode = bool(workspace_id)
        entries: list[Any] = []

        for fact in all_facts:
            text       = str(fact.get("text", "")).strip()
            mem_type   = fact.get("type", "semantic")
            importance = float(fact.get("importance", 0.5))
            scope_str  = fact.get("scope", "personal")

            if not text or importance < 0.2:  # 过滤极低重要性
                continue

            if is_workspace_ltm and is_workspace_mode:
                from workspace.models import MemoryScope
                try:
                    scope = MemoryScope(scope_str)
                except ValueError:
                    scope = MemoryScope.PERSONAL

                entry = await self._ltm.write_scoped(
                    scope        = scope,
                    author_id    = task.user_id,
                    text         = text,
                    workspace_id = workspace_id,
                    project_id   = project_id if scope == MemoryScope.PROJECT else None,
                    memory_type  = mem_type,
                    importance   = importance,
                    metadata     = {"task_id": task.id, "task_status": task.status.value},
                )
            else:
                entry = MemoryEntry(
                    user_id    = task.user_id,
                    type       = MemoryType(mem_type) if mem_type in ("semantic","episodic","profile") else MemoryType.SEMANTIC,
                    text       = text,
                    importance = importance,
                    metadata   = {"task_id": task.id, "task_status": task.status.value},
                )
                await self._ltm.write(entry)

            entries.append(entry)

        return entries

    async def _write_episodic(
        self,
        task:         AgentTask,
        workspace_id: str | None,
        project_id:   str | None,
    ) -> list[Any]:
        """写入任务结果的情节记忆。"""
        episode_text = f"任务[{task.input.text[:60]}]执行结果: {task.status.value}"
        is_workspace_ltm  = hasattr(self._ltm, "write_scoped")
        is_workspace_mode = bool(workspace_id)

        if is_workspace_ltm and is_workspace_mode:
            from workspace.models import MemoryScope
            ep = await self._ltm.write_scoped(
                scope        = MemoryScope.PERSONAL,
                author_id    = task.user_id,
                text         = episode_text,
                workspace_id = workspace_id,
                memory_type  = "episodic",
                importance   = 0.4,
                metadata     = {"task_id": task.id},
            )
        else:
            ep = MemoryEntry(
                user_id    = task.user_id,
                type       = MemoryType.EPISODIC,
                text       = episode_text,
                importance = 0.4,
                metadata   = {"task_id": task.id},
            )
            await self._ltm.write(ep)
        return [ep]

    async def _safe_prune(self, user_id: str) -> None:
        try:
            await self._ltm.prune(user_id, max_items=5000, score_threshold=0.1)
        except Exception as exc:
            log.warning("memory.prune.failed", user_id=user_id, error=str(exc))

    # ──────────────────────────────────────────────────────────────
    # 格式化工具
    # ──────────────────────────────────────────────────────────────

    def _format_messages(self, messages: list[Any]) -> str:
        lines = []
        for msg in messages:
            if msg.role.value == "system":
                continue
            if msg.content:
                # 适当增加截断长度（旧版 200，新版 500）
                lines.append(f"[{msg.role.value}]: {msg.content[:500]}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(
                        f"[tool_call]: {tc.tool_name}"
                        f"({json.dumps(tc.arguments, ensure_ascii=False)[:150]})"
                    )
            if msg.tool_result:
                lines.append(f"[tool_result]: {str(msg.tool_result.content)[:200]}")
        return "\n".join(lines)

    def _parse_facts(self, raw: str) -> list[dict]:
        """
        健壮的 JSON 解析：
          1. 直接 json.loads
          2. 去除 ```json ... ``` markdown 包裹
          3. 用正则提取第一个 [...] 数组
          4. 全部失败返回 []
        """
        if not raw or not raw.strip():
            return []

        clean = raw.strip()

        # 尝试 1：直接解析
        try:
            data = json.loads(clean)
            if isinstance(data, list):
                return data
        except Exception:
            pass

        # 尝试 2：去 markdown 代码块
        if "```" in clean:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if isinstance(data, list):
                        return data
                except Exception:
                    pass

        # 尝试 3：正则找 JSON 数组
        match = re.search(r"\[[\s\S]*\]", clean)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        log.warning("consolidation.parse_failed", raw=raw[:300])
        return []
