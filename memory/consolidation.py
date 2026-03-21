"""
memory/consolidation.py — 记忆固化服务（Workspace 感知版）

在任务完成后异步提取重要信息写入长期记忆。

workspace 模式下，LLM 还会对每条记忆打上 scope 标签：
  personal  → 用户个人偏好、习惯、风格
  project   → 项目决策、状态、活动结果
  workspace → 行业知识、品牌规范、通用事实

没有工作区上下文时，所有记忆写入 scope=personal（兼容旧行为）。
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from core.models import AgentTask, MemoryEntry, MemoryType

log = structlog.get_logger(__name__)


# 不含 scope 的旧版 prompt（无工作区时使用）
_PROMPT_PLAIN = """从以下对话历史中提取值得长期记忆的信息。
返回 JSON 数组，每项包含：
  - text: 要记住的事实（一句话）
  - type: semantic（知识/事实）| episodic（任务经验）| profile（用户偏好）
  - importance: 0.0-1.0 的重要性评分

只提取真正有价值的信息，忽略临时指令和工具调用细节。
直接返回 JSON，不要其他内容。

对话历史：
{history}"""

# 含 scope 的工作区版 prompt
_PROMPT_WORKSPACE = """从以下对话历史中提取值得长期记忆的信息。
返回 JSON 数组，每项包含：
  - text:       要记住的事实（一句话）
  - type:       semantic（知识/事实）| episodic（任务经验）| profile（用户偏好）
  - importance: 0.0-1.0 的重要性评分
  - scope:      记忆存储范围
      personal  → 关于用户自身的偏好、习惯、风格（仅用户本人可见）
      project   → 项目决策、活动结果、阶段状态（项目成员可见）
      workspace → 行业知识、品牌规范、产品事实（全工作区可见）

分类示例：
  "用户喜欢简洁的文案风格"                → scope=personal
  "上海门店国庆促销ROI达到3.2倍"          → scope=project
  "品牌配色规范：主色调为品牌橙#FF6B35"    → scope=workspace

只提取真正有价值的信息，忽略临时指令和工具调用细节。
直接返回 JSON，不要其他内容。

项目上下文：{project_context}

对话历史：
{history}"""


class MemoryConsolidator:
    """
    将已完成任务的短期记忆固化到长期记忆。
    依赖 LLMEngine 进行摘要和重要性打分。

    workspace_ltm: WorkspaceAwareLTM 实例（优先），或旧版 InMemoryLongTermMemory。
    """

    def __init__(self, llm_engine: Any, long_term_memory: Any) -> None:
        self._llm = llm_engine
        self._ltm = long_term_memory

    async def consolidate(
        self,
        task:         AgentTask,
        workspace_id: str | None = None,
        project_id:   str | None = None,
    ) -> list[Any]:
        """
        提取并写入长期记忆，返回写入的条目列表。

        workspace_id / project_id 均可选：
          - 两者都有 → workspace 模式，LLM 分三范围分类
          - 只有 workspace_id → 个人 + workspace 两级
          - 都没有 → 纯个人模式（向后兼容）
        """
        if not task.history:
            return []

        history_text = self._format_history(task)
        is_workspace_mode = bool(workspace_id)

        if is_workspace_mode:
            project_ctx = (
                f"workspace_id={workspace_id}, project_id={project_id}"
                if project_id
                else f"workspace_id={workspace_id}"
            )
            prompt = _PROMPT_WORKSPACE.format(
                history=history_text,
                project_context=project_ctx,
            )
        else:
            prompt = _PROMPT_PLAIN.format(history=history_text)

        # 调用 LLM 提取
        sum_fn = getattr(self._llm, "summarize", None)
        kw     = dict(node_id="consolidate") if hasattr(self._llm, "chat") else {}
        if sum_fn:
            raw = await sum_fn(prompt, max_tokens=1000, **kw)
        else:
            raw = ""

        facts = self._parse_facts(raw)
        entries = []

        # 判断是否为 WorkspaceAwareLTM
        is_workspace_ltm = hasattr(self._ltm, "write_scoped")

        for fact in facts:
            text       = fact.get("text", "").strip()
            mem_type   = fact.get("type", "semantic")
            importance = float(fact.get("importance", 0.5))
            scope_str  = fact.get("scope", "personal")

            if not text:
                continue

            if is_workspace_ltm and is_workspace_mode:
                from workspace.models import MemoryScope
                try:
                    scope = MemoryScope(scope_str)
                except ValueError:
                    scope = MemoryScope.PERSONAL

                entry = await self._ltm.write_scoped(
                    scope=scope,
                    author_id=task.user_id,
                    text=text,
                    workspace_id=workspace_id,
                    project_id=project_id if scope == MemoryScope.PROJECT else None,
                    memory_type=mem_type,
                    importance=importance,
                    metadata={"task_id": task.id, "task_status": task.status.value},
                )
            else:
                # 旧版接口：全部写 personal
                entry = MemoryEntry(
                    user_id=task.user_id,
                    type=MemoryType(mem_type) if mem_type in ("semantic","episodic","profile") else MemoryType.SEMANTIC,
                    text=text,
                    importance=importance,
                    metadata={"task_id": task.id, "task_status": task.status.value},
                )
                await self._ltm.write(entry)

            entries.append(entry)

        # 情节记忆：任务结果摘要
        if task.status.value in ("done", "error"):
            episode_text = f"任务[{task.input.text[:60]}]执行结果: {task.status.value}"
            if is_workspace_ltm and is_workspace_mode:
                from workspace.models import MemoryScope
                ep = await self._ltm.write_scoped(
                    scope=MemoryScope.PERSONAL,
                    author_id=task.user_id,
                    text=episode_text,
                    workspace_id=workspace_id,
                    memory_type="episodic",
                    importance=0.4,
                    metadata={"task_id": task.id},
                )
            else:
                ep = MemoryEntry(
                    user_id=task.user_id,
                    type=MemoryType.EPISODIC,
                    text=episode_text,
                    importance=0.4,
                    metadata={"task_id": task.id},
                )
                await self._ltm.write(ep)
            entries.append(ep)

        log.info(
            "memory.consolidated",
            task_id=task.id,
            entries=len(entries),
            workspace_id=workspace_id,
            project_id=project_id,
        )

        # 裁剪（仅针对个人记忆）
        await self._ltm.prune(task.user_id, max_items=5000, score_threshold=0.1)
        return entries

    # ──────────────────────────────────────────────────────────────

    def _format_history(self, task: AgentTask) -> str:
        lines = []
        for msg in task.history[-20:]:
            if msg.role.value == "system":
                continue
            if msg.content:
                lines.append(f"[{msg.role.value}]: {msg.content[:200]}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(
                        f"[tool_call]: {tc.tool_name}"
                        f"({json.dumps(tc.arguments, ensure_ascii=False)[:100]})"
                    )
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
            data = json.loads(clean)
            if isinstance(data, list):
                return data
        except Exception:
            log.warning("consolidation.parse_failed", raw=raw[:200])
        return []
