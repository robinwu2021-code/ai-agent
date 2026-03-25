"""
workspace/memory.py — 工作区感知的混合记忆系统

═══════════════════════════════════════════════════════════════════════
  写入策略（如何记）
═══════════════════════════════════════════════════════════════════════

1. 按范围写入
   write_personal(user_id, text, ...)
     → scope=PERSONAL, owner_id=user_id
     → 仅该用户在任意工作区均可见

   write_project(workspace_id, project_id, author_id, text, ...)
     → scope=PROJECT, owner_id="proj::{ws_id}::{proj_id}"
     → 该项目全体成员可见

   write_workspace(workspace_id, author_id, text, ...)
     → scope=WORKSPACE, owner_id="ws::{ws_id}"
     → 该工作区全体成员可见

2. 跨项目共享（不复制数据，只建授权记录）
   share_to_projects(entry_id, [proj_b, proj_c], by_user_id)
     → 在 memory_shares 表中插入授权行
     → 读取时 Layer-3 自动包含

3. 自动整合（Consolidation）—— 由 LLM 决定 scope
   LLM 分析对话，对每条提取结果打上 scope 标签：
     personal  → 个人偏好、习惯
     project   → 项目决策、状态、结果
     workspace → 行业知识、品牌规范

═══════════════════════════════════════════════════════════════════════
  读取策略（如何读）
═══════════════════════════════════════════════════════════════════════

search_mixed(user_id, workspace_id, project_id, query, top_k) 并发执行 4 层查询：

  Layer 1 — 个人记忆 (weight=1.0, pool=200)
    SELECT * WHERE scope='personal' AND owner_id=user_id

  Layer 2 — 项目记忆 (weight=0.9, pool=200)
    SELECT * WHERE scope='project' AND owner_id='proj::{ws}::{proj}'

  Layer 3 — 跨项目共享 (weight=0.85, pool=100)
    SELECT me.* JOIN memory_shares ms WHERE ms.shared_to_project_id=project_id

  Layer 4 — 工作区知识 (weight=0.7, pool=150)
    SELECT * WHERE scope='workspace' AND workspace_id=workspace_id

每层分别用 n-gram 相关度评分：
  relevance = |text_ngrams ∩ query_ngrams| / |query_ngrams|

最终分数：
  score = relevance × layer_weight × importance

合并所有层结果 → 按 score 排序 → 去重（same entry_id）→ 返回 top_k

═══════════════════════════════════════════════════════════════════════
  Context 注入格式（如何注入到 Prompt）
═══════════════════════════════════════════════════════════════════════

[个人记忆] 用户偏好简洁直接的文案风格（0.92）
[项目记忆] 8月份折扣活动ROI为3.2倍（0.87）
[共享记忆 来自:运营项目] 竞品上周上线了类似活动（0.81）
[工作区知识] 品牌规范：禁止使用红色以外的折扣标签（0.74）
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Any

import structlog

from workspace.models import (
    MemoryScope,
    MemoryShare,
    RankedEntry,
    Workspace,
    WorkspaceMemoryEntry,
)
from workspace.store import WorkspaceStore

log = structlog.get_logger(__name__)

# 各层检索权重
_LAYER_WEIGHTS = {
    "personal":  1.00,
    "project":   0.90,
    "shared":    0.85,
    "workspace": 0.70,
}

# 各层候选池大小（拉取后在内存做相关度重排）
_LAYER_POOLS = {
    "personal":  200,
    "project":   200,
    "shared":    100,
    "workspace": 150,
}


# ──────────────────────────────────────────────────────────────
# 文本相关度（n-gram + 词级别）
# ──────────────────────────────────────────────────────────────

def _ngram_score(text: str, query: str, n: int = 2) -> float:
    """
    双层 n-gram 相关度：字符 bigram + 词级别 unigram，取最大值。
    无需安装 jieba 或 FTS，纯 Python 实现，适合 SQLite/MySQL 场景。
    """
    if not query.strip():
        return 0.0
    tl, ql = text.lower(), query.lower()

    def char_ngrams(s: str, k: int) -> set[str]:
        return {s[i:i+k] for i in range(len(s) - k + 1)} if len(s) >= k else set()

    def words(s: str) -> set[str]:
        return {w for w in s.split() if w}

    # 字符 bigram 相似度
    tng = char_ngrams(tl, n)
    qng = char_ngrams(ql, n)
    char_score = len(tng & qng) / len(qng) if qng else 0.0

    # 词级别召回
    tw = words(tl)
    qw = words(ql)
    word_score = len(tw & qw) / len(qw) if qw else 0.0

    return max(char_score, word_score)


def _score(entry: WorkspaceMemoryEntry, query: str, layer_weight: float) -> float:
    rel = _ngram_score(entry.text, query)
    # 时间衰减：超过 30 天的记忆轻微降权
    days = max(0, (datetime.utcnow() - entry.created_at).days)
    decay = math.exp(-0.005 * days)  # 30天约0.86，90天约0.64
    return rel * layer_weight * entry.importance * decay


# ──────────────────────────────────────────────────────────────
# WorkspaceAwareLTM
# ──────────────────────────────────────────────────────────────

class WorkspaceAwareLTM:
    """
    工作区感知的长期记忆管理器。

    职责：
      - 所有 MemoryEntry 的读写路由
      - 4 层混合检索（personal / project / shared / workspace）
      - 跨项目共享管理
      - 兼容旧版 InMemoryLongTermMemory 接口（write / search / get_profile）
    """

    def __init__(self, store: WorkspaceStore) -> None:
        self._store = store

    # ══════════════════════════════════════════════════════════════
    # 写入 API
    # ══════════════════════════════════════════════════════════════

    async def write_personal(
        self,
        user_id:     str,
        text:        str,
        memory_type: str   = "semantic",
        importance:  float = 0.5,
        tags:        list[str] | None = None,
        metadata:    dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> WorkspaceMemoryEntry:
        """写入个人记忆（仅 user_id 本人可见）。"""
        entry = WorkspaceMemoryEntry(
            scope=MemoryScope.PERSONAL,
            owner_id=user_id,
            workspace_id=workspace_id,
            author_id=user_id,
            memory_type=memory_type,
            text=text,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )
        await self._store.save_memory(entry)
        log.debug("ltm.write_personal", entry_id=entry.entry_id, user_id=user_id)
        return entry

    async def write_project(
        self,
        workspace_id: str,
        project_id:   str,
        author_id:    str,
        text:         str,
        memory_type:  str   = "semantic",
        importance:   float = 0.6,
        tags:         list[str] | None = None,
        metadata:     dict[str, Any] | None = None,
    ) -> WorkspaceMemoryEntry:
        """写入项目记忆（项目全体成员可见）。"""
        owner = Workspace.proj_memory_key(workspace_id, project_id)
        entry = WorkspaceMemoryEntry(
            scope=MemoryScope.PROJECT,
            owner_id=owner,
            workspace_id=workspace_id,
            project_id=project_id,
            author_id=author_id,
            memory_type=memory_type,
            text=text,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )
        await self._store.save_memory(entry)
        log.debug("ltm.write_project", entry_id=entry.entry_id, project_id=project_id)
        return entry

    async def write_workspace(
        self,
        workspace_id: str,
        author_id:    str,
        text:         str,
        memory_type:  str   = "semantic",
        importance:   float = 0.7,
        tags:         list[str] | None = None,
        metadata:     dict[str, Any] | None = None,
    ) -> WorkspaceMemoryEntry:
        """写入工作区知识（全工作区成员可见）。"""
        owner = f"ws::{workspace_id}"
        entry = WorkspaceMemoryEntry(
            scope=MemoryScope.WORKSPACE,
            owner_id=owner,
            workspace_id=workspace_id,
            author_id=author_id,
            memory_type=memory_type,
            text=text,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )
        await self._store.save_memory(entry)
        log.debug("ltm.write_workspace", entry_id=entry.entry_id, workspace_id=workspace_id)
        return entry

    async def write_scoped(
        self,
        scope:        MemoryScope,
        author_id:    str,
        text:         str,
        workspace_id: str | None = None,
        project_id:   str | None = None,
        memory_type:  str   = "semantic",
        importance:   float = 0.5,
        tags:         list[str] | None = None,
        metadata:     dict[str, Any] | None = None,
    ) -> WorkspaceMemoryEntry:
        """
        统一写入入口，根据 scope 自动路由。
        Consolidation 使用此方法写入 LLM 分类后的条目。
        """
        if scope == MemoryScope.PERSONAL:
            return await self.write_personal(
                user_id=author_id, text=text, memory_type=memory_type,
                importance=importance, tags=tags, metadata=metadata,
                workspace_id=workspace_id,
            )
        elif scope == MemoryScope.PROJECT:
            if not workspace_id or not project_id:
                raise ValueError("PROJECT scope 需要 workspace_id 和 project_id")
            return await self.write_project(
                workspace_id=workspace_id, project_id=project_id,
                author_id=author_id, text=text, memory_type=memory_type,
                importance=importance, tags=tags, metadata=metadata,
            )
        elif scope == MemoryScope.WORKSPACE:
            if not workspace_id:
                raise ValueError("WORKSPACE scope 需要 workspace_id")
            return await self.write_workspace(
                workspace_id=workspace_id, author_id=author_id, text=text,
                memory_type=memory_type, importance=importance,
                tags=tags, metadata=metadata,
            )
        raise ValueError(f"未知 scope: {scope}")

    # ══════════════════════════════════════════════════════════════
    # 跨项目共享
    # ══════════════════════════════════════════════════════════════

    async def share_to_projects(
        self,
        entry_id:      str,
        to_project_ids: list[str],
        shared_by:     str,
        permission:    str = "read",
        note:          str = "",
    ) -> list[MemoryShare]:
        """
        将一条记忆授权给多个目标项目可见（不复制数据，只建授权行）。

        场景示例：
          - 营销项目的"竞品分析"共享给运营项目
          - 管理员将"品牌手册"下发给所有项目（先 write_workspace，再无需共享）
          - 用户主动分享个人洞察给当前项目
        """
        shares = []
        for pid in to_project_ids:
            share = MemoryShare(
                entry_id=entry_id,
                shared_to_project_id=pid,
                shared_by_user_id=shared_by,
                permission=permission,
                note=note,
            )
            await self._store.share_memory(share)
            shares.append(share)
            log.info("ltm.shared", entry_id=entry_id, to_project=pid, by=shared_by)
        return shares

    async def revoke_share(self, entry_id: str, project_id: str) -> None:
        await self._store.revoke_share(entry_id, project_id)

    async def list_shares(self, entry_id: str) -> list[MemoryShare]:
        return await self._store.list_shares(entry_id)

    # ══════════════════════════════════════════════════════════════
    # 读取 API — 核心：4 层混合检索
    # ══════════════════════════════════════════════════════════════

    async def search_mixed(
        self,
        query:        str,
        user_id:      str,
        workspace_id: str | None = None,
        project_id:   str | None = None,
        top_k:        int  = 10,
        min_score:    float = 0.05,
    ) -> list[RankedEntry]:
        """
        4 层并发检索，合并去重后按综合得分返回 top_k 条。

        返回的 RankedEntry.context_label 可直接作为 prompt 前缀：
          "[个人记忆]"、"[项目记忆]"、"[共享记忆 来自:xxx]"、"[工作区知识]"
        """
        tasks: list[asyncio.Task] = []

        async def layer1() -> list[RankedEntry]:
            pool = await self._store.query_personal(user_id, _LAYER_POOLS["personal"])
            return self._rank(pool, query, "personal", "[个人记忆]")

        tasks.append(asyncio.create_task(layer1()))

        if workspace_id and project_id:
            async def layer2() -> list[RankedEntry]:
                pool = await self._store.query_project(
                    workspace_id, project_id, _LAYER_POOLS["project"]
                )
                return self._rank(pool, query, "project", "[项目记忆]")

            async def layer3() -> list[RankedEntry]:
                pool = await self._store.query_shared_to_project(
                    project_id, _LAYER_POOLS["shared"]
                )
                return self._rank(pool, query, "shared", "[共享记忆]")

            tasks.append(asyncio.create_task(layer2()))
            tasks.append(asyncio.create_task(layer3()))

        if workspace_id:
            async def layer4() -> list[RankedEntry]:
                pool = await self._store.query_workspace(
                    workspace_id, _LAYER_POOLS["workspace"]
                )
                return self._rank(pool, query, "workspace", "[工作区知识]")

            tasks.append(asyncio.create_task(layer4()))

        results_per_layer: list[list[RankedEntry]] = await asyncio.gather(*tasks)

        # 合并、去重、过滤、排序
        seen: set[str] = set()
        merged: list[RankedEntry] = []
        for layer_results in results_per_layer:
            for re in layer_results:
                if re.entry.entry_id not in seen and re.score >= min_score:
                    seen.add(re.entry.entry_id)
                    merged.append(re)

        merged.sort(key=lambda x: x.score, reverse=True)
        top = merged[:top_k]

        # 批量更新 accessed_at（后台执行，不阻塞返回）
        async def _touch_all() -> None:
            for re in top:
                await self._store.touch_memory(re.entry.entry_id)

        asyncio.create_task(_touch_all())

        return top

    def _rank(
        self,
        pool:          list[WorkspaceMemoryEntry],
        query:         str,
        layer:         str,
        label_prefix:  str,
    ) -> list[RankedEntry]:
        weight = _LAYER_WEIGHTS[layer]
        ranked = []
        for entry in pool:
            s = _score(entry, query, weight)
            # 丰富 label：共享记忆带来源
            if layer == "shared" and entry.project_id:
                label = f"[共享记忆 来自:{entry.project_id}]"
            else:
                label = label_prefix
            ranked.append(RankedEntry(entry=entry, score=s, layer=layer, context_label=label))
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    # ══════════════════════════════════════════════════════════════
    # 个人记忆（无工作区上下文）
    # ══════════════════════════════════════════════════════════════

    async def search_personal(
        self,
        query:   str,
        user_id: str,
        top_k:   int = 5,
    ) -> list[RankedEntry]:
        """仅查个人记忆，用于无工作区的纯个人对话。"""
        pool = await self._store.query_personal(user_id, 200)
        ranked = self._rank(pool, query, "personal", "[个人记忆]")
        return ranked[:top_k]

    # ══════════════════════════════════════════════════════════════
    # 用户画像
    # ══════════════════════════════════════════════════════════════

    async def get_profile(self, user_id: str, workspace_id: str = "") -> dict[str, Any]:
        return await self._store.get_profile(user_id, workspace_id)

    async def update_profile(
        self, user_id: str, workspace_id: str = "", data: dict[str, Any] = {}
    ) -> None:
        await self._store.update_profile(user_id, workspace_id, data)

    # ══════════════════════════════════════════════════════════════
    # 旧版兼容接口（给 MemoryConsolidator 用）
    # ══════════════════════════════════════════════════════════════

    async def write(
        self,
        text:        str,
        user_id:     str,
        memory_type: str   = "semantic",
        importance:  float = 0.5,
        metadata:    dict[str, Any] | None = None,
        # workspace 扩展参数（可选）
        scope:        MemoryScope = MemoryScope.PERSONAL,
        workspace_id: str | None = None,
        project_id:   str | None = None,
    ) -> WorkspaceMemoryEntry:
        """兼容旧版 InMemoryLongTermMemory.write() 接口。"""
        return await self.write_scoped(
            scope=scope,
            author_id=user_id,
            text=text,
            workspace_id=workspace_id,
            project_id=project_id,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )

    async def search(
        self,
        user_id:      str,
        query:        str,
        top_k:        int  = 5,
        workspace_id: str | None = None,
        project_id:   str | None = None,
    ) -> list[WorkspaceMemoryEntry]:
        """兼容旧版接口，返回原始 entries（不带分数）。"""
        ranked = await self.search_mixed(
            query=query, user_id=user_id,
            workspace_id=workspace_id, project_id=project_id,
            top_k=top_k,
        )
        return [r.entry for r in ranked]

    async def prune(
        self,
        user_id: str,
        max_items: int = 5000,
        score_threshold: float = 0.1,
    ) -> int:
        return await self._store.prune_personal(user_id, max_items, score_threshold)

    # ══════════════════════════════════════════════════════════════
    # Context 注入格式化
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def format_for_context(ranked: list[RankedEntry], max_chars: int = 1200) -> str:
        """
        将混合记忆格式化为 Prompt 字符串，带来源标注和得分。

        输出示例：
          [个人记忆] 用户偏好简洁直接的文案风格
          [项目记忆] 8月份折扣活动ROI为3.2倍
          [共享记忆 来自:proj_ops] 竞品上周上线了类似活动
          [工作区知识] 品牌规范：禁止使用红色以外的折扣标签
        """
        if not ranked:
            return ""
        lines = []
        total = 0
        for r in ranked:
            line = f"{r.context_label} {r.entry.text}"
            total += len(line)
            if total > max_chars:
                break
            lines.append(line)
        return "\n".join(lines)
