"""
skills/hub/knowledge_graph/__init__.py — 知识图谱检索 Skill

通过局部子图、全局社区摘要或路径查询三种模式检索知识图谱，
将结果格式化为 Markdown 文本返回给 LLM。
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import yaml

# ── 延迟导入 rag.graph，避免在未安装依赖时导致整个 skills 包加载失败 ──────────


def _import_rag():
    """懒加载 rag.graph 模块，返回 (GraphRetriever, SQLiteKGStore)。"""
    from rag.graph import GraphRetriever, SQLiteKGStore  # noqa: PLC0415
    return GraphRetriever, SQLiteKGStore


# ── ToolDescriptor 兼容层 ─────────────────────────────────────────────────────
# 本项目的 hub skill 约定：skill 类必须有 .descriptor 属性（ToolDescriptor 实例）
# 以及 async execute(arguments: dict) -> Any 方法。
# ToolDescriptor 来自 core.models，这里直接导入。

from core.models import PermissionLevel, ToolDescriptor  # noqa: E402


# ── 主类 ──────────────────────────────────────────────────────────────────────

class KnowledgeGraphSkill:
    """
    知识图谱检索 Skill。

    环境变量：
      KB_STORE_URL  — SQLite 连接字符串，默认 sqlite:///workspace.db
    """

    # ------------------------------------------------------------------
    # 构造与描述符
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        _yaml_path = Path(__file__).parent / "skill.yaml"
        cfg: dict[str, Any] = yaml.safe_load(_yaml_path.read_text(encoding="utf-8"))

        # skill.yaml 使用 "parameters" 键；loader 期望 input_schema
        input_schema = cfg.get("parameters") or cfg.get("input_schema", {"type": "object", "properties": {}})

        self._descriptor = ToolDescriptor(
            name=cfg["name"],
            description=cfg.get("description", "").strip(),
            input_schema=input_schema,
            source="skill",
            permission=PermissionLevel.READ,
            timeout_ms=int(cfg.get("timeout_ms", 30_000)),
            tags=cfg.get("tags", []),
        )

        self._store = None       # SQLiteKGStore，懒初始化
        self._retriever = None   # GraphRetriever，懒初始化

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    # ------------------------------------------------------------------
    # 懒初始化
    # ------------------------------------------------------------------

    async def _ensure_retriever(self) -> None:
        """首次调用时初始化存储层与检索器。"""
        if self._retriever is not None:
            return

        GraphRetriever, SQLiteKGStore = _import_rag()

        store_url = os.getenv("KB_STORE_URL", "sqlite:///workspace.db")
        store = SQLiteKGStore(store_url)
        await store.initialize()

        self._store = store
        self._retriever = GraphRetriever(store)

    # ------------------------------------------------------------------
    # 格式化辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_node(node: Any) -> str:  # node: rag.graph.models.Node
        parts = [f"**{node.name}** [{node.node_type.value}]"]
        if node.description:
            parts.append(f"  _{node.description[:120]}_")
        return "\n".join(parts)

    @staticmethod
    def _fmt_edge(edge: Any, node_map: dict[str, Any]) -> str:
        src_name = node_map.get(edge.src_id, edge.src_id)
        dst_name = node_map.get(edge.dst_id, edge.dst_id)
        if hasattr(src_name, "name"):
            src_name = src_name.name
        if hasattr(dst_name, "name"):
            dst_name = dst_name.name
        line = f"- {src_name} **—[{edge.relation}]→** {dst_name}"
        if edge.context:
            line += f"\n  > {edge.context[:100]}"
        return line

    @staticmethod
    def _fmt_reasoning_chain(chain: list[dict]) -> str:
        if not chain:
            return ""
        steps = []
        for i, step in enumerate(chain, 1):
            src = step.get("src", "?")
            rel = step.get("relation", "?")
            dst = step.get("dst", "?")
            evidence = step.get("evidence", "")
            line = f"{i}. **{src}** —[{rel}]→ **{dst}**"
            if evidence:
                line += f"\n   > {evidence[:100]}"
            steps.append(line)
        return "\n".join(steps)

    # ------------------------------------------------------------------
    # 检索模式实现
    # ------------------------------------------------------------------

    async def _retrieve_local(self, query: str, kb_id: str, hops: int) -> str:
        """局部子图检索。"""
        subgraph = await self._retriever.retrieve_local(
            query=query, kb_id=kb_id, hops=hops
        )
        if not subgraph.nodes:
            return f"在知识库 `{kb_id}` 中未找到与 **{query}** 相关的实体。"

        node_map = {n.id: n for n in subgraph.nodes}

        sections: list[str] = []

        # 实体列表
        sections.append("## 相关实体\n")
        for node in subgraph.nodes[:20]:
            sections.append(self._fmt_node(node))

        # 关系列表
        if subgraph.edges:
            sections.append("\n## 关系网络\n")
            for edge in subgraph.edges[:30]:
                sections.append(self._fmt_edge(edge, node_map))

        # 推理链
        if subgraph.reasoning_chain:
            sections.append("\n## 推理链\n")
            sections.append(self._fmt_reasoning_chain(subgraph.reasoning_chain))

        # 上下文摘要（如果检索器填充了）
        if subgraph.context_text:
            sections.append("\n## 背景摘要\n")
            sections.append(subgraph.context_text[:800])

        return "\n".join(sections)

    async def _retrieve_global(self, query: str, kb_id: str) -> str:
        """全局社区摘要检索。"""
        subgraph = await self._retriever.retrieve_global(query=query, kb_id=kb_id)

        if not subgraph.nodes and not subgraph.context_text:
            return f"在知识库 `{kb_id}` 中暂无全局摘要数据，请先构建知识图谱。"

        sections: list[str] = []

        if subgraph.context_text:
            sections.append("## 全局知识摘要\n")
            sections.append(subgraph.context_text[:1200])

        if subgraph.nodes:
            sections.append("\n## 关键实体\n")
            for node in subgraph.nodes[:10]:
                sections.append(self._fmt_node(node))

        return "\n".join(sections)

    async def _retrieve_path(self, query: str, kb_id: str) -> str:
        """路径查询（两实体间最短路径）。"""
        subgraph = await self._retriever.retrieve_path(query=query, kb_id=kb_id)

        if not subgraph.nodes:
            return f"未能找到查询中两个实体之间的路径。查询：**{query}**"

        node_map = {n.id: n for n in subgraph.nodes}
        sections: list[str] = []

        sections.append("## 路径查询结果\n")

        if subgraph.reasoning_chain:
            sections.append("### 连接路径\n")
            sections.append(self._fmt_reasoning_chain(subgraph.reasoning_chain))

        if subgraph.edges:
            sections.append("\n### 路径上的关系\n")
            for edge in subgraph.edges:
                sections.append(self._fmt_edge(edge, node_map))

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    async def execute(self, arguments: dict) -> str:
        """
        执行知识图谱检索并返回 Markdown 格式的结果字符串。

        参数（来自 LLM tool_call arguments）：
          query  — 用户问题或关键词（必填）
          mode   — local | global | path（默认 local）
          kb_id  — 知识库 ID（默认 global）
          hops   — 局部检索跳数 1-3（默认 2）
        """
        query: str = arguments.get("query", "").strip()
        if not query:
            return "错误：query 参数不能为空。"

        mode: str = arguments.get("mode", "local")
        kb_id: str = arguments.get("kb_id", "global")
        hops: int = max(1, min(3, int(arguments.get("hops", 2))))

        try:
            await self._ensure_retriever()
        except ImportError as exc:
            return (
                f"知识图谱模块未安装，无法执行检索。\n"
                f"请确认 rag.graph 模块已正确安装。\n"
                f"详细错误：{exc}"
            )
        except Exception as exc:
            return (
                f"初始化知识图谱存储失败：{exc}\n"
                f"请检查环境变量 KB_STORE_URL（当前：{os.getenv('KB_STORE_URL', 'sqlite:///workspace.db')}）。"
            )

        try:
            if mode == "global":
                result = await self._retrieve_global(query=query, kb_id=kb_id)
            elif mode == "path":
                result = await self._retrieve_path(query=query, kb_id=kb_id)
            else:
                result = await self._retrieve_local(query=query, kb_id=kb_id, hops=hops)
        except Exception as exc:
            return (
                f"知识图谱检索过程中发生错误：{exc}\n"
                f"查询：{query}，模式：{mode}，知识库：{kb_id}"
            )

        header = textwrap.dedent(f"""\
            # 知识图谱检索结果

            **查询**：{query}
            **模式**：{mode}  **知识库**：{kb_id}

            ---
        """)
        return header + "\n" + result
