"""rag/graph/community.py — 社区检测与摘要"""
from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any, Callable

import structlog

from rag.graph.models import Community, Edge, Node
from rag.graph.store import KGStore

logger = structlog.get_logger(__name__)

_COMMUNITY_SUMMARY_PROMPT_TEMPLATE = """\
以下是知识图谱中一组相关实体和它们之间的关系。请生成一段100-200字的摘要，
描述这个主题群的核心内容、主要实体以及它们之间的关键关系。

实体：
{entities_text}

关系：
{relations_text}

摘要："""


# ---------------------------------------------------------------------------
# Label Propagation Community Detection
# ---------------------------------------------------------------------------

class CommunityDetector:
    """
    基于标签传播算法（Label Propagation）的社区检测器。

    不依赖 networkx/cdlib，纯 Python 实现。
    """

    def detect(
        self,
        nodes: list[Node],
        edges: list[Edge],
        iterations: int = 10,
        seed: int | None = None,
    ) -> dict[str, str]:
        """
        运行标签传播算法，返回 node_id -> community_id 的映射。

        Algorithm:
        1. Initialize each node with its own label (= node_id)
        2. In each iteration, for each node (in random order):
           - Look at all neighbor labels
           - Assign the most frequent neighbor label
           - Ties broken randomly
        3. After convergence, nodes with the same label form a community

        Args:
            nodes: 图中所有节点。
            edges: 图中所有边。
            iterations: 最大迭代次数（默认 10）。
            seed: 随机种子，用于可重复实验。

        Returns:
            dict[str, str]: node_id -> community_label 的映射。
        """
        if not nodes:
            return {}

        rng = random.Random(seed)

        # Build adjacency list (undirected)
        adjacency: dict[str, set[str]] = {n.id: set() for n in nodes}
        node_ids_set = {n.id for n in nodes}
        for edge in edges:
            if edge.src_id in adjacency and edge.dst_id in adjacency:
                adjacency[edge.src_id].add(edge.dst_id)
                adjacency[edge.dst_id].add(edge.src_id)

        # Initialize labels: each node has its own id as label
        labels: dict[str, str] = {n.id: n.id for n in nodes}
        node_ids = [n.id for n in nodes]

        for iteration in range(iterations):
            changed = False
            # Process nodes in random order to avoid order bias
            shuffled = list(node_ids)
            rng.shuffle(shuffled)

            for nid in shuffled:
                neighbors = adjacency[nid]
                if not neighbors:
                    continue

                # Count neighbor label frequencies
                label_counts: dict[str, int] = {}
                for neighbor_id in neighbors:
                    lbl = labels[neighbor_id]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                if not label_counts:
                    continue

                # Find max frequency
                max_count = max(label_counts.values())
                best_labels = [
                    lbl for lbl, cnt in label_counts.items() if cnt == max_count
                ]

                # Pick randomly among ties
                new_label = rng.choice(best_labels)
                if new_label != labels[nid]:
                    labels[nid] = new_label
                    changed = True

            if not changed:
                logger.debug(
                    "label_propagation_converged",
                    iteration=iteration,
                    num_nodes=len(node_ids),
                )
                break

        # Map labels to community IDs (stable, sorted label -> uuid)
        unique_labels = sorted(set(labels.values()))
        label_to_community: dict[str, str] = {}
        for lbl in unique_labels:
            label_to_community[lbl] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"community:{lbl}"))

        return {nid: label_to_community[lbl] for nid, lbl in labels.items()}


# ---------------------------------------------------------------------------
# Community Summarizer
# ---------------------------------------------------------------------------

class CommunitySummarizer:
    """
    社区摘要生成器：运行社区检测，并为每个社区生成 LLM 摘要。

    Args:
        llm_engine: 具备 generate(prompt) -> str 接口的 LLM 引擎。
        store: KGStore 实例。
        embed_fn: 嵌入函数 (text: str) -> list[float]，用于为社区生成向量。
        min_community_size: 最小社区节点数，小于此值的社区不生成摘要（默认 3）。
    """

    def __init__(
        self,
        llm_engine: Any,
        store: KGStore,
        embed_fn: Callable[[str], Any] | None = None,
        min_community_size: int = 3,
    ) -> None:
        self._llm = llm_engine
        self._store = store
        self._embed_fn = embed_fn
        self._min_size = min_community_size
        self._detector = CommunityDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def summarize_all(self, kb_id: str, level: int = 0) -> dict:
        """
        对知识库中的所有节点和边运行社区检测，并为每个社区生成 LLM 摘要。

        Steps:
        1. Fetch all nodes + edges for kb_id
        2. Run Label Propagation
        3. For each community with >= min_community_size nodes: generate LLM summary
        4. Store Community objects in store with embeddings

        Args:
            kb_id: 知识库 ID。
            level: 社区层级（0=叶级，用于后续分层聚合）。

        Returns:
            dict: {"communities_created": int, "communities_skipped": int}
        """
        logger.info("summarize_all_start", kb_id=kb_id, level=level)

        # Fetch full graph
        nodes, edges = await self._store.get_full_graph(kb_id, limit=5000)
        if not nodes:
            logger.info("summarize_all_no_nodes", kb_id=kb_id)
            return {"communities_created": 0, "communities_skipped": 0}

        # Run community detection
        node_to_community: dict[str, str] = self._detector.detect(nodes, edges)

        # Group nodes by community
        community_to_nodes: dict[str, list[str]] = {}
        for node_id, comm_id in node_to_community.items():
            community_to_nodes.setdefault(comm_id, []).append(node_id)

        # Build node lookup map
        nodes_map: dict[str, Node] = {n.id: n for n in nodes}

        # Summarize each qualifying community
        created = 0
        skipped = 0

        for comm_id, node_ids in community_to_nodes.items():
            if len(node_ids) < self._min_size:
                skipped += 1
                continue

            try:
                community = await self._summarize_community(
                    node_ids=node_ids,
                    nodes_map=nodes_map,
                    edges=edges,
                    kb_id=kb_id,
                    community_id=comm_id,
                    level=level,
                )
                await self._store.upsert_community(community)
                created += 1
                logger.debug(
                    "community_summarized",
                    community_id=comm_id,
                    node_count=len(node_ids),
                )
            except Exception as exc:
                logger.warning(
                    "community_summarize_failed",
                    community_id=comm_id,
                    error=str(exc),
                )
                skipped += 1

        logger.info(
            "summarize_all_done",
            kb_id=kb_id,
            level=level,
            created=created,
            skipped=skipped,
        )
        return {"communities_created": created, "communities_skipped": skipped}

    async def _summarize_community(
        self,
        node_ids: list[str],
        nodes_map: dict[str, Node],
        edges: list[Edge],
        kb_id: str,
        community_id: str,
        level: int = 0,
    ) -> Community:
        """
        为单个社区生成 LLM 摘要，返回 Community 对象。

        Args:
            node_ids: 社区中的节点 ID 列表。
            nodes_map: 全图节点 id -> Node 的查找表。
            edges: 全图所有边（用于筛选社区内部边）。
            kb_id: 知识库 ID。
            community_id: 社区 ID。
            level: 社区层级。

        Returns:
            Community: 含摘要和嵌入向量的社区对象。
        """
        node_ids_set = set(node_ids)
        community_nodes = [nodes_map[nid] for nid in node_ids if nid in nodes_map]

        # Filter edges to only intra-community edges
        internal_edges = [
            e for e in edges
            if e.src_id in node_ids_set and e.dst_id in node_ids_set
        ]

        # Build entities text
        entities_lines: list[str] = []
        for node in community_nodes:
            type_label = node.node_type.value
            desc = node.description.strip() if node.description else "无描述"
            entities_lines.append(f"- {node.name} ({type_label}): {desc}")
        entities_text = "\n".join(entities_lines) if entities_lines else "无实体"

        # Build relations text
        relations_lines: list[str] = []
        # Build node id -> name lookup for display
        id_to_name = {n.id: n.name for n in community_nodes}
        for edge in internal_edges[:30]:  # limit to avoid oversized prompts
            src_name = id_to_name.get(edge.src_id, edge.src_id)
            dst_name = id_to_name.get(edge.dst_id, edge.dst_id)
            line = f"- {src_name} --[{edge.relation}]--> {dst_name}"
            if edge.context:
                line += f'  [证据: "{edge.context[:80]}"]'
            relations_lines.append(line)
        relations_text = "\n".join(relations_lines) if relations_lines else "无关系"

        # Build prompt
        prompt = _COMMUNITY_SUMMARY_PROMPT_TEMPLATE.format(
            entities_text=entities_text,
            relations_text=relations_text,
        )

        # Call LLM
        summary = ""
        try:
            summary = await self._call_llm(prompt)
            summary = summary.strip()
        except Exception as exc:
            logger.warning(
                "community_llm_summary_failed",
                community_id=community_id,
                error=str(exc),
            )
            # Fallback: use entity names as summary
            names = [n.name for n in community_nodes[:10]]
            summary = f"包含实体：{', '.join(names)}"

        # Generate embedding for summary
        embedding: list[float] | None = None
        if self._embed_fn is not None and summary:
            try:
                embedding = await self._get_embedding(summary)
            except Exception as exc:
                logger.warning(
                    "community_embed_failed",
                    community_id=community_id,
                    error=str(exc),
                )

        return Community(
            id=community_id,
            kb_id=kb_id,
            node_ids=node_ids,
            summary=summary,
            level=level,
            embedding=embedding,
            created_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM engine, supporting both sync and async generate methods."""
        if asyncio.iscoroutinefunction(self._llm.generate):
            return await self._llm.generate(prompt)
        else:
            return await asyncio.to_thread(self._llm.generate, prompt)

    async def _get_embedding(self, text: str) -> list[float]:
        """Call embed_fn, supporting both sync and async."""
        if asyncio.iscoroutinefunction(self._embed_fn):
            return await self._embed_fn(text)
        else:
            return await asyncio.to_thread(self._embed_fn, text)
