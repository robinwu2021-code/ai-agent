"""rag/graph/retriever.py — 图谱检索策略"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable

import structlog

from rag.graph.models import Community, Edge, Node, SubGraph
from rag.graph.store import KGStore

logger = structlog.get_logger(__name__)

_ENTITY_EXTRACT_PROMPT = """\
从以下问题中提取实体名称列表。只返回JSON格式，不要解释。

问题：{query}

输出格式：
{{"entities": ["实体1", "实体2", ...]}}
"""

_TWO_ENTITY_EXTRACT_PROMPT = """\
从以下问题中提取两个最重要的实体名称，用于查找它们之间的关系路径。
只返回JSON格式，不要解释。

问题：{query}

输出格式：
{{"entity1": "第一个实体", "entity2": "第二个实体"}}
"""


def _parse_json_response(text: str) -> dict | None:
    """Extract and parse JSON from LLM response, handles markdown code blocks."""
    if not text:
        return None
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code blocks
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1) if pattern.startswith("```") else match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


class GraphRetriever:
    """
    知识图谱检索器，提供三种检索模式：
    - local_search:  局部搜索（实体 + BFS 邻居扩展）
    - global_search: 全局搜索（社区摘要向量匹配）
    - path_search:   路径搜索（两实体间最短路径）

    Args:
        store: KGStore 实例。
        embed_fn: 嵌入函数 (text: str) -> list[float]，可为 None。
        llm_engine: LLM 引擎实例（用于实体抽取），可为 None（此时跳过 LLM 步骤）。
    """

    def __init__(
        self,
        store: KGStore,
        embed_fn: Callable[[str], Any] | None = None,
        llm_engine: Any | None = None,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._llm = llm_engine

    # ------------------------------------------------------------------
    # Mode A: Local Search
    # ------------------------------------------------------------------

    async def local_search(
        self,
        query: str,
        kb_id: str,
        hops: int = 2,
        top_k: int = 5,
    ) -> SubGraph:
        """
        局部搜索：从查询中提取实体，扩展邻居节点，返回子图。

        Steps:
        1. Extract entity mentions from query using LLM
        2. For each entity: search_nodes_by_text + search_nodes_by_embedding
        3. BFS expand hops neighbors via get_neighbors()
        4. Collect nodes + edges, build reasoning chain
        5. Format context text

        Args:
            query: 用户查询文本。
            kb_id: 知识库 ID。
            hops: BFS 扩展层数（默认 2）。
            top_k: 每个实体取 top_k 匹配节点（默认 5）。

        Returns:
            SubGraph: 包含相关节点、边和推理链的子图。
        """
        # Step 1: Extract entity names from query
        entity_names = await self._extract_entities(query)
        if not entity_names:
            # Fallback: use the full query as a text search
            entity_names = [query]

        logger.debug(
            "local_search_entities",
            query=query[:80],
            entities=entity_names,
        )

        # Step 2: Find seed nodes for each entity
        seed_nodes: dict[str, Node] = {}
        for entity_name in entity_names[:top_k]:
            # Text search
            text_results = await self._store.search_nodes_by_text(
                entity_name, kb_id, limit=top_k
            )
            for n in text_results:
                seed_nodes[n.id] = n

            # Embedding search (if available)
            if self._embed_fn is not None:
                try:
                    emb = await self._get_embedding(entity_name)
                    emb_results = await self._store.search_nodes_by_embedding(
                        emb, kb_id, limit=top_k
                    )
                    for n in emb_results:
                        seed_nodes[n.id] = n
                except Exception as exc:
                    logger.warning(
                        "local_search_embed_failed",
                        entity=entity_name,
                        error=str(exc),
                    )

        if not seed_nodes:
            logger.info("local_search_no_seeds", query=query[:80], kb_id=kb_id)
            return SubGraph(context_text="未找到相关图谱内容。")

        # Step 3: BFS expand neighbors
        all_nodes: dict[str, Node] = dict(seed_nodes)
        all_edges: dict[str, Edge] = {}

        for seed_id in list(seed_nodes.keys()):
            neighbor_nodes, neighbor_edges = await self._store.get_neighbors(
                seed_id, kb_id, hops=hops
            )
            for n in neighbor_nodes:
                all_nodes[n.id] = n
            for e in neighbor_edges:
                all_edges[e.id] = e

        # Also get edges between seed nodes themselves
        kb_edges = await self._store.list_edges(kb_id)
        node_ids_set = set(all_nodes.keys())
        for e in kb_edges:
            if e.src_id in node_ids_set and e.dst_id in node_ids_set:
                all_edges[e.id] = e

        # Step 4: Build reasoning chain
        reasoning_chain = self._build_reasoning_chain(
            list(all_nodes.values()), list(all_edges.values()), list(seed_nodes.keys())
        )

        # Step 5: Format context
        subgraph = SubGraph(
            nodes=list(all_nodes.values()),
            edges=list(all_edges.values()),
            reasoning_chain=reasoning_chain,
        )
        subgraph.context_text = self.format_for_prompt(subgraph)

        logger.info(
            "local_search_done",
            query=query[:80],
            nodes=len(subgraph.nodes),
            edges=len(subgraph.edges),
        )
        return subgraph

    # ------------------------------------------------------------------
    # Mode B: Global Search
    # ------------------------------------------------------------------

    async def global_search(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5,
    ) -> SubGraph:
        """
        全局搜索：基于社区摘要的向量匹配检索。

        Steps:
        1. Embed query
        2. search_communities_by_embedding() for level >= 1 communities
        3. For each top community: gather node info + include summary in context
        4. Return SubGraph with community context

        Args:
            query: 用户查询文本。
            kb_id: 知识库 ID。
            top_k: 返回的顶部社区数（默认 5）。

        Returns:
            SubGraph: 基于社区摘要的子图。
        """
        if self._embed_fn is None:
            # Fallback: list communities directly
            communities = await self._store.list_communities(kb_id, level=None)
            communities = communities[:top_k]
        else:
            try:
                query_emb = await self._get_embedding(query)
                communities = await self._store.search_communities_by_embedding(
                    query_emb, kb_id, level=1, limit=top_k
                )
                # If no level-1 communities, search all levels
                if not communities:
                    communities = await self._store.search_communities_by_embedding(
                        query_emb, kb_id, level=None, limit=top_k
                    )
            except Exception as exc:
                logger.warning("global_search_embed_failed", error=str(exc))
                communities = await self._store.list_communities(kb_id, level=None)
                communities = communities[:top_k]

        if not communities:
            logger.info("global_search_no_communities", query=query[:80], kb_id=kb_id)
            return SubGraph(context_text="未找到相关社区摘要。")

        # Collect nodes from communities
        all_node_ids: set[str] = set()
        for comm in communities:
            all_node_ids.update(comm.node_ids)

        # Fetch node objects
        all_nodes: dict[str, Node] = {}
        for nid in list(all_node_ids)[:200]:  # limit to avoid huge fetches
            node = await self._store.get_node(nid)
            if node:
                all_nodes[nid] = node

        # Build context text with community summaries
        context_parts: list[str] = ["## 知识图谱社区摘要\n"]
        for i, comm in enumerate(communities, 1):
            context_parts.append(f"### 社区 {i} (包含 {len(comm.node_ids)} 个实体)")
            if comm.summary:
                context_parts.append(comm.summary)
            else:
                # No summary, list entity names
                names = [
                    all_nodes[nid].name
                    for nid in comm.node_ids
                    if nid in all_nodes
                ]
                context_parts.append(f"实体：{', '.join(names[:10])}")
            context_parts.append("")

        context_text = "\n".join(context_parts)

        subgraph = SubGraph(
            nodes=list(all_nodes.values()),
            edges=[],
            reasoning_chain=[],
            context_text=context_text,
        )

        logger.info(
            "global_search_done",
            query=query[:80],
            communities=len(communities),
            nodes=len(subgraph.nodes),
        )
        return subgraph

    # ------------------------------------------------------------------
    # Mode C: Path Search
    # ------------------------------------------------------------------

    async def path_search(
        self,
        query: str,
        kb_id: str,
        max_hops: int = 4,
    ) -> SubGraph:
        """
        路径搜索：找出查询中两个实体间的最短路径。

        Steps:
        1. Extract TWO entity names from query using LLM
        2. Find both nodes in store (text + embedding search)
        3. Call store.get_path(src_id, dst_id) for BFS path
        4. Return path as SubGraph with reasoning_chain

        Args:
            query: 用户查询文本。
            kb_id: 知识库 ID。
            max_hops: BFS 最大跳数（默认 4）。

        Returns:
            SubGraph: 两实体间的路径子图。
        """
        # Step 1: Extract two entity names
        entity1, entity2 = await self._extract_two_entities(query)

        logger.debug(
            "path_search_entities",
            query=query[:80],
            entity1=entity1,
            entity2=entity2,
        )

        if not entity1 or not entity2:
            return SubGraph(context_text="无法从查询中提取两个实体用于路径搜索。")

        # Step 2: Find nodes for both entities
        node1 = await self._find_best_node(entity1, kb_id)
        node2 = await self._find_best_node(entity2, kb_id)

        if node1 is None:
            return SubGraph(
                context_text=f"未找到实体 '{entity1}' 对应的节点。"
            )
        if node2 is None:
            return SubGraph(
                context_text=f"未找到实体 '{entity2}' 对应的节点。"
            )

        if node1.id == node2.id:
            return SubGraph(
                nodes=[node1],
                edges=[],
                reasoning_chain=[],
                context_text=f"'{entity1}' 和 '{entity2}' 指向同一实体：{node1.name}",
            )

        # Step 3: BFS path
        path_edges = await self._store.get_path(
            node1.id, node2.id, kb_id, max_hops=max_hops
        )

        if not path_edges:
            return SubGraph(
                nodes=[node1, node2],
                edges=[],
                reasoning_chain=[],
                context_text=(
                    f"在 {max_hops} 跳范围内未找到 '{node1.name}' 到 '{node2.name}' 的路径。"
                ),
            )

        # Collect all nodes on path
        path_node_ids: set[str] = {node1.id, node2.id}
        for e in path_edges:
            path_node_ids.add(e.src_id)
            path_node_ids.add(e.dst_id)

        path_nodes: dict[str, Node] = {}
        for nid in path_node_ids:
            n = await self._store.get_node(nid)
            if n:
                path_nodes[nid] = n

        # Build reasoning chain
        reasoning_chain: list[dict] = []
        id_to_name = {nid: n.name for nid, n in path_nodes.items()}
        for edge in path_edges:
            reasoning_chain.append(
                {
                    "src": id_to_name.get(edge.src_id, edge.src_id),
                    "relation": edge.relation,
                    "dst": id_to_name.get(edge.dst_id, edge.dst_id),
                    "evidence": edge.context,
                }
            )

        # Build path string for context
        path_str = self._format_path_string(path_edges, id_to_name)

        subgraph = SubGraph(
            nodes=list(path_nodes.values()),
            edges=path_edges,
            reasoning_chain=reasoning_chain,
        )

        # Build context text
        context_parts = [f"## 关系路径：{node1.name} → {node2.name}\n"]
        context_parts.append(f"### 路径\n{path_str}\n")
        context_parts.append("### 路径详情")
        for step in reasoning_chain:
            line = f"- {step['src']} --[{step['relation']}]--> {step['dst']}"
            if step.get("evidence"):
                line += f'  [证据: "{step["evidence"][:80]}"]'
            context_parts.append(line)

        subgraph.context_text = "\n".join(context_parts)

        logger.info(
            "path_search_done",
            entity1=node1.name,
            entity2=node2.name,
            path_length=len(path_edges),
        )
        return subgraph

    # ------------------------------------------------------------------
    # Format for LLM prompt
    # ------------------------------------------------------------------

    def format_for_prompt(self, subgraph: SubGraph) -> str:
        """
        将子图格式化为 LLM 上下文文本。

        格式示例：
        ## 知识图谱相关内容

        ### 相关实体
        - 小米 (ORG): 中国消费电子公司...

        ### 关键关系
        - 小米 --[竞争对手]--> 华为  [证据: "..."]

        ### 推理路径
        小米 → 竞争对手 → 华为 → ...

        Args:
            subgraph: SubGraph 实例。

        Returns:
            str: 格式化后的上下文文本。
        """
        if not subgraph.nodes and not subgraph.edges:
            return "## 知识图谱相关内容\n\n未找到相关图谱信息。"

        parts: list[str] = ["## 知识图谱相关内容\n"]

        # Nodes section
        if subgraph.nodes:
            parts.append("### 相关实体")
            id_to_node: dict[str, Node] = {n.id: n for n in subgraph.nodes}
            for node in subgraph.nodes[:50]:  # limit for prompt size
                type_label = node.node_type.value
                desc = node.description.strip() if node.description else ""
                aliases_str = ""
                if node.aliases:
                    aliases_str = f" (又名: {', '.join(node.aliases[:3])})"
                line = f"- {node.name} ({type_label}){aliases_str}"
                if desc:
                    line += f": {desc[:100]}"
                parts.append(line)
            parts.append("")

        # Edges section
        if subgraph.edges:
            parts.append("### 关键关系")
            id_to_node = {n.id: n for n in subgraph.nodes}
            for edge in subgraph.edges[:50]:  # limit for prompt size
                src_name = id_to_node.get(edge.src_id, Node(
                    id=edge.src_id, kb_id="", name=edge.src_id
                )).name
                dst_name = id_to_node.get(edge.dst_id, Node(
                    id=edge.dst_id, kb_id="", name=edge.dst_id
                )).name
                # 优先显示人类可读标签，如 "REQUIRES: 必须通过"
                if edge.relation_label:
                    rel_display = f"{edge.relation}: {edge.relation_label}"
                else:
                    rel_display = edge.relation
                line = f"- {src_name} --[{rel_display}]--> {dst_name}"
                evidence_text = edge.evidence
                if evidence_text:
                    line += f'  [证据: "{evidence_text[:80]}"]'
                parts.append(line)
            parts.append("")

        # Reasoning chain section
        if subgraph.reasoning_chain:
            parts.append("### 推理路径")
            # Build path string from first chain
            path_segments: list[str] = []
            for step in subgraph.reasoning_chain[:10]:
                src = step.get("src", "?")
                rel = step.get("relation", "?")
                dst = step.get("dst", "?")
                if not path_segments:
                    path_segments.append(src)
                path_segments.append(f"→ {rel} → {dst}")
            if path_segments:
                parts.append(" ".join(path_segments))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _extract_entities(self, query: str) -> list[str]:
        """Extract entity names from query using LLM. Returns list of entity names."""
        if self._llm is None:
            return []
        prompt = _ENTITY_EXTRACT_PROMPT.format(query=query)
        try:
            raw = await self._call_llm(prompt)
            parsed = _parse_json_response(raw)
            if parsed and isinstance(parsed.get("entities"), list):
                return [str(e).strip() for e in parsed["entities"] if e]
        except Exception as exc:
            logger.warning("extract_entities_failed", query=query[:80], error=str(exc))
        return []

    async def _extract_two_entities(self, query: str) -> tuple[str, str]:
        """Extract exactly two entity names from query for path search."""
        if self._llm is None:
            return "", ""
        prompt = _TWO_ENTITY_EXTRACT_PROMPT.format(query=query)
        try:
            raw = await self._call_llm(prompt)
            parsed = _parse_json_response(raw)
            if parsed:
                e1 = str(parsed.get("entity1", "")).strip()
                e2 = str(parsed.get("entity2", "")).strip()
                return e1, e2
        except Exception as exc:
            logger.warning(
                "extract_two_entities_failed", query=query[:80], error=str(exc)
            )
        return "", ""

    async def _find_best_node(self, entity_name: str, kb_id: str) -> Node | None:
        """Find the best matching node for an entity name."""
        # Exact/alias match
        node = await self._store.get_node_by_name(entity_name, kb_id)
        if node:
            return node

        # Text search
        results = await self._store.search_nodes_by_text(entity_name, kb_id, limit=3)
        if results:
            return results[0]

        # Embedding search
        if self._embed_fn is not None:
            try:
                emb = await self._get_embedding(entity_name)
                emb_results = await self._store.search_nodes_by_embedding(
                    emb, kb_id, limit=3
                )
                if emb_results:
                    return emb_results[0]
            except Exception as exc:
                logger.warning(
                    "path_search_embed_failed", entity=entity_name, error=str(exc)
                )

        return None

    def _build_reasoning_chain(
        self,
        nodes: list[Node],
        edges: list[Edge],
        seed_node_ids: list[str],
    ) -> list[dict]:
        """Build a reasoning chain from edges, starting from seed nodes."""
        id_to_node: dict[str, Node] = {n.id: n for n in nodes}
        chain: list[dict] = []

        # Prioritize edges that originate from seed nodes
        seed_set = set(seed_node_ids)
        prioritized_edges = sorted(
            edges,
            key=lambda e: (e.src_id not in seed_set and e.dst_id not in seed_set),
        )

        for edge in prioritized_edges[:20]:  # limit chain length
            src_name = id_to_node.get(edge.src_id, Node(
                id=edge.src_id, kb_id="", name=edge.src_id
            )).name
            dst_name = id_to_node.get(edge.dst_id, Node(
                id=edge.dst_id, kb_id="", name=edge.dst_id
            )).name
            chain.append(
                {
                    "src": src_name,
                    "relation": edge.relation,
                    "relation_label": edge.relation_label,
                    "dst": dst_name,
                    "evidence": edge.evidence,
                }
            )
        return chain

    def _format_path_string(
        self, edges: list[Edge], id_to_name: dict[str, str]
    ) -> str:
        """Format a path as: A → rel → B → rel → C ..."""
        if not edges:
            return ""
        parts: list[str] = []
        for i, edge in enumerate(edges):
            src_name = id_to_name.get(edge.src_id, edge.src_id)
            dst_name = id_to_name.get(edge.dst_id, edge.dst_id)
            if i == 0:
                parts.append(src_name)
            parts.append(f"→ {edge.relation} →")
            parts.append(dst_name)
        return " ".join(parts)

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
