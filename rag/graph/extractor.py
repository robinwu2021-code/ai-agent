"""rag/graph/extractor.py — LLM 驱动三元组抽取"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable

import structlog

from rag.graph.models import NodeType, Triple, RelationType

_VALID_RELATION_TYPES = {r.value for r in RelationType}

logger = structlog.get_logger(__name__)

_EXTRACT_PROMPT_TEMPLATE = """\
你是知识图谱专家。从以下文本中抽取实体和关系三元组，重点捕捉业务逻辑、规则约束和流程关联。

【实体类型（必须从以下选择）】
PERSON=人物, ORG=组织/公司/部门, PRODUCT=产品/系统/平台, CONCEPT=概念/技术/方法论,
EVENT=事件/活动, LOCATION=地点/区域, PROCESS=业务流程/操作步骤,
RULE=业务规则/判断条件/策略, METRIC=量化指标/阈值/参数(如"成功率≥99.9%"),
DOCUMENT_SECTION=文档章节/条款(如"第3章"/"4.2.1"), CONSTRAINT=约束条件/限制,
STANDARD=标准/规范编号(如"ISO 27001"), OTHER=其他

【关系类型（relation_type 必须从以下选择）】
层级: IS_A(是一种/属于), PART_OF(是…的组成部分), HAS_PART(包含)
因果: CAUSES(导致), PREVENTS(防止), DEPENDS_ON(依赖于)
定义: DEFINED_AS(定义为/是指), SIMILAR_TO(类似于), CONTRADICTS(与…冲突)
流程: PRECEDES(前置步骤), FOLLOWS(后续步骤), TRIGGERS(触发)
业务: REQUIRES(要求/必须), GOVERNS(适用于/规定), IMPLEMENTS(实现/落实)
约束: CONSTRAINED_BY(受约束于), THRESHOLD(阈值为), INVOLVES(涉及)

要求：
1. 优先提取有明确业务含义的关系，不抽取模糊关联
2. relation_type 从上方选择，relation_label 写原文中的具体描述词（2-6字）
3. confidence（0-1），evidence 必须是原文中的具体句子（不要改写）
4. 重点关注：流程步骤顺序、规则触发条件、指标阈值、章节从属关系
5. 章节编号（如"4.2.1"）作为 DOCUMENT_SECTION 类型实体

文本：
{text}

严格按以下 JSON 格式输出：
{{
  "entities": [
    {{"name": "实体名", "type": "PROCESS", "description": "一句话描述"}}
  ],
  "triples": [
    {{
      "src": "实体A",
      "relation_type": "REQUIRES",
      "relation_label": "必须通过",
      "dst": "实体B",
      "confidence": 0.9,
      "evidence": "原文中的具体句子"
    }}
  ]
}}
"""

_VALID_NODE_TYPES = {t.value for t in NodeType}


def _normalize_name(name: str) -> str:
    """Strip whitespace, normalize internal whitespace."""
    if not name:
        return name
    # Strip surrounding whitespace
    name = name.strip()
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    return name


def _parse_node_type(type_str: str | None) -> NodeType:
    """Parse a type string into NodeType, defaulting to OTHER."""
    if not type_str:
        return NodeType.OTHER
    upper = type_str.strip().upper()
    if upper in _VALID_NODE_TYPES:
        return NodeType(upper)
    return NodeType.OTHER


def _extract_json_from_text(text: str) -> dict | None:
    """
    Attempt to extract a JSON object from LLM response text.
    Handles cases where LLM wraps JSON in markdown code blocks.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code block
    # ```json ... ``` or ``` ... ```
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


class TripleExtractor:
    """
    LLM 驱动三元组抽取器。

    Args:
        llm_engine: 具备 generate(prompt) -> str 接口的 LLM 引擎。
        min_confidence: 过滤低置信度三元组的阈值（默认 0.3）。
    """

    def __init__(self, llm_engine: Any, min_confidence: float = 0.5) -> None:
        self._llm = llm_engine
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(
        self, chunk_text: str, doc_id: str = "", chunk_id: str = ""
    ) -> list[Triple]:
        """
        从单个文本块中抽取三元组。

        Args:
            chunk_text: 输入文本块。
            doc_id: 文档 ID（附加到三元组证据中用于溯源）。
            chunk_id: 块 ID。

        Returns:
            list[Triple]: 抽取出的三元组列表。
        """
        if not chunk_text or not chunk_text.strip():
            return []

        prompt = _EXTRACT_PROMPT_TEMPLATE.format(text=chunk_text.strip())

        try:
            raw_response = await self._call_llm(prompt)
        except Exception as exc:
            logger.warning(
                "llm_call_failed",
                doc_id=doc_id,
                chunk_id=chunk_id,
                error=str(exc),
            )
            return []

        triples = self._parse_response(raw_response, doc_id=doc_id, chunk_id=chunk_id)

        logger.debug(
            "triples_extracted",
            doc_id=doc_id,
            chunk_id=chunk_id,
            count=len(triples),
        )
        return triples

    async def extract_batch(
        self,
        chunks: list[dict],
        concurrency: int = 4,
    ) -> list[Triple]:
        """
        批量从多个文本块中抽取三元组。

        Args:
            chunks: 每个元素为包含 text/doc_id/chunk_id 键的字典。
                    示例: [{"text": "...", "doc_id": "d1", "chunk_id": "c1"}, ...]
            concurrency: 并发数量。

        Returns:
            list[Triple]: 所有块抽取的三元组合并列表。
        """
        semaphore = asyncio.Semaphore(concurrency)
        all_triples: list[Triple] = []

        async def _extract_one(chunk: dict) -> list[Triple]:
            text = chunk.get("text", chunk.get("content", ""))
            doc_id = chunk.get("doc_id", "")
            chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
            async with semaphore:
                return await self.extract(text, doc_id=doc_id, chunk_id=chunk_id)

        tasks = [_extract_one(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "batch_extract_chunk_failed",
                    chunk_index=i,
                    error=str(result),
                )
            else:
                all_triples.extend(result)

        return all_triples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM engine.

        Supports two interface styles:
        1. LLMRouter / LLMEngine (project standard): chat(messages, tools, config)
        2. Legacy: generate(prompt) — sync or async
        """
        # ── LLMRouter / LLMEngine: chat() interface ──────────────────────
        if hasattr(self._llm, "chat") and not hasattr(self._llm, "generate"):
            from core.models import AgentConfig, Message, MessageRole
            messages = [Message(role=MessageRole.USER, content=prompt)]
            config   = AgentConfig()
            resp     = await self._llm.chat(messages, [], config)
            return resp.content or ""

        # ── Legacy: generate() method ─────────────────────────────────────
        if asyncio.iscoroutinefunction(self._llm.generate):
            return await self._llm.generate(prompt)
        return await asyncio.to_thread(self._llm.generate, prompt)

    def _parse_response(
        self, raw_response: str, doc_id: str = "", chunk_id: str = ""
    ) -> list[Triple]:
        """
        Parse LLM response text into Triple objects.
        Falls back gracefully on parse errors.
        """
        if not raw_response or not raw_response.strip():
            return []

        parsed = _extract_json_from_text(raw_response)
        if parsed is None:
            logger.warning(
                "failed_to_parse_llm_response",
                doc_id=doc_id,
                chunk_id=chunk_id,
                response_preview=raw_response[:200],
            )
            return []

        # Build entity lookup: name -> (type, description)
        entity_map: dict[str, tuple[NodeType, str]] = {}
        entities_raw = parsed.get("entities", [])
        if isinstance(entities_raw, list):
            for ent in entities_raw:
                if not isinstance(ent, dict):
                    continue
                name = _normalize_name(str(ent.get("name", "")))
                if not name:
                    continue
                ent_type = _parse_node_type(ent.get("type"))
                desc = str(ent.get("description", "")).strip()
                entity_map[name] = (ent_type, desc)

        triples_raw = parsed.get("triples", [])
        if not isinstance(triples_raw, list):
            return []

        results: list[Triple] = []
        for item in triples_raw:
            if not isinstance(item, dict):
                continue

            # Parse confidence
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0

            # Filter low-confidence
            if confidence < self._min_confidence:
                continue

            src_name = _normalize_name(str(item.get("src", "")))
            dst_name = _normalize_name(str(item.get("dst", "")))
            relation = str(item.get("relation", "")).strip()
            evidence = str(item.get("evidence", "")).strip()

            relation_type  = str(item.get("relation_type", "")).strip().upper()
            relation_label = str(item.get("relation_label", "")).strip()
            # Validate relation_type against known types; fall back gracefully
            if relation_type not in _VALID_RELATION_TYPES:
                relation_type = ""

            # For new-format prompts, relation may be empty; use relation_label as fallback
            effective_relation = relation or relation_label
            if not src_name or not dst_name or not effective_relation:
                continue

            # Lookup entity info
            src_type, src_desc = entity_map.get(src_name, (NodeType.OTHER, ""))
            dst_type, dst_desc = entity_map.get(dst_name, (NodeType.OTHER, ""))

            triple = Triple(
                src_name=src_name,
                src_type=src_type,
                src_desc=src_desc,
                relation=relation_label or relation,   # human-readable label, fallback to raw relation text
                dst_name=dst_name,
                dst_type=dst_type,
                dst_desc=dst_desc,
                confidence=confidence,
                evidence=evidence,
                relation_type=relation_type,
                relation_label=relation_label,
            )
            results.append(triple)

        return results
