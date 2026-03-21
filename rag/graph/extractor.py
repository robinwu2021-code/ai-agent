"""rag/graph/extractor.py — LLM 驱动三元组抽取"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable

import structlog

from rag.graph.models import NodeType, Triple

logger = structlog.get_logger(__name__)

_EXTRACT_PROMPT_TEMPLATE = """\
你是知识图谱专家。从以下文本中抽取实体和关系三元组。

实体类型：PERSON(人物)/ORG(组织/公司)/PRODUCT(产品)/CONCEPT(概念/技术)/EVENT(事件)/LOCATION(地点)/OTHER(其他)

要求：
1. 只抽取文本中明确提到的关系，不要推断
2. 关系用简洁的中文动词短语（如"创立于"/"竞争对手"/"属于"/"发布"/"位于"）
3. 每个三元组给出置信度(0-1)和原文证据片段
4. 同一实体在不同地方可能有不同写法，尽量统一

文本：
{text}

以JSON格式输出（严格遵守格式）：
{{
  "entities": [
    {{"name": "实体名", "type": "ORG", "description": "一句话描述"}}
  ],
  "triples": [
    {{"src": "实体A", "relation": "关系", "dst": "实体B", "confidence": 0.9, "evidence": "原文片段"}}
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

    def __init__(self, llm_engine: Any, min_confidence: float = 0.3) -> None:
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
        """Call LLM engine, supporting both sync and async generate methods."""
        if asyncio.iscoroutinefunction(self._llm.generate):
            return await self._llm.generate(prompt)
        else:
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

            if not src_name or not dst_name or not relation:
                continue

            # Lookup entity info
            src_type, src_desc = entity_map.get(src_name, (NodeType.OTHER, ""))
            dst_type, dst_desc = entity_map.get(dst_name, (NodeType.OTHER, ""))

            triple = Triple(
                src_name=src_name,
                src_type=src_type,
                src_desc=src_desc,
                relation=relation,
                dst_name=dst_name,
                dst_type=dst_type,
                dst_desc=dst_desc,
                confidence=confidence,
                evidence=evidence,
            )
            results.append(triple)

        return results
