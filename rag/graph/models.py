"""rag/graph/models.py — 知识图谱数据模型"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    PERSON   = "PERSON"
    ORG      = "ORG"
    PRODUCT  = "PRODUCT"
    CONCEPT  = "CONCEPT"
    EVENT    = "EVENT"
    LOCATION = "LOCATION"
    OTHER    = "OTHER"


@dataclass
class Node:
    id:          str
    kb_id:       str
    name:        str
    node_type:   NodeType = NodeType.OTHER
    aliases:     list[str] = field(default_factory=list)
    description: str = ""
    doc_ids:     list[str] = field(default_factory=list)
    embedding:   list[float] | None = None
    degree:      int = 0          # 计算属性，查询时填充
    created_at:  float = 0.0


@dataclass
class Edge:
    id:          str
    kb_id:       str
    src_id:      str
    dst_id:      str
    relation:    str
    weight:      float = 1.0     # 置信度 0~1
    doc_id:      str = ""
    chunk_id:    str = ""
    context:     str = ""        # 原文片段（边的证据）
    created_at:  float = 0.0


@dataclass
class Triple:
    """LLM 抽取的原始三元组，尚未写入存储。"""
    src_name:    str
    src_type:    NodeType
    src_desc:    str
    relation:    str
    dst_name:    str
    dst_type:    NodeType
    dst_desc:    str
    confidence:  float = 1.0
    evidence:    str = ""        # 原文片段


@dataclass
class Community:
    id:          str
    kb_id:       str
    node_ids:    list[str] = field(default_factory=list)
    summary:     str = ""
    level:       int = 0         # 0=叶, 1=中间, 2=全局
    embedding:   list[float] | None = None
    created_at:  float = 0.0


@dataclass
class SubGraph:
    """检索结果子图，含节点、边和推理链。"""
    nodes:           list[Node] = field(default_factory=list)
    edges:           list[Edge] = field(default_factory=list)
    reasoning_chain: list[dict] = field(default_factory=list)  # [{src, relation, dst, evidence}]
    context_text:    str = ""      # 格式化为 Prompt 的文本

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n.id, "name": n.name, "type": n.node_type.value,
                 "description": n.description, "degree": n.degree}
                for n in self.nodes
            ],
            "edges": [
                {"id": e.id, "src_id": e.src_id, "dst_id": e.dst_id,
                 "relation": e.relation, "weight": e.weight, "context": e.context}
                for e in self.edges
            ],
            "reasoning_chain": self.reasoning_chain,
        }
