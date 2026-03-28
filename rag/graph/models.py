"""rag/graph/models.py — 知识图谱数据模型"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    # ── 原有类型 ────────────────────────────────────────────
    PERSON           = "PERSON"       # 人物
    ORG              = "ORG"          # 组织/公司/部门
    PRODUCT          = "PRODUCT"      # 产品/系统/平台
    CONCEPT          = "CONCEPT"      # 概念/技术/方法论
    EVENT            = "EVENT"        # 事件/活动
    LOCATION         = "LOCATION"     # 地点/区域
    OTHER            = "OTHER"        # 其他
    # ── 新增业务语义类型（Phase 2）──────────────────────────
    PROCESS          = "PROCESS"      # 业务流程/操作步骤/方法
    RULE             = "RULE"         # 业务规则/判断条件/策略
    METRIC           = "METRIC"       # 量化指标/阈值/参数（如"成功率≥99.9%"）
    DOCUMENT_SECTION = "DOCUMENT_SECTION"  # 文档章节/条款（如"第3章"、"4.2.1条"）
    CONSTRAINT       = "CONSTRAINT"   # 约束条件/限制/前提
    STANDARD         = "STANDARD"     # 标准/规范/规程编号（如"ISO 27001"）


# 结构化关系类型本体（对应提示词中的 relation_type 字段）
class RelationType(str, Enum):
    # 层级关系
    IS_A             = "IS_A"         # 是一种/属于
    PART_OF          = "PART_OF"      # 是…的组成部分
    HAS_PART         = "HAS_PART"     # 包含/由…组成
    # 因果关系
    CAUSES           = "CAUSES"       # 导致/引起
    PREVENTS         = "PREVENTS"     # 防止/避免
    DEPENDS_ON       = "DEPENDS_ON"   # 依赖于/需要前置
    # 定义关系
    DEFINED_AS       = "DEFINED_AS"   # 定义为/是指
    SIMILAR_TO       = "SIMILAR_TO"   # 类似于/等同于
    CONTRADICTS      = "CONTRADICTS"  # 与…冲突/矛盾
    # 流程关系
    PRECEDES         = "PRECEDES"     # 前置步骤/先于
    FOLLOWS          = "FOLLOWS"      # 后置步骤/后于
    TRIGGERS         = "TRIGGERS"     # 触发/激活
    # 业务关系
    REQUIRES         = "REQUIRES"     # 要求/规定必须
    GOVERNS          = "GOVERNS"      # 适用于/规定/管辖
    IMPLEMENTS       = "IMPLEMENTS"   # 实现/落实/执行
    # 约束关系
    CONSTRAINED_BY   = "CONSTRAINED_BY"  # 受约束于
    THRESHOLD        = "THRESHOLD"    # 阈值为/参数值为
    INVOLVES         = "INVOLVES"     # 涉及/参与


_VALID_RELATION_TYPES = {r.value for r in RelationType}


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
    relation:    str              # 结构化关系类型（RelationType 值，如 REQUIRES）
    weight:      float = 1.0     # 置信度 0~1
    doc_id:      str = ""
    chunk_id:    str = ""
    context:     str = ""        # 格式："{relation_label} || {原文证据片段}"
    created_at:  float = 0.0

    @property
    def relation_label(self) -> str:
        """解析 context 中的人类可读关系描述（relation_label 部分）。"""
        if " || " in self.context:
            return self.context.split(" || ", 1)[0]
        return ""

    @property
    def evidence(self) -> str:
        """解析 context 中的原文证据部分。"""
        if " || " in self.context:
            return self.context.split(" || ", 1)[1]
        return self.context


@dataclass
class Triple:
    """LLM 抽取的原始三元组，尚未写入存储。"""
    src_name:       str
    src_type:       NodeType
    src_desc:       str
    relation:       str          # 向后兼容：旧版自由文本关系描述
    dst_name:       str
    dst_type:       NodeType
    dst_desc:       str
    confidence:     float = 1.0
    evidence:       str = ""          # 原文证据片段
    relation_type:  str = ""          # 结构化关系类型（Phase 2 新增）
    relation_label: str = ""          # 人类可读关系描述（Phase 2 新增）


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
    reasoning_chain: list[dict] = field(default_factory=list)  # [{src, relation, relation_label, dst, evidence}]
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
                 "relation": e.relation, "relation_label": e.relation_label,
                 "weight": e.weight, "evidence": e.evidence}
                for e in self.edges
            ],
            "reasoning_chain": self.reasoning_chain,
        }
