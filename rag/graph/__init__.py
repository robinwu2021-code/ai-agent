"""
rag/graph — 知识图谱构建与检索系统

模块说明：
  models.py     数据模型（Node / Edge / Community / Triple）
  store.py      存储层（SQLiteKGStore / Neo4jKGStore）
  extractor.py  三元组抽取（LLM 驱动）
  resolver.py   实体消解（向量相似度去重合并）
  community.py  社区检测（Label Propagation）+ LLM 摘要
  retriever.py  检索策略（局部/全局/路径）
  builder.py    构建管线（串联上述所有组件）
"""
from rag.graph.models import Node, Edge, Community, Triple, NodeType, SubGraph
from rag.graph.store import KGStore, SQLiteKGStore
from rag.graph.extractor import TripleExtractor
from rag.graph.resolver import EntityResolver
from rag.graph.community import CommunityDetector, CommunitySummarizer
from rag.graph.retriever import GraphRetriever
from rag.graph.builder import GraphBuilder

__all__ = [
    "Node", "Edge", "Community", "Triple", "NodeType", "SubGraph",
    "KGStore", "SQLiteKGStore",
    "TripleExtractor",
    "EntityResolver",
    "CommunityDetector", "CommunitySummarizer",
    "GraphRetriever",
    "GraphBuilder",
]
