"""
rag/config.py — 知识库可插拔配置系统

支持从 YAML 文件或 dict 加载，所有组件均可通过 backend 字段切换实现。
默认配置文件：项目根目录 kb_config.yaml
可通过环境变量 KB_CONFIG_FILE 覆盖路径。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── 解析器配置 ────────────────────────────────────────────────────────────────

@dataclass
class ParserDoclingConfig:
    do_ocr: bool = True
    do_table_structure: bool = True
    table_mode: str = "accurate"          # accurate | fast
    ocr_lang: list[str] = field(default_factory=lambda: ["zh", "en"])

@dataclass
class ParserUnstructuredConfig:
    strategy: str = "hi_res"             # hi_res | fast | auto
    api_url: str = ""
    api_key: str = ""

@dataclass
class ParserMarkItDownConfig:
    enable_plugins: bool = True

@dataclass
class ParserConfig:
    backend: str = "docling"             # docling | unstructured | markitdown
    docling: ParserDoclingConfig = field(default_factory=ParserDoclingConfig)
    unstructured: ParserUnstructuredConfig = field(default_factory=ParserUnstructuredConfig)
    markitdown: ParserMarkItDownConfig = field(default_factory=ParserMarkItDownConfig)


# ── 分块器配置 ────────────────────────────────────────────────────────────────

@dataclass
class ChunkerStructuralConfig:
    min_size: int = 100
    max_size: int = 1500
    levels: list[int] = field(default_factory=lambda: [1, 2, 3])
    include_heading: bool = True

@dataclass
class ChunkerSemanticConfig:
    backend: str = "chonkie"
    threshold: float = 0.5
    target_size: int = 512
    max_size: int = 800

@dataclass
class ChunkerSentenceConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64

@dataclass
class ChunkerFixedConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64

@dataclass
class ContextualEnhancementConfig:
    enabled: bool = False
    use_prompt_cache: bool = True
    batch_size: int = 10
    prompt: str = "请用1-2句话描述以下片段的主题和关键信息，不要复述原文：\n{chunk_text}"

@dataclass
class ChunkerConfig:
    strategy: str = "auto"              # auto | structural | semantic | sentence | fixed
    structural: ChunkerStructuralConfig = field(default_factory=ChunkerStructuralConfig)
    semantic: ChunkerSemanticConfig = field(default_factory=ChunkerSemanticConfig)
    sentence: ChunkerSentenceConfig = field(default_factory=ChunkerSentenceConfig)
    fixed: ChunkerFixedConfig = field(default_factory=ChunkerFixedConfig)
    contextual_enhancement: ContextualEnhancementConfig = field(
        default_factory=ContextualEnhancementConfig
    )


# ── Embedder 配置 ─────────────────────────────────────────────────────────────

@dataclass
class EmbedderQwenConfig:
    model: str = "text-embedding-v3"
    dimensions: int = 1024
    batch_size: int = 32
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

@dataclass
class EmbedderOpenAIConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    api_key: str = ""
    base_url: str = ""

@dataclass
class EmbedderBGELocalConfig:
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 16

@dataclass
class EmbedderConfig:
    backend: str = "qwen"               # qwen | openai | bge_local
    qwen: EmbedderQwenConfig = field(default_factory=EmbedderQwenConfig)
    openai: EmbedderOpenAIConfig = field(default_factory=EmbedderOpenAIConfig)
    bge_local: EmbedderBGELocalConfig = field(default_factory=EmbedderBGELocalConfig)


# ── 向量存储配置 ──────────────────────────────────────────────────────────────

@dataclass
class VectorStoreQdrantConfig:
    mode: str = "embedded"              # embedded | server
    path: str = "./data/qdrant"
    url: str = "http://localhost:6333"
    collection: str = "kb_chunks"

@dataclass
class VectorStoreMilvusConfig:
    mode: str = "lite"                  # lite | standalone
    uri: str = "./data/milvus.db"
    host: str = "localhost"
    port: int = 19530

@dataclass
class VectorStoreChromaConfig:
    mode: str = "persistent"
    path: str = "./data/chroma"

@dataclass
class VectorStoreConfig:
    backend: str = "qdrant"             # qdrant | milvus | chroma
    qdrant: VectorStoreQdrantConfig = field(default_factory=VectorStoreQdrantConfig)
    milvus: VectorStoreMilvusConfig = field(default_factory=VectorStoreMilvusConfig)
    chroma: VectorStoreChromaConfig = field(default_factory=VectorStoreChromaConfig)


# ── 关键词检索配置 ────────────────────────────────────────────────────────────

@dataclass
class KeywordStoreESConfig:
    url: str = "http://localhost:9200"
    index_prefix: str = "kb_"
    username: str = ""
    password: str = ""
    analyzer: str = "ik_max_word"       # ik_max_word（中文）| standard

@dataclass
class KeywordStoreOpenSearchConfig:
    url: str = "http://localhost:9200"
    index_prefix: str = "kb_"
    username: str = ""
    password: str = ""

@dataclass
class KeywordStoreMemoryConfig:
    persist: bool = True
    path: str = "./data/bm25.pkl"

@dataclass
class KeywordStoreConfig:
    backend: str = "memory"             # elasticsearch | opensearch | memory
    elasticsearch: KeywordStoreESConfig = field(default_factory=KeywordStoreESConfig)
    opensearch: KeywordStoreOpenSearchConfig = field(default_factory=KeywordStoreOpenSearchConfig)
    memory: KeywordStoreMemoryConfig = field(default_factory=KeywordStoreMemoryConfig)


# ── 图谱配置 ──────────────────────────────────────────────────────────────────

@dataclass
class GraphStoreNeo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

@dataclass
class GraphStoreSQLiteConfig:
    path: str = "./data/graph.db"

@dataclass
class GraphStoreConfig:
    backend: str = "sqlite"             # neo4j | sqlite | memory
    neo4j: GraphStoreNeo4jConfig = field(default_factory=GraphStoreNeo4jConfig)
    sqlite: GraphStoreSQLiteConfig = field(default_factory=GraphStoreSQLiteConfig)

@dataclass
class GraphBuilderConfig:
    extract_entities: bool = True
    extract_relations: bool = True
    resolve_entities: bool = True
    build_communities: bool = True

@dataclass
class GraphConfig:
    enabled: bool = True
    per_document: bool = True
    workspace_level: bool = True
    store: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    builder: GraphBuilderConfig = field(default_factory=GraphBuilderConfig)


# ── 重排序配置 ────────────────────────────────────────────────────────────────

@dataclass
class RerankerBGEConfig:
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"

@dataclass
class RerankerCohereConfig:
    model: str = "rerank-multilingual-v3.0"
    api_key: str = ""

@dataclass
class RerankerConfig:
    enabled: bool = True
    backend: str = "llm"                # bge | cohere | llm | none
    top_n: int = 5
    bge: RerankerBGEConfig = field(default_factory=RerankerBGEConfig)
    cohere: RerankerCohereConfig = field(default_factory=RerankerCohereConfig)


# ── 检索融合配置 ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalWeightsConfig:
    vector: float = 0.5
    keyword: float = 0.3
    graph: float = 0.2

@dataclass
class RetrievalConfig:
    vector_top_k: int = 20
    keyword_top_k: int = 20
    graph_top_k: int = 10
    fusion: str = "rrf"                 # rrf | weighted
    rrf_k: int = 60
    weights: RetrievalWeightsConfig = field(default_factory=RetrievalWeightsConfig)
    enable_graph_retrieval: bool = True


# ── 权限配置 ──────────────────────────────────────────────────────────────────

@dataclass
class PermissionsDBConfig:
    backend: str = "sqlite"
    url: str = "./data/workspace.db"

@dataclass
class PermissionsConfig:
    enabled: bool = True
    db: PermissionsDBConfig = field(default_factory=PermissionsDBConfig)


# ── 文件管理配置 ──────────────────────────────────────────────────────────────

@dataclass
class FileStorageConfig:
    base_path: str = "./data/files"
    keep_versions: int = 10             # 保留最近 N 个版本的原始文件；0=不保留

@dataclass
class FileSummaryConfig:
    enabled: bool = True
    generate_diff: bool = True          # content_updated 时生成差异摘要
    max_input_chars: int = 3000

@dataclass
class FileAuditConfig:
    retention_days: int = 365           # 0 = 永久保留

@dataclass
class FileManagementConfig:
    db_url: str = "./data/kb_files.db"
    storage: FileStorageConfig = field(default_factory=FileStorageConfig)
    summary: FileSummaryConfig = field(default_factory=FileSummaryConfig)
    audit: FileAuditConfig = field(default_factory=FileAuditConfig)


# ── 顶层配置 ──────────────────────────────────────────────────────────────────

@dataclass
class KBConfig:
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    keyword_store: KeywordStoreConfig = field(default_factory=KeywordStoreConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    file_management: FileManagementConfig = field(default_factory=FileManagementConfig)


# ── YAML 加载 ─────────────────────────────────────────────────────────────────

def _apply_dict(cfg: Any, data: dict) -> None:
    """将 dict 数据递归应用到 dataclass 实例（跳过未知字段）。"""
    for k, v in data.items():
        if not hasattr(cfg, k):
            continue
        attr = getattr(cfg, k)
        if hasattr(attr, '__dataclass_fields__') and isinstance(v, dict):
            _apply_dict(attr, v)
        else:
            setattr(cfg, k, v)


def load_config(path: str | Path | None = None) -> KBConfig:
    """
    从 YAML 文件加载配置，文件不存在时返回默认配置。

    查找顺序：
      1. 显式传入的 path
      2. 环境变量 KB_CONFIG_FILE
      3. 项目根目录 kb_config.yaml
    """
    if path is None:
        path = os.environ.get("KB_CONFIG_FILE", "kb_config.yaml")

    cfg = KBConfig()
    p = Path(path)
    if not p.exists():
        return cfg

    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    kb_data = raw.get("knowledge_base", raw)
    _apply_dict(cfg, kb_data)
    return cfg


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_config: KBConfig | None = None


def get_kb_config() -> KBConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_kb_config(cfg: KBConfig | None = None) -> None:
    """测试用：重置全局配置单例。"""
    global _config
    _config = cfg
