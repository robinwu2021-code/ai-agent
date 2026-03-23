"""
utils/config.py — Agent 系统全局配置

LLM 实例配置已迁移至 utils/llm_config.py（LLMConfig / RouterConfig）。
本文件负责：
  1. 系统级配置（存储后端、API 服务器、日志等）
  2. 单引擎快捷配置（LLM_* 前缀，向后兼容旧版环境变量）
  3. 多引擎路由配置（ENGINE_ALIASES + 各实例独立前缀）

优先级：环境变量 > .env 文件 > 默认值

──────────────────────────────────────────────────────────────────
单引擎模式（最简配置，向后兼容）：

    LLM_SDK=anthropic               # 或 openai_compatible
    LLM_API_KEY=sk-ant-xxx
    LLM_MODEL=claude-sonnet-4-20250514

  旧版 provider 风格也继续支持：
    LLM_PROVIDER=anthropic          # 自动映射到 sdk 类型
    ANTHROPIC_API_KEY=sk-ant-xxx

──────────────────────────────────────────────────────────────────
多引擎路由模式（详见 utils/llm_config.py 中的完整示例）：

    ENGINE_ALIASES=opus-prod,haiku-backup,azure-east

    OPUS_PROD_SDK=anthropic
    OPUS_PROD_API_KEY=sk-ant-111
    OPUS_PROD_MODEL=claude-opus-4-5

    HAIKU_BACKUP_SDK=anthropic
    HAIKU_BACKUP_API_KEY=sk-ant-222     # 不同账号
    HAIKU_BACKUP_MODEL=claude-haiku-4-5-20251001

    AZURE_EAST_SDK=openai_compatible
    AZURE_EAST_API_KEY=az-key
    AZURE_EAST_BASE_URL=https://my-east.openai.azure.com/...
    AZURE_EAST_MODEL=gpt-4o
    AZURE_EAST_IS_AZURE=true

    ROUTER_DEFAULT=opus-prod
    ROUTER_SUMMARIZE=haiku-backup
    ROUTER_EMBED=azure-east
    ROUTER_FALLBACK=haiku-backup,azure-east
"""
from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

import structlog

log = structlog.get_logger(__name__)

# ── 旧版 LLM_PROVIDER 到 sdk 类型的映射（向后兼容）──────────────
_PROVIDER_TO_SDK: dict[str, str] = {
    "anthropic": "anthropic",
    "openai":    "openai_compatible",
    "azure":     "openai_compatible",
    "deepseek":  "openai_compatible",
    "groq":      "openai_compatible",
    "ollama":    "openai_compatible",
    "custom":    "openai_compatible",
}

# 旧版 LLM_PROVIDER 对应的默认 base_url（azure 除外，由 base_url 指定）
_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai":   "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq":     "https://api.groq.com/openai/v1",
    "ollama":   "http://localhost:11434/v1",
}


class Settings(BaseSettings):
    """
    Agent 系统全局配置。
    优先级：环境变量 > .env 文件 > 默认值
    """

    # ── 单引擎快捷配置（alias="default"，向后兼容）────────────────
    # 新：LLM_SDK=anthropic|openai_compatible
    llm_sdk:               str        = Field("anthropic",  alias="LLM_SDK")
    # 旧：LLM_PROVIDER=anthropic|openai|azure|deepseek|groq|ollama|custom
    llm_provider:          str | None = Field(None,         alias="LLM_PROVIDER")

    llm_api_key:           str | None = Field(None,         alias="LLM_API_KEY")
    llm_base_url:          str | None = Field(None,         alias="LLM_BASE_URL")
    llm_model:             str | None = Field(None,         alias="LLM_MODEL")
    llm_embedding_model:   str | None = Field(None,         alias="LLM_EMBEDDING_MODEL")
    llm_summarize_model:   str | None = Field(None,         alias="LLM_SUMMARIZE_MODEL")
    llm_max_tokens:        int        = Field(4096,         alias="LLM_MAX_TOKENS")
    llm_timeout_sec:       float      = Field(120.0,        alias="LLM_TIMEOUT_SEC")
    llm_max_retries:       int        = Field(3,            alias="LLM_MAX_RETRIES")
    llm_temperature:       float      = Field(0.7,          alias="LLM_TEMPERATURE")
    llm_http_proxy:        str | None = Field(None,         alias="LLM_HTTP_PROXY")
    llm_is_azure:          bool       = Field(False,        alias="LLM_IS_AZURE")
    llm_azure_api_version: str        = Field("2024-02-01", alias="LLM_AZURE_API_VERSION")

    # 兼容旧版厂商专用环境变量
    anthropic_api_key:     str | None = Field(None, alias="ANTHROPIC_API_KEY")
    openai_api_key:        str | None = Field(None, alias="OPENAI_API_KEY")
    openai_base_url:       str | None = Field(None, alias="OPENAI_BASE_URL")

    # ── YAML 配置文件路径（最高优先级）──────────────────────────
    # 默认读取工作目录下的 llm.yaml；设为空字符串可强制禁用 YAML 加载
    llm_config_file: str = Field("llm.yaml", alias="LLM_CONFIG_FILE")

    # ── 多引擎路由配置（YAML 不存在时生效）──────────────────────
    # 逗号分隔的实例别名列表，如 "opus-prod,haiku-backup,azure-east"
    # 每个 alias 对应 {ALIAS}_SDK / {ALIAS}_API_KEY / ... 环境变量
    engine_aliases: str | None = Field(None, alias="ENGINE_ALIASES")

    # 任务路由（alias 必须在 engine_aliases 中声明）
    router_default:     str | None = Field(None, alias="ROUTER_DEFAULT")
    router_chat:        str | None = Field(None, alias="ROUTER_CHAT")
    router_plan:        str | None = Field(None, alias="ROUTER_PLAN")
    router_summarize:   str | None = Field(None, alias="ROUTER_SUMMARIZE")
    router_consolidate: str | None = Field(None, alias="ROUTER_CONSOLIDATE")
    router_embed:       str | None = Field(None, alias="ROUTER_EMBED")
    router_eval:        str | None = Field(None, alias="ROUTER_EVAL")
    router_route:       str | None = Field(None, alias="ROUTER_ROUTE")
    router_fallback:    str | None = Field(None, alias="ROUTER_FALLBACK")

    # ── 存储后端 ─────────────────────────────────────────────────
    redis_url:  str  = Field("redis://localhost:6379", alias="REDIS_URL")
    qdrant_url: str  = Field("http://localhost:6333",  alias="QDRANT_URL")
    use_redis:  bool = Field(False,                    alias="USE_REDIS")
    use_qdrant: bool = Field(False,                    alias="USE_QDRANT")

    # ── 向量数据库后端（llm.yaml vector_store 节优先，以下为环境变量兜底）──
    # 选择后端：milvus | qdrant | sqlite（默认 sqlite，向后兼容）
    vector_backend:      str = Field("sqlite",          alias="VECTOR_BACKEND")
    # Milvus
    milvus_uri:          str = Field("",                alias="MILVUS_URI")
    milvus_token:        str = Field("",                alias="MILVUS_TOKEN")
    milvus_collection:   str = Field("kb_chunks",       alias="MILVUS_COLLECTION")
    milvus_vector_size:  int = Field(1536,              alias="MILVUS_VECTOR_SIZE")
    milvus_index_type:   str = Field("HNSW",            alias="MILVUS_INDEX_TYPE")
    # Qdrant（env var 兜底，llm.yaml qdrant 节优先）
    qdrant_path:         str = Field("./data/qdrant",   alias="QDRANT_PATH")
    qdrant_api_key:      str = Field("",                alias="QDRANT_API_KEY")
    qdrant_vector_size:  int = Field(1536,              alias="QDRANT_VECTOR_SIZE")

    # ── API 服务器 ────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000,      alias="API_PORT")

    # ── Agent 运行参数 ─────────────────────────────────────────────
    orchestrator_type: str = Field("react", alias="ORCHESTRATOR_TYPE")
    max_steps:         int = Field(20,      alias="MAX_STEPS")
    token_budget:      int = Field(12000,   alias="TOKEN_BUDGET")

    # ── 日志 ─────────────────────────────────────────────────────
    log_level: str  = Field("INFO",  alias="LOG_LEVEL")
    json_logs: bool = Field(False,   alias="JSON_LOGS")

    # ── LLM 调用全链路日志 ────────────────────────────────────────
    # 每次 chat / stream_chat / embed 均会产生 request / response / error 三类事件
    llm_call_log_enabled:      bool  = Field(True,                    alias="LLM_CALL_LOG_ENABLED")
    llm_call_log_level:        str   = Field("DEBUG",                 alias="LLM_CALL_LOG_LEVEL")
    llm_call_log_file:         str   = Field("logs/llm_calls.jsonl",  alias="LLM_CALL_LOG_FILE")
    llm_call_log_max_bytes:    int   = Field(10 * 1024 * 1024,        alias="LLM_CALL_LOG_MAX_BYTES")
    llm_call_log_backup_count: int   = Field(5,                       alias="LLM_CALL_LOG_BACKUP_COUNT")
    llm_call_log_msg_preview:  int   = Field(500,                     alias="LLM_CALL_LOG_MSG_PREVIEW")

    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name":  True,
        "extra":             "ignore",
    }

    # ── 向量数据库后端构建 ────────────────────────────────────────

    def build_vector_store(self) -> Any:
        """
        构建向量数据库后端实例。

        优先级：
          1. llm.yaml vector_store 节（最高，含 ${ENV_VAR} 插值）
          2. VECTOR_BACKEND / MILVUS_* / QDRANT_* 环境变量（兜底）
          3. 无配置 → 返回 None（退回 SQLite 模式）

        返回：MilvusVectorStore | QdrantVectorStore | None
        """
        from utils.llm_config import VectorStoreConfig, load_from_yaml
        from pathlib import Path

        # 优先级 1：llm.yaml vector_store 节
        if self.llm_config_file:
            yaml_path = Path(self.llm_config_file)
            if yaml_path.exists():
                try:
                    _, _, vs_cfg, _ = load_from_yaml(yaml_path)
                    if vs_cfg and vs_cfg.backend != "sqlite":
                        return vs_cfg.build()
                except Exception as exc:
                    log.warning("settings.build_vector_store.yaml_failed",
                                error=str(exc))

        # 优先级 2：环境变量兜底
        backend = self.vector_backend.lower().strip()
        if backend == "milvus":
            vs_cfg = VectorStoreConfig(
                backend            = "milvus",
                milvus_uri         = self.milvus_uri,
                milvus_token       = self.milvus_token,
                milvus_collection  = self.milvus_collection,
                milvus_vector_size = self.milvus_vector_size,
                milvus_index_type  = self.milvus_index_type,
            )
            return vs_cfg.build()

        if backend == "qdrant":
            vs_cfg = VectorStoreConfig(
                backend            = "qdrant",
                qdrant_url         = self.qdrant_url if self.use_qdrant else "",
                qdrant_path        = self.qdrant_path,
                qdrant_api_key     = self.qdrant_api_key,
                qdrant_vector_size = self.qdrant_vector_size,
            )
            return vs_cfg.build()

        return None   # sqlite fallback

    # ── 单引擎 LLMConfig 构建 ─────────────────────────────────────

    def default_llm_config(self) -> "LLMConfig":
        """
        将 LLM_* 环境变量组装为 alias='default' 的 LLMConfig。

        SDK 类型解析优先级：LLM_SDK > LLM_PROVIDER 映射 > 默认 anthropic
        API Key 解析优先级：LLM_API_KEY > ANTHROPIC_API_KEY / OPENAI_API_KEY
        Base URL 解析优先级：LLM_BASE_URL > OPENAI_BASE_URL > LLM_PROVIDER 默认值
        """
        from utils.llm_config import LLMConfig

        # 解析 sdk 类型
        sdk = self.llm_sdk
        if self.llm_provider and self.llm_provider in _PROVIDER_TO_SDK:
            # LLM_PROVIDER 覆盖 LLM_SDK（旧版兼容优先）
            sdk = _PROVIDER_TO_SDK[self.llm_provider]

        # 解析 API Key
        api_key = self.llm_api_key
        if not api_key:
            api_key = (
                self.anthropic_api_key
                if sdk == "anthropic"
                else self.openai_api_key
            )

        # 解析 base_url
        base_url = self.llm_base_url
        if not base_url and sdk == "openai_compatible":
            # OPENAI_BASE_URL 兼容
            base_url = self.openai_base_url
        if not base_url and self.llm_provider in _PROVIDER_BASE_URLS:
            # LLM_PROVIDER 对应的厂商默认地址
            base_url = _PROVIDER_BASE_URLS[self.llm_provider]

        # Azure 模式：LLM_PROVIDER=azure 隐式启用
        is_azure = self.llm_is_azure or self.llm_provider == "azure"

        return LLMConfig(
            alias             = "default",
            sdk               = sdk,          # type: ignore[arg-type]
            api_key           = api_key,
            base_url          = base_url,
            model             = self.llm_model,
            embedding_model   = self.llm_embedding_model,
            summarize_model   = self.llm_summarize_model,
            max_tokens        = self.llm_max_tokens,
            timeout_sec       = self.llm_timeout_sec,
            max_retries       = self.llm_max_retries,
            temperature       = self.llm_temperature,
            http_proxy        = self.llm_http_proxy,
            is_azure          = is_azure,
            azure_api_version = self.llm_azure_api_version,
            supports_embed    = True,
        )

    def build_llm_engine(self) -> Any:
        """构建单引擎实例（无 ENGINE_ALIASES 时使用）。"""
        return self.default_llm_config().build_engine()

    # ── 多引擎路由构建 ────────────────────────────────────────────

    def build_router(self) -> Any:
        """
        构建 LLMRouter，配置来源优先级：
          1. llm.yaml（LLM_CONFIG_FILE，默认 "llm.yaml"）
          2. ENGINE_ALIASES + {ALIAS}_* 环境变量
          3. LLM_* 单引擎环境变量（包装为单引擎 Router，向后兼容）
        """
        from llm.router import LLMRouter, ModelRegistry, TaskRouter, single_engine_router
        from utils.llm_config import LLMConfig, load_from_yaml

        # ── 优先级 1：YAML 文件 ───────────────────────────────────
        if self.llm_config_file:
            from pathlib import Path
            yaml_path = Path(self.llm_config_file)
            if yaml_path.exists():
                configs, router_cfg, _vs, _mem = load_from_yaml(yaml_path)
                return self._build_router_from_configs(configs, router_cfg)
            else:
                log.debug("router.yaml_not_found",
                          path=str(yaml_path), fallback="env_vars")

        # ── 优先级 2：ENGINE_ALIASES 环境变量 ─────────────────────
        if self.engine_aliases:
            aliases = [a.strip() for a in self.engine_aliases.split(",") if a.strip()]
            configs = [LLMConfig.from_env(alias) for alias in aliases]
            fallback = [
                a.strip()
                for a in (self.router_fallback or "").split(",")
                if a.strip()
            ]
            router_cfg_cls = __import__(
                "utils.llm_config", fromlist=["RouterConfig"]
            ).RouterConfig
            router_cfg = router_cfg_cls(
                default     = self.router_default or aliases[0],
                chat        = self.router_chat,
                plan        = self.router_plan,
                summarize   = self.router_summarize,
                consolidate = self.router_consolidate,
                embed       = self.router_embed,
                eval        = self.router_eval,
                route       = self.router_route,
                fallback    = fallback,
            )
            return self._build_router_from_configs(configs, router_cfg)

        # ── 优先级 3：LLM_* 单引擎（向后兼容）────────────────────
        engine = self.build_llm_engine()
        return single_engine_router(engine, alias="default")

    def _build_router_from_configs(
        self,
        configs: list,
        router_cfg: Any,
    ) -> Any:
        """将 LLMConfig 列表 + RouterConfig 组装成 LLMRouter。"""
        from llm.router import LLMRouter, ModelRegistry, TaskRouter

        registry = ModelRegistry()
        for cfg in configs:
            engine = cfg.build_engine()
            registry.register(
                alias          = cfg.alias,
                engine         = engine,
                provider       = cfg.sdk,
                model          = cfg.resolved_model(),
                supports_embed = cfg.supports_embed,
                supports_tools = cfg.supports_tools,
                cost_tier      = cfg.cost_tier,
                tags           = cfg.tags,
            )

        task_router = TaskRouter(
            default        = router_cfg.default,
            chat           = router_cfg.chat,
            plan           = router_cfg.plan,
            summarize      = router_cfg.summarize,
            consolidate    = router_cfg.consolidate,
            embed          = router_cfg.embed,
            rerank         = router_cfg.rerank,
            eval           = router_cfg.eval,
            route          = router_cfg.route,
            node_overrides = router_cfg.node_overrides,
            fallback       = router_cfg.fallback,
        )

        router = LLMRouter(registry, task_router)
        log.info(
            "router.built",
            engines  = [c.alias for c in configs],
            default  = router_cfg.default,
            fallback = router_cfg.fallback,
        )
        return router

    def build_container(self):
        """根据配置自动装配 AgentContainer。"""
        from core.container import AgentContainer

        llm = self.build_llm_engine()

        if self.use_redis:
            from memory.stores import RedisShortTermMemory
            stm = RedisShortTermMemory(redis_url=self.redis_url)
        else:
            from memory.stores import InMemoryShortTermMemory
            stm = InMemoryShortTermMemory()

        if self.use_qdrant:
            from memory.stores import QdrantLongTermMemory
            ltm = QdrantLongTermMemory(url=self.qdrant_url, embed_fn=llm.embed)
        else:
            from memory.stores import InMemoryLongTermMemory
            ltm = InMemoryLongTermMemory()

        from skills.registry import (
            LocalSkillRegistry, PythonExecutorSkill,
            FileReadSkill, FileWriteSkill, WebSearchSkill,
        )
        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(FileReadSkill())
        registry.register(FileWriteSkill())
        registry.register(WebSearchSkill())

        from mcp.hub import DefaultMCPHub
        from context.manager import PriorityContextManager
        from security.middleware import SecurityManager
        from eval.framework import FeedbackStore
        from tenant.manager import TenantManager
        from queue.scheduler import TaskQueue
        from prompt_mgr.manager import PromptRegistry, ABTestRouter, PromptRenderer
        from utils.cost import CostTracker, QuotaManager, ModelDowngrader

        cost_tracker     = CostTracker()
        quota_manager    = QuotaManager(cost_tracker)
        model_downgrader = ModelDowngrader(quota_manager)
        registry_pm      = PromptRegistry()
        ab_router        = ABTestRouter(registry_pm)
        prompt_renderer  = PromptRenderer(registry_pm, ab_router)

        c = AgentContainer(
            llm_engine        = llm,
            short_term_memory = stm,
            long_term_memory  = ltm,
            skill_registry    = registry,
            mcp_hub           = DefaultMCPHub(),
            context_manager   = PriorityContextManager(llm_engine=llm),
            orchestrator_type = self.orchestrator_type,
            security_manager  = SecurityManager(),
            cost_tracker      = cost_tracker,
            quota_manager     = quota_manager,
            model_downgrader  = model_downgrader,
            feedback_store    = FeedbackStore(),
            tenant_manager    = TenantManager(),
            task_queue        = TaskQueue(),
            prompt_renderer   = prompt_renderer,
        )
        return c.build()


# ── 单例 ──────────────────────────────────────────────────────────
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# ── 类型别名（方便外部 import）────────────────────────────────────
from utils.llm_config import LLMConfig, RouterConfig, SDK_DEFAULTS, VENDOR_BASE_URLS  # noqa: E402, F401
