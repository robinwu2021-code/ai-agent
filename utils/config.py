"""
utils/config.py — 完整 LLM 连接与模型配置

所有 LLM 连接信息和模型选择均通过环境变量或 .env 文件配置，
代码中无任何硬编码的 API Key 或模型名。

优先级：环境变量 > .env 文件 > 默认值

支持的 LLM 提供商：
  anthropic   Claude 系列（默认）
  openai      OpenAI GPT 系列
  azure       Azure OpenAI
  deepseek    DeepSeek（OpenAI 兼容）
  groq        Groq（OpenAI 兼容）
  ollama      本地 Ollama（OpenAI 兼容）
  custom      任意 OpenAI 兼容接口
"""
from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

import structlog
log = structlog.get_logger(__name__)


# ── 已知提供商的默认端点 ─────────────────────────────────────────
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "base_url":        "",   # SDK 自动处理，不需要手动设置
        "default_model":   "claude-sonnet-4-20250514",
        "embedding_model": "",   # Anthropic 暂无 embedding API
    },
    "openai": {
        "base_url":        "https://api.openai.com/v1",
        "default_model":   "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "azure": {
        "base_url":        "",   # 由 AZURE_OPENAI_ENDPOINT 单独设置
        "default_model":   "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "deepseek": {
        "base_url":        "https://api.deepseek.com/v1",
        "default_model":   "deepseek-chat",
        "embedding_model": "",
    },
    "groq": {
        "base_url":        "https://api.groq.com/openai/v1",
        "default_model":   "llama-3.3-70b-versatile",
        "embedding_model": "",
    },
    "ollama": {
        "base_url":        "http://localhost:11434/v1",
        "default_model":   "llama3.2",
        "embedding_model": "nomic-embed-text",
    },
    "custom": {
        "base_url":        "",
        "default_model":   "",
        "embedding_model": "",
    },
}


class LLMProviderConfig(BaseSettings):
    """
    单个 LLM 提供商的连接配置。
    通过前缀环境变量读取，例如主提供商用 LLM_ 前缀。
    """

    # ── 提供商选择 ────────────────────────────────────────────────
    provider:   str = Field("anthropic", description="提供商: anthropic|openai|azure|deepseek|groq|ollama|custom")

    # ── 认证 ──────────────────────────────────────────────────────
    api_key:    str | None = Field(None, description="API Key（也可通过 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 设置）")

    # ── 连接端点 ──────────────────────────────────────────────────
    base_url:   str | None = Field(None, description="API Base URL，留空使用提供商默认值")

    # Azure 专用
    azure_endpoint:    str | None = Field(None, description="Azure OpenAI endpoint URL")
    azure_api_version: str        = Field("2024-02-01", description="Azure API version")
    azure_deployment:  str | None = Field(None, description="Azure deployment name")

    # ── 模型选择 ──────────────────────────────────────────────────
    model:           str | None = Field(None, description="推理模型名，留空使用提供商默认值")
    embedding_model: str | None = Field(None, description="Embedding 模型名，留空使用提供商默认值")
    summarize_model: str | None = Field(None, description="摘要专用模型（可使用更便宜的模型），留空同 model")

    # ── 请求参数 ──────────────────────────────────────────────────
    max_tokens:     int   = Field(4096,  description="单次最大输出 token 数")
    timeout_sec:    float = Field(120.0, description="请求超时秒数")
    max_retries:    int   = Field(3,     description="自动重试次数")
    temperature:    float = Field(0.7,   description="采样温度 0-1")

    # ── 代理 ─────────────────────────────────────────────────────
    http_proxy:     str | None = Field(None, description="HTTP 代理，例如 http://127.0.0.1:7890")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid = set(PROVIDER_DEFAULTS.keys())
        if v not in valid:
            raise ValueError(f"provider 必须是 {valid} 之一，得到: {v!r}")
        return v

    def resolved_model(self) -> str:
        """返回最终使用的推理模型名。"""
        if self.model:
            return self.model
        return PROVIDER_DEFAULTS[self.provider]["default_model"]

    def resolved_embedding_model(self) -> str:
        """返回最终使用的 embedding 模型名。"""
        if self.embedding_model:
            return self.embedding_model
        return PROVIDER_DEFAULTS[self.provider]["embedding_model"]

    def resolved_summarize_model(self) -> str:
        """返回摘要专用模型（默认与推理模型相同）。"""
        return self.summarize_model or self.resolved_model()

    def resolved_base_url(self) -> str | None:
        """返回最终使用的 base URL。"""
        if self.base_url:
            return self.base_url
        default = PROVIDER_DEFAULTS[self.provider]["base_url"]
        return default or None

    def resolved_api_key(self) -> str | None:
        """
        按优先级查找 API Key：
          1. 本字段 api_key
          2. 提供商专用环境变量（ANTHROPIC_API_KEY / OPENAI_API_KEY 等）
        """
        import os
        if self.api_key:
            return self.api_key
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "azure":     "AZURE_OPENAI_API_KEY",
            "deepseek":  "DEEPSEEK_API_KEY",
            "groq":      "GROQ_API_KEY",
            "ollama":    "",            # Ollama 本地无需 key
            "custom":    "CUSTOM_LLM_API_KEY",
        }
        env_var = env_map.get(self.provider, "")
        return os.environ.get(env_var) if env_var else None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8",
                    "populate_by_name": True, "extra": "ignore"}


class Settings(BaseSettings):
    """
    Agent 系统全局配置。
    优先级：环境变量 > .env 文件 > 默认值
    """

    # ── 主 LLM（必填） ────────────────────────────────────────────
    llm_provider:          str        = Field("anthropic",                alias="LLM_PROVIDER")
    llm_api_key:           str | None = Field(None,                       alias="LLM_API_KEY")
    llm_base_url:          str | None = Field(None,                       alias="LLM_BASE_URL")
    llm_model:             str | None = Field(None,                       alias="LLM_MODEL")
    llm_embedding_model:   str | None = Field(None,                       alias="LLM_EMBEDDING_MODEL")
    llm_summarize_model:   str | None = Field(None,                       alias="LLM_SUMMARIZE_MODEL")
    llm_max_tokens:        int        = Field(4096,                       alias="LLM_MAX_TOKENS")
    llm_timeout_sec:       float      = Field(120.0,                      alias="LLM_TIMEOUT_SEC")
    llm_max_retries:       int        = Field(3,                          alias="LLM_MAX_RETRIES")
    llm_temperature:       float      = Field(0.7,                        alias="LLM_TEMPERATURE")
    llm_http_proxy:        str | None = Field(None,                       alias="LLM_HTTP_PROXY")

    # Azure 专用
    azure_endpoint:        str | None = Field(None,                       alias="AZURE_OPENAI_ENDPOINT")
    azure_api_version:     str        = Field("2024-02-01",               alias="AZURE_OPENAI_API_VERSION")
    azure_deployment:      str | None = Field(None,                       alias="AZURE_OPENAI_DEPLOYMENT")

    # 兼容旧版环境变量（自动回退读取）
    anthropic_api_key:     str | None = Field(None,                       alias="ANTHROPIC_API_KEY")
    openai_api_key:        str | None = Field(None,                       alias="OPENAI_API_KEY")
    openai_base_url:       str | None = Field(None,                       alias="OPENAI_BASE_URL")

    # ── 存储后端 ─────────────────────────────────────────────────
    redis_url:             str        = Field("redis://localhost:6379",   alias="REDIS_URL")
    qdrant_url:            str        = Field("http://localhost:6333",    alias="QDRANT_URL")
    use_redis:             bool       = Field(False,                      alias="USE_REDIS")
    use_qdrant:            bool       = Field(False,                      alias="USE_QDRANT")

    # ── API 服务器 ────────────────────────────────────────────────
    api_host:              str        = Field("0.0.0.0",                  alias="API_HOST")
    api_port:              int        = Field(8000,                       alias="API_PORT")

    # ── Agent 运行参数 ─────────────────────────────────────────────
    orchestrator_type:     str        = Field("react",                    alias="ORCHESTRATOR_TYPE")
    max_steps:             int        = Field(20,                         alias="MAX_STEPS")
    token_budget:          int        = Field(12000,                      alias="TOKEN_BUDGET")

    # ── 日志 ─────────────────────────────────────────────────────
    log_level:             str        = Field("INFO",                     alias="LOG_LEVEL")
    json_logs:             bool       = Field(False,                      alias="JSON_LOGS")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8",
                    "populate_by_name": True, "extra": "ignore"}

    # ── 派生方法 ──────────────────────────────────────────────────

    def llm_config(self) -> LLMProviderConfig:
        """将扁平环境变量组装为 LLMProviderConfig 对象。"""
        return LLMProviderConfig(
            provider=self.llm_provider,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            model=self.llm_model,
            embedding_model=self.llm_embedding_model,
            summarize_model=self.llm_summarize_model,
            max_tokens=self.llm_max_tokens,
            timeout_sec=self.llm_timeout_sec,
            max_retries=self.llm_max_retries,
            temperature=self.llm_temperature,
            http_proxy=self.llm_http_proxy,
            azure_endpoint=self.azure_endpoint,
            azure_api_version=self.azure_api_version,
            azure_deployment=self.azure_deployment,
        )

    def effective_api_key(self) -> str | None:
        """
        按优先级解析最终使用的 API Key：
          LLM_API_KEY > ANTHROPIC_API_KEY / OPENAI_API_KEY > 无
        """
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        if self.llm_provider in ("openai", "azure", "deepseek", "groq", "custom"):
            return self.openai_api_key
        return None

    def effective_base_url(self) -> str | None:
        """按优先级解析 base URL：LLM_BASE_URL > OPENAI_BASE_URL > 提供商默认值。"""
        if self.llm_base_url:
            return self.llm_base_url
        if self.openai_base_url and self.llm_provider != "anthropic":
            return self.openai_base_url
        return PROVIDER_DEFAULTS.get(self.llm_provider, {}).get("base_url") or None

    def effective_model(self) -> str:
        """解析最终推理模型名。"""
        if self.llm_model:
            return self.llm_model
        return PROVIDER_DEFAULTS.get(self.llm_provider, {}).get("default_model", "")

    # ── 多模型路由配置 ────────────────────────────────────────────
    # 引擎别名列表，逗号分隔，如 "claude-smart,claude-fast,gpt-embed"
    router_engines:      str | None = Field(None, alias="ROUTER_ENGINES")

    # 任务路由映射（别名必须在 router_engines 中注册）
    router_chat:         str | None = Field(None, alias="ROUTER_CHAT")
    router_plan:         str | None = Field(None, alias="ROUTER_PLAN")
    router_summarize:    str | None = Field(None, alias="ROUTER_SUMMARIZE")
    router_consolidate:  str | None = Field(None, alias="ROUTER_CONSOLIDATE")
    router_embed:        str | None = Field(None, alias="ROUTER_EMBED")
    router_eval:         str | None = Field(None, alias="ROUTER_EVAL")
    router_route:        str | None = Field(None, alias="ROUTER_ROUTE")
    router_fallback:     str | None = Field(None, alias="ROUTER_FALLBACK")

    def build_router(self) -> Any:
        """
        从配置构建 LLMRouter。
        如果未配置 ROUTER_ENGINES，退化为单引擎 Router（向后兼容）。
        """
        from llm.router import (
            LLMRouter, ModelRegistry, TaskRouter, single_engine_router
        )

        if not self.router_engines:
            # 单引擎模式：包装成 Router
            engine = self.build_llm_engine()
            return single_engine_router(engine, alias="default")

        # 多引擎模式
        engine_aliases = [a.strip() for a in self.router_engines.split(",") if a.strip()]
        registry = ModelRegistry()

        for alias in engine_aliases:
            prefix  = alias.upper().replace("-", "_")
            import os
            provider = os.environ.get(f"{prefix}_PROVIDER") or self.llm_provider
            api_key  = os.environ.get(f"{prefix}_API_KEY")  or self.effective_api_key()
            base_url = os.environ.get(f"{prefix}_BASE_URL")
            model    = os.environ.get(f"{prefix}_MODEL")
            sup_embed = os.environ.get(f"{prefix}_EMBED", "").lower() == "true"

            # 构造一个临时 Settings 只为该引擎
            engine_settings = Settings(
                LLM_PROVIDER=provider,
                LLM_API_KEY=api_key,
                LLM_BASE_URL=base_url,
                LLM_MODEL=model,
            )
            engine = engine_settings.build_llm_engine()
            registry.register(
                alias=alias,
                engine=engine,
                provider=provider,
                model=model or engine_settings.effective_model(),
                supports_embed=sup_embed,
            )

        fallback_chain = (
            [a.strip() for a in self.router_fallback.split(",") if a.strip()]
            if self.router_fallback else []
        )

        task_router = TaskRouter(
            default=engine_aliases[0],
            chat=self.router_chat,
            plan=self.router_plan,
            summarize=self.router_summarize,
            consolidate=self.router_consolidate,
            embed=self.router_embed,
            eval=self.router_eval,
            route=self.router_route,
            fallback=fallback_chain,
        )

        router = LLMRouter(registry, task_router)
        import structlog
        structlog.get_logger(__name__).info(
            "router.built",
            engines=engine_aliases,
            fallback=fallback_chain,
        )
        return router

    def build_llm_engine(self) -> Any:
        """根据 LLM_PROVIDER 实例化对应的 LLMEngine。"""
        from llm.engines import AnthropicEngine, OpenAIEngine, MockLLMEngine

        cfg = self.llm_config()
        api_key  = self.effective_api_key() or cfg.resolved_api_key()
        base_url = self.effective_base_url()
        model    = self.effective_model()
        emb_model = cfg.resolved_embedding_model()
        sum_model = cfg.resolved_summarize_model()

        log.info("llm.engine.init",
                 provider=self.llm_provider,
                 model=model,
                 base_url=base_url or "(default)",
                 embedding_model=emb_model or "(none)")

        if self.llm_provider == "anthropic":
            return AnthropicEngine(
                api_key=api_key,
                default_model=model,
                summarize_model=sum_model,
                max_tokens=cfg.max_tokens,
                timeout_sec=cfg.timeout_sec,
                max_retries=cfg.max_retries,
                http_proxy=cfg.http_proxy,
            )

        if self.llm_provider == "azure":
            return OpenAIEngine(
                api_key=api_key,
                base_url=cfg.azure_endpoint,
                default_model=cfg.azure_deployment or model,
                embedding_model=emb_model,
                summarize_model=sum_model,
                max_tokens=cfg.max_tokens,
                timeout_sec=cfg.timeout_sec,
                max_retries=cfg.max_retries,
                http_proxy=cfg.http_proxy,
                azure_api_version=cfg.azure_api_version,
                is_azure=True,
            )

        if self.llm_provider in ("openai", "deepseek", "groq", "ollama", "custom"):
            return OpenAIEngine(
                api_key=api_key,
                base_url=base_url,
                default_model=model,
                embedding_model=emb_model,
                summarize_model=sum_model,
                max_tokens=cfg.max_tokens,
                timeout_sec=cfg.timeout_sec,
                max_retries=cfg.max_retries,
                http_proxy=cfg.http_proxy,
            )

        # fallback
        log.warning("llm.engine.unknown_provider", provider=self.llm_provider)
        return MockLLMEngine()

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

        cost_tracker    = CostTracker()
        quota_manager   = QuotaManager(cost_tracker)
        model_downgrader = ModelDowngrader(quota_manager)
        registry_pm     = PromptRegistry()
        ab_router       = ABTestRouter(registry_pm)
        prompt_renderer = PromptRenderer(registry_pm, ab_router)

        c = AgentContainer(
            llm_engine=llm,
            short_term_memory=stm,
            long_term_memory=ltm,
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(llm_engine=llm),
            orchestrator_type=self.orchestrator_type,
            security_manager=SecurityManager(),
            cost_tracker=cost_tracker,
            quota_manager=quota_manager,
            model_downgrader=model_downgrader,
            feedback_store=FeedbackStore(),
            tenant_manager=TenantManager(),
            task_queue=TaskQueue(),
            prompt_renderer=prompt_renderer,
        )
        return c.build()


# ── 单例 ──────────────────────────────────────────────────────────
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
