"""
utils/llm_config.py — LLM 引擎实例配置

核心设计：
  LLMConfig    描述一个具体的引擎实例（alias + sdk + 全部连接参数）
  RouterConfig 描述任务路由规则

设计原则：
  - 以 alias（实例名）为唯一 key，而不是 provider 类型
  - 同一厂商可注册多个实例（不同账号 / 地址 / 模型）互不干扰
  - sdk 只有两个值：anthropic | openai_compatible
  - 所有连接参数均为实例级，彻底消除"一个 key 管一个厂商"的限制

──────────────────────────────────────────────────────────────────
手工配置（直接实例化，适合代码内硬配置或测试）：

    from utils.llm_config import LLMConfig, RouterConfig, VENDOR_BASE_URLS

    configs = [
        LLMConfig(
            alias   = "opus-prod",
            sdk     = "anthropic",
            api_key = "sk-ant-111",
            model   = "claude-opus-4-5",
        ),
        LLMConfig(
            alias   = "haiku-backup",
            sdk     = "anthropic",
            api_key = "sk-ant-222",          # 不同账号，不同 key
            model   = "claude-haiku-4-5-20251001",
            http_proxy = "http://proxy:7890",
        ),
        LLMConfig(
            alias    = "azure-east",
            sdk      = "openai_compatible",
            api_key  = "az-key-east",
            base_url = "https://my-east.openai.azure.com/openai/deployments/gpt4o",
            model    = "gpt-4o",
            is_azure = True,
            supports_embed = True,
        ),
        LLMConfig(
            alias    = "deepseek-main",
            sdk      = "openai_compatible",
            api_key  = "ds-key",
            base_url = VENDOR_BASE_URLS["deepseek"],
            model    = "deepseek-chat",
            cost_tier = 1,
        ),
    ]

    router_cfg = RouterConfig(
        default   = "opus-prod",
        chat      = "opus-prod",
        summarize = "haiku-backup",
        embed     = "azure-east",
        fallback  = ["haiku-backup", "azure-east"],
    )

──────────────────────────────────────────────────────────────────
环境变量配置（每个实例独立前缀，ENGINE_ALIASES 声明列表）：

    ENGINE_ALIASES=opus-prod,haiku-backup,azure-east,deepseek-main

    OPUS_PROD_SDK=anthropic
    OPUS_PROD_API_KEY=sk-ant-111
    OPUS_PROD_MODEL=claude-opus-4-5
    OPUS_PROD_COST_TIER=3

    HAIKU_BACKUP_SDK=anthropic
    HAIKU_BACKUP_API_KEY=sk-ant-222
    HAIKU_BACKUP_MODEL=claude-haiku-4-5-20251001
    HAIKU_BACKUP_HTTP_PROXY=http://proxy:7890

    AZURE_EAST_SDK=openai_compatible
    AZURE_EAST_API_KEY=az-key-east
    AZURE_EAST_BASE_URL=https://my-east.openai.azure.com/openai/deployments/gpt4o
    AZURE_EAST_MODEL=gpt-4o
    AZURE_EAST_IS_AZURE=true
    AZURE_EAST_SUPPORTS_EMBED=true

    DEEPSEEK_MAIN_SDK=openai_compatible
    DEEPSEEK_MAIN_API_KEY=ds-key
    DEEPSEEK_MAIN_BASE_URL=https://api.deepseek.com/v1
    DEEPSEEK_MAIN_MODEL=deepseek-chat
    DEEPSEEK_MAIN_COST_TIER=1

    ROUTER_DEFAULT=opus-prod
    ROUTER_CHAT=opus-prod
    ROUTER_SUMMARIZE=haiku-backup
    ROUTER_EMBED=azure-east
    ROUTER_FALLBACK=haiku-backup,azure-east

──────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

log = structlog.get_logger(__name__)

# ── SDK 类型 ──────────────────────────────────────────────────────
SdkType = Literal["anthropic", "openai_compatible"]

# ── SDK 级技术默认值（与厂商无关，仅在实例未指定 model 时回落）────
SDK_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "default_model":   "claude-sonnet-4-20250514",
        "embedding_model": "",          # Anthropic 无原生 embedding API
    },
    "openai_compatible": {
        "default_model":   "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
}

# ── 知名厂商的 base_url（手工配置时可直接引用）────────────────────
VENDOR_BASE_URLS: dict[str, str] = {
    "openai":   "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq":     "https://api.groq.com/openai/v1",
    "ollama":   "http://localhost:11434/v1",
    # anthropic SDK 内置处理，不需要 base_url
    # azure 每个部署地址不同，不在此预设
}


# ─────────────────────────────────────────────────────────────────
# LLMConfig — 引擎实例配置
# ─────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    描述一个具体的 LLM 引擎实例。

    alias  是全局唯一标识，在 ModelRegistry 和路由规则中引用。
    sdk    决定使用哪个客户端库（anthropic 或 openai_compatible）。
    其余均为实例级参数——同一厂商不同账号 / 地址 / 模型可各自独立配置。
    """

    # ── 必填 ──────────────────────────────────────────────────────
    alias: str       # 唯一标识，如 "opus-prod", "haiku-backup", "azure-east"
    sdk:   SdkType   # "anthropic" | "openai_compatible"

    # ── 连接参数（实例级，同类实例互不干扰）──────────────────────
    api_key:  str | None = None
    base_url: str | None = None   # 留空则用 SDK 内置默认值

    # ── 模型参数（实例级）────────────────────────────────────────
    model:           str | None = None   # 主推理模型，留空回落 SDK_DEFAULTS
    embedding_model: str | None = None   # Embedding 模型
    summarize_model: str | None = None   # 摘要专用模型，留空同 model

    # ── 请求参数 ─────────────────────────────────────────────────
    max_tokens:  int   = 4096
    timeout_sec: float = 120.0
    max_retries: int   = 3
    temperature: float = 0.7
    http_proxy:  str | None = None

    # ── Azure 专用 ────────────────────────────────────────────────
    is_azure:          bool = False
    azure_api_version: str  = "2024-02-01"

    # ── 路由元信息（供 ModelRegistry 使用）──────────────────────
    cost_tier:      int       = 2    # 1=最便宜 3=最贵，用于自动降级排序
    supports_embed: bool      = False
    supports_tools: bool      = True
    tags:           list[str] = field(default_factory=list)

    # ── 解析方法 ─────────────────────────────────────────────────

    def resolved_model(self) -> str:
        """返回最终推理模型名（实例设置 > SDK 默认值）。"""
        return self.model or SDK_DEFAULTS.get(self.sdk, {}).get("default_model", "")

    def resolved_embedding_model(self) -> str:
        """返回最终 Embedding 模型名。"""
        return self.embedding_model or SDK_DEFAULTS.get(self.sdk, {}).get("embedding_model", "")

    def resolved_summarize_model(self) -> str:
        """返回摘要专用模型（默认与推理模型相同）。"""
        return self.summarize_model or self.resolved_model()

    # ── 工厂方法 ─────────────────────────────────────────────────

    def build_engine(self) -> Any:
        """根据 sdk 类型实例化对应的 LLMEngine。"""
        from llm.engines import AnthropicEngine, OpenAIEngine

        log.info(
            "llm_config.build_engine",
            alias    = self.alias,
            sdk      = self.sdk,
            model    = self.resolved_model(),
            base_url = self.base_url or "(sdk-default)",
        )

        if self.sdk == "anthropic":
            return AnthropicEngine(
                api_key         = self.api_key,
                default_model   = self.resolved_model(),
                summarize_model = self.resolved_summarize_model(),
                max_tokens      = self.max_tokens,
                timeout_sec     = self.timeout_sec,
                max_retries     = self.max_retries,
                http_proxy      = self.http_proxy,
            )

        if self.sdk == "openai_compatible":
            return OpenAIEngine(
                api_key           = self.api_key,
                base_url          = self.base_url,
                default_model     = self.resolved_model(),
                embedding_model   = self.resolved_embedding_model(),
                summarize_model   = self.resolved_summarize_model(),
                max_tokens        = self.max_tokens,
                timeout_sec       = self.timeout_sec,
                max_retries       = self.max_retries,
                http_proxy        = self.http_proxy,
                is_azure          = self.is_azure,
                azure_api_version = self.azure_api_version,
            )

        raise ValueError(
            f"[{self.alias}] 未知 sdk 类型: {self.sdk!r}，"
            f"支持: anthropic | openai_compatible"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """
        从字典构建实例配置（通常来自 llm.yaml 解析结果）。
        字典中的字符串值若包含 ${ENV_VAR}，会自动替换为对应环境变量值。
        """
        d = {k: _expand_env(v) for k, v in data.items()}

        alias   = d.get("alias", "")
        sdk_raw = d.get("sdk", "anthropic")
        if sdk_raw not in ("anthropic", "openai_compatible"):
            raise ValueError(
                f"[{alias}] sdk 必须是 anthropic | openai_compatible，"
                f"得到: {sdk_raw!r}"
            )

        return cls(
            alias             = alias,
            sdk               = sdk_raw,              # type: ignore[arg-type]
            api_key           = d.get("api_key") or None,
            base_url          = d.get("base_url") or None,
            model             = d.get("model") or None,
            embedding_model   = d.get("embedding_model") or None,
            summarize_model   = d.get("summarize_model") or None,
            max_tokens        = int(d.get("max_tokens", 4096)),
            timeout_sec       = float(d.get("timeout_sec", 120.0)),
            max_retries       = int(d.get("max_retries", 3)),
            temperature       = float(d.get("temperature", 0.7)),
            http_proxy        = d.get("http_proxy") or None,
            is_azure          = bool(d.get("is_azure", False)),
            azure_api_version = d.get("azure_api_version", "2024-02-01"),
            cost_tier         = int(d.get("cost_tier", 2)),
            supports_embed    = bool(d.get("supports_embed", False)),
            supports_tools    = bool(d.get("supports_tools", True)),
            tags              = list(d.get("tags", [])),
        )

    @classmethod
    def from_env(cls, alias: str) -> "LLMConfig":
        """
        从环境变量读取实例配置。
        前缀规则：alias.upper().replace('-', '_')
        示例：alias="opus-prod" → 读取 OPUS_PROD_SDK, OPUS_PROD_API_KEY, ...
        """
        prefix = alias.upper().replace("-", "_")

        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{key}", default)

        sdk_raw = _get("SDK") or "anthropic"
        if sdk_raw not in ("anthropic", "openai_compatible"):
            raise ValueError(
                f"[{alias}] {prefix}_SDK 必须是 anthropic | openai_compatible，"
                f"得到: {sdk_raw!r}"
            )

        return cls(
            alias             = alias,
            sdk               = sdk_raw,          # type: ignore[arg-type]
            api_key           = _get("API_KEY") or None,
            base_url          = _get("BASE_URL") or None,
            model             = _get("MODEL") or None,
            embedding_model   = _get("EMBEDDING_MODEL") or None,
            summarize_model   = _get("SUMMARIZE_MODEL") or None,
            max_tokens        = int(_get("MAX_TOKENS") or 4096),
            timeout_sec       = float(_get("TIMEOUT_SEC") or 120.0),
            max_retries       = int(_get("MAX_RETRIES") or 3),
            temperature       = float(_get("TEMPERATURE") or 0.7),
            http_proxy        = _get("HTTP_PROXY") or None,
            is_azure          = _get("IS_AZURE", "").lower() == "true",
            azure_api_version = _get("AZURE_API_VERSION") or "2024-02-01",
            cost_tier         = int(_get("COST_TIER") or 2),
            supports_embed    = _get("SUPPORTS_EMBED", "").lower() == "true",
            supports_tools    = _get("SUPPORTS_TOOLS", "true").lower() != "false",
            tags              = [t.strip() for t in _get("TAGS", "").split(",") if t.strip()],
        )


# ─────────────────────────────────────────────────────────────────
# RouterConfig — 任务路由规则
# ─────────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    """
    多引擎路由规则。alias 值必须与已注册的 LLMConfig.alias 对应。

    任务类型说明：
      chat        主推理对话（ReAct 循环核心调用）
      plan        Plan-Execute 规划阶段
      summarize   文本摘要压缩（上下文裁剪、记忆固化）
      consolidate 记忆提炼（可用便宜模型）
      embed       向量化（RAG 检索、长期记忆）
      rerank      RAG 重排序打分
      eval        LLM-as-judge 打分
      route       Multi-Agent 路由决策
    """

    default:     str = "default"   # 所有未覆盖任务的回落引擎

    # ── 按任务类型路由（留 None 则回落 default）──────────────────
    chat:        str | None = None
    plan:        str | None = None
    summarize:   str | None = None
    consolidate: str | None = None
    embed:       str | None = None
    rerank:      str | None = None
    eval:        str | None = None
    route:       str | None = None

    # ── 工作流节点级覆盖（粒度比任务类型更细）────────────────────
    node_overrides: dict[str, str] = field(default_factory=dict)
    # 示例：{"react_step_1": "haiku-backup", "react_final": "opus-prod"}

    # ── Fallback 链（主引擎失败后按序尝试）──────────────────────
    fallback: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict, default_alias: str) -> "RouterConfig":
        """从字典构建路由规则（通常来自 llm.yaml 的 router 节）。"""
        d = {k: _expand_env(v) if isinstance(v, str) else v for k, v in data.items()}

        fallback_raw = d.get("fallback", [])
        fallback = (
            [a.strip() for a in fallback_raw.split(",") if a.strip()]
            if isinstance(fallback_raw, str)
            else list(fallback_raw)
        )

        return cls(
            default        = d.get("default", default_alias),
            chat           = d.get("chat"),
            plan           = d.get("plan"),
            summarize      = d.get("summarize"),
            consolidate    = d.get("consolidate"),
            embed          = d.get("embed"),
            rerank         = d.get("rerank"),
            eval           = d.get("eval"),
            route          = d.get("route"),
            node_overrides = dict(d.get("node_overrides") or {}),
            fallback       = fallback,
        )

    @classmethod
    def from_env(cls, default_alias: str) -> "RouterConfig":
        """从 ROUTER_* 环境变量读取路由规则。"""
        def _get(key: str) -> str | None:
            return os.environ.get(f"ROUTER_{key}") or None

        fallback_raw = _get("FALLBACK") or ""
        fallback = [a.strip() for a in fallback_raw.split(",") if a.strip()]

        return cls(
            default     = _get("DEFAULT") or default_alias,
            chat        = _get("CHAT"),
            plan        = _get("PLAN"),
            summarize   = _get("SUMMARIZE"),
            consolidate = _get("CONSOLIDATE"),
            embed       = _get("EMBED"),
            rerank      = _get("RERANK"),
            eval        = _get("EVAL"),
            route       = _get("ROUTE"),
            fallback    = fallback,
        )


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: Any) -> Any:
    """
    将字符串中的 ${VAR_NAME} 替换为对应的环境变量值。
    非字符串值原样返回。未定义的环境变量保留原始占位符并打印警告。
    """
    if not isinstance(value, str):
        return value

    def _replace(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            log.warning("llm_config.env_var_missing", var=var)
            return m.group(0)   # 保留占位符，不崩溃
        return val

    return _ENV_VAR_RE.sub(_replace, value)


# ─────────────────────────────────────────────────────────────────
# YAML 加载入口
# ─────────────────────────────────────────────────────────────────

def load_from_yaml(
    path: str | Path = "llm.yaml",
) -> tuple[list[LLMConfig], RouterConfig]:
    """
    从 YAML 文件加载引擎实例列表和路由规则。

    返回 (engines, router_cfg)。
    YAML 格式见项目根目录的 llm.yaml 示例。

    异常：
      FileNotFoundError  文件不存在
      ValueError         yaml 结构不符合预期
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError("请安装 pyyaml：pip install pyyaml")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"llm 配置文件不存在: {p.resolve()}")

    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{p} 解析结果不是 dict，请检查 YAML 格式")

    # ── 解析 engines ─────────────────────────────────────────────
    engine_list: list[dict] = raw.get("engines") or []
    if not engine_list:
        raise ValueError(f"{p} 中未找到 engines 列表")

    configs = [LLMConfig.from_dict(e) for e in engine_list]
    default_alias = configs[0].alias

    # ── 解析 router ───────────────────────────────────────────────
    router_raw: dict = raw.get("router") or {}
    router_cfg = RouterConfig.from_dict(router_raw, default_alias)

    log.info(
        "llm_config.yaml_loaded",
        path      = str(p.resolve()),
        engines   = [c.alias for c in configs],
        default   = router_cfg.default,
        fallback  = router_cfg.fallback,
    )
    return configs, router_cfg
