"""
llm/router.py — 多模型路由层

核心设计：
  ModelRegistry   注册任意数量的引擎实例，每个实例有唯一 alias
  TaskRouter      按任务类型（chat / summarize / embed / plan / eval 等）
                  路由到不同的引擎 alias
  LLMRouter       统一入口，实现与单个 LLMEngine 相同的接口，
                  对所有调用方完全透明；内置 Fallback 链和熔断

支持的路由维度：
  1. 按任务类型    chat / summarize / embed / plan / eval / rerank
  2. 按工作流节点  react_step / plan_generate / consolidate / rag_rerank 等
  3. 按模型别名    显式指定 AgentConfig.model_alias
  4. Fallback 链   主模型失败时按序尝试备选

推荐配置方式（通过 utils/llm_config.py）：

    from utils.llm_config import LLMConfig, RouterConfig, VENDOR_BASE_URLS
    from llm.router import LLMRouter, ModelRegistry, TaskRouter

    # 1. 手工定义实例（同一厂商可注册多个，互不干扰）
    configs = [
        LLMConfig(alias="opus-prod",    sdk="anthropic",
                  api_key="sk-ant-111", model="claude-opus-4-5"),
        LLMConfig(alias="haiku-backup", sdk="anthropic",
                  api_key="sk-ant-222", model="claude-haiku-4-5-20251001"),
        LLMConfig(alias="azure-east",   sdk="openai_compatible",
                  api_key="az-key", base_url="https://east.azure.com/...",
                  model="gpt-4o", is_azure=True, supports_embed=True),
        LLMConfig(alias="deepseek",     sdk="openai_compatible",
                  api_key="ds-key", base_url=VENDOR_BASE_URLS["deepseek"],
                  model="deepseek-chat", cost_tier=1),
    ]

    # 2. 注册到 registry
    registry = ModelRegistry()
    for cfg in configs:
        registry.register(cfg.alias, cfg.build_engine(),
                          supports_embed=cfg.supports_embed,
                          cost_tier=cfg.cost_tier)

    # 3. 定义路由规则
    task_router = TaskRouter(
        default   = "opus-prod",
        chat      = "opus-prod",
        summarize = "haiku-backup",
        embed     = "azure-east",
        fallback  = ["haiku-backup", "azure-east", "deepseek"],
    )

    router = LLMRouter(registry, task_router)

    # 当作普通 LLMEngine 使用（所有调用方无需修改）
    resp = await router.chat(messages, tools, config)

    # 在 AgentConfig 里指定节点级覆盖
    config = AgentConfig(model_alias="deepseek")
    resp   = await router.chat(messages, tools, config)

环境变量方式见 utils/llm_config.py 文档。
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

from core.models import AgentConfig, LLMResponse, Message, ToolDescriptor

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# ModelSpec — 注册元信息
# ─────────────────────────────────────────────

@dataclass
class ModelSpec:
    """描述一个已注册引擎的元信息。"""
    alias:       str               # 唯一标识，如 "claude-fast"
    provider:    str               # 提供商名，如 "anthropic"
    model:       str               # 模型名，如 "claude-haiku-4-5-20251001"
    engine:      Any               # 实际 LLMEngine 实例
    capabilities: set[str]        = field(default_factory=lambda: {
        "chat", "summarize", "stream"
    })
    supports_embed:   bool = False
    supports_tools:   bool = True
    cost_tier:        int  = 2     # 1=最便宜, 3=最贵（用于自动降级排序）
    tags:             list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# ModelRegistry — 引擎注册表
# ─────────────────────────────────────────────

class ModelRegistry:
    """
    所有可用引擎的注册表。
    一个 provider 可注册多个别名（对应不同模型或不同参数）。
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(
        self,
        alias: str,
        engine: Any,
        provider: str = "",
        model: str = "",
        capabilities: set[str] | None = None,
        supports_embed: bool = False,
        supports_tools: bool = True,
        cost_tier: int = 2,
        tags: list[str] | None = None,
    ) -> "ModelRegistry":
        """注册一个引擎别名。支持链式调用。"""
        # 自动从引擎实例读取 provider/model（如果有）
        _provider = provider or getattr(engine, "_provider", "") or type(engine).__name__
        _model    = model    or getattr(engine, "_default_model", "") or alias

        spec = ModelSpec(
            alias=alias,
            provider=_provider,
            model=_model,
            engine=engine,
            capabilities=capabilities or {"chat", "summarize", "stream"},
            supports_embed=supports_embed,
            supports_tools=supports_tools,
            cost_tier=cost_tier,
            tags=tags or [],
        )
        self._specs[alias] = spec
        log.info("registry.registered", alias=alias, provider=_provider, model=_model)
        return self

    def get(self, alias: str) -> ModelSpec | None:
        return self._specs.get(alias)

    def require(self, alias: str) -> ModelSpec:
        spec = self._specs.get(alias)
        if spec is None:
            available = list(self._specs.keys())
            raise KeyError(f"Model alias '{alias}' not found. Available: {available}")
        return spec

    def list_all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def find_by_tag(self, tag: str) -> list[ModelSpec]:
        return [s for s in self._specs.values() if tag in s.tags]

    def find_embed(self) -> ModelSpec | None:
        """找第一个支持 embed 的引擎。"""
        return next((s for s in self._specs.values() if s.supports_embed), None)

    def __repr__(self) -> str:
        lines = [f"ModelRegistry({len(self._specs)} engines):"]
        for s in self._specs.values():
            lines.append(f"  {s.alias:20s} {s.provider}/{s.model}  tier={s.cost_tier}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# TaskRouter — 任务类型路由表
# ─────────────────────────────────────────────

@dataclass
class TaskRouter:
    """
    把任务类型映射到引擎别名。
    未配置的任务类型回落到 default。

    任务类型说明：
      chat          主推理对话（ReAct 循环的核心调用）
      plan          Plan-Execute 的规划阶段（生成 TaskStep 列表）
      summarize     文本摘要压缩（上下文裁剪、记忆固化）
      consolidate   记忆固化提炼（可用便宜模型）
      embed         向量化（RAG 检索、长期记忆写入）
      rerank        RAG 重排序打分
      eval          Eval 框架的 LLM-as-judge 打分
      route         Multi-Agent 路由决策
    """
    # ── 必填：默认引擎（所有未覆盖的任务类型回落到这里）────────
    default:     str = "default"

    # ── 可选：按任务类型覆盖 ──────────────────────────────────
    chat:        str | None = None
    plan:        str | None = None
    summarize:   str | None = None
    consolidate: str | None = None
    embed:       str | None = None
    rerank:      str | None = None
    eval:        str | None = None
    route:       str | None = None

    # ── 工作流节点级覆盖（比任务类型更细粒度）────────────────
    node_overrides: dict[str, str] = field(default_factory=dict)
    # 示例：{"react_step_1": "claude-fast", "react_step_final": "claude-smart"}

    # ── Fallback 链 ────────────────────────────────────────────
    fallback: list[str] = field(default_factory=list)
    # 按序尝试，全部失败才报错

    def resolve(self, task_type: str, node_id: str | None = None) -> list[str]:
        """
        返回优先级从高到低的别名列表：
          节点覆盖 > 任务类型 > default > fallback 链
        """
        chain: list[str] = []

        # 1. 节点级覆盖（最高优先级）
        if node_id and node_id in self.node_overrides:
            chain.append(self.node_overrides[node_id])

        # 2. 任务类型映射
        task_alias = getattr(self, task_type, None)
        if task_alias:
            chain.append(task_alias)

        # 3. default
        if self.default not in chain:
            chain.append(self.default)

        # 4. fallback 链（去重，保持顺序）
        for fb in self.fallback:
            if fb not in chain:
                chain.append(fb)

        return chain


# ─────────────────────────────────────────────
# Circuit Breaker（简单计数器熔断）
# ─────────────────────────────────────────────

@dataclass
class _CircuitState:
    failures:     int   = 0
    open_until:   float = 0.0   # Unix timestamp
    threshold:    int   = 3
    cooldown_sec: float = 60.0

    def is_open(self) -> bool:
        if self.open_until and time.monotonic() < self.open_until:
            return True
        return False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown_sec
            log.warning("circuit_breaker.open",
                        failures=self.failures, cooldown=self.cooldown_sec)

    def record_success(self) -> None:
        self.failures  = 0
        self.open_until = 0.0


# ─────────────────────────────────────────────
# Call Stats（路由决策可观测性）
# ─────────────────────────────────────────────

@dataclass
class _CallStats:
    total:    int = 0
    success:  int = 0
    fallback: int = 0
    errors:   int = 0
    total_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(self.total, 1)


# ─────────────────────────────────────────────
# LLMRouter — 统一路由入口
# ─────────────────────────────────────────────

class LLMRouter:
    """
    多模型路由器。对外暴露与单个 LLMEngine 完全相同的接口：
      chat / stream_chat / embed / summarize
    内部按任务类型和节点 ID 路由到正确的引擎，失败时沿 Fallback 链尝试。

    与单引擎的差异：AgentConfig 新增两个可选字段：
      model_alias   直接指定引擎别名（最高优先级）
      node_id       工作流节点标识（用于节点级路由覆盖）
    """

    def __init__(
        self,
        registry:       ModelRegistry,
        task_router:    TaskRouter,
        circuit_breaker_threshold: int   = 3,
        circuit_breaker_cooldown:  float = 60.0,
        enable_fallback: bool = False,
    ) -> None:
        self._registry       = registry
        self._task_router    = task_router
        self._enable_fallback = enable_fallback
        self._circuits:   dict[str, _CircuitState] = defaultdict(
            lambda: _CircuitState(
                threshold=circuit_breaker_threshold,
                cooldown_sec=circuit_breaker_cooldown,
            )
        )
        self._stats:      dict[str, _CallStats] = defaultdict(_CallStats)

    # ── Public API（与 LLMEngine Protocol 完全一致）────────────

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> LLMResponse:
        alias = self._explicit_alias(config)
        chain = [alias] if alias else self._task_router.resolve(
            "chat", node_id=getattr(config, "node_id", None)
        )
        return await self._call_chain(
            chain, "chat",
            lambda eng: eng.chat(messages, tools, config),
        )

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> AsyncIterator[str]:
        alias = self._explicit_alias(config)
        chain = [alias] if alias else self._task_router.resolve(
            "chat", node_id=getattr(config, "node_id", None)
        )
        # stream 不走 fallback（已开始流式不能回滚），直接用第一个健康引擎
        spec  = await self._pick_healthy(chain)
        async for token in spec.engine.stream_chat(messages, tools, config):
            yield token

    async def embed(self, text: str) -> list[float]:
        chain = self._task_router.resolve("embed")
        # chain 为空时才用 find_embed() 兜底（不插队，不覆盖 router.embed 配置）
        if not chain:
            embed_spec = self._registry.find_embed()
            if embed_spec:
                chain = [embed_spec.alias]
        return await self._call_chain(
            chain, "embed",
            lambda eng: eng.embed(text),
        )

    async def summarize(self, text: str, max_tokens: int,
                        node_id: str | None = None) -> str:
        task = "consolidate" if node_id and "consolidat" in node_id else "summarize"
        chain = self._task_router.resolve(task, node_id=node_id)
        # Pass node_id through so engines that support it (e.g. LLMRouter itself
        # when nested) can do further task-specific routing.
        async def _call(eng):
            fn = getattr(eng, "summarize", None)
            if fn is None:
                raise RuntimeError(f"{eng} has no summarize()")
            import inspect
            sig = inspect.signature(fn)
            if "node_id" in sig.parameters:
                return await fn(text, max_tokens, node_id=node_id)
            return await fn(text, max_tokens)
        return await self._call_chain(chain, task, _call)

    async def plan(
        self,
        messages: list[Message],
        config: AgentConfig,
    ) -> LLMResponse:
        """Plan-Execute 规划专用调用，走 plan 任务路由。"""
        chain = self._task_router.resolve("plan")
        return await self._call_chain(
            chain, "plan",
            lambda eng: eng.chat(messages, [], config),
        )

    async def eval_score(self, prompt: str, max_tokens: int = 100) -> str:
        """评估框架 LLM-as-judge 打分，走 eval 任务路由。
        优先调用引擎的 eval_score()；不存在时回落到 summarize()。"""
        chain = self._task_router.resolve("eval")
        async def _call(eng):
            fn = getattr(eng, "eval_score", None)
            if fn is not None:
                return await fn(prompt, max_tokens)
            return await eng.summarize(prompt, max_tokens)
        return await self._call_chain(chain, "eval", _call)

    async def route_decision(self, prompt: str, max_tokens: int = 30) -> str:
        """Multi-Agent 路由决策，走 route 任务路由。
        优先调用引擎的 route_decision()；不存在时回落到 summarize()。"""
        chain = self._task_router.resolve("route")
        async def _call(eng):
            fn = getattr(eng, "route_decision", None)
            if fn is not None:
                return await fn(prompt, max_tokens)
            return await eng.summarize(prompt, max_tokens)
        return await self._call_chain(chain, "route", _call)

    # ── Stats & Introspection ─────────────────────────────────

    def get_stats(self) -> dict[str, dict]:
        return {
            alias: {
                "total":    s.total,
                "success":  s.success,
                "fallback": s.fallback,
                "errors":   s.errors,
                "avg_ms":   round(s.avg_ms, 1),
                "circuit":  "open" if self._circuits[alias].is_open() else "closed",
            }
            for alias, s in self._stats.items()
        }

    def describe(self) -> str:
        """打印路由配置摘要，方便调试。"""
        lines = ["LLMRouter configuration:", str(self._registry), "",
                 "Task routing:"]
        for task in ("chat", "plan", "summarize", "consolidate",
                     "embed", "rerank", "eval", "route"):
            chain = self._task_router.resolve(task)
            lines.append(f"  {task:12s} → {' → '.join(chain)}")
        if self._task_router.node_overrides:
            lines.append("\nNode overrides:")
            for nid, alias in self._task_router.node_overrides.items():
                lines.append(f"  {nid:20s} → {alias}")
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────

    async def _call_chain(
        self, chain: list[str], task: str, fn
    ) -> Any:
        """
        沿 chain 依次尝试引擎，直到成功或全部失败。

        enable_fallback=False（默认）时，chain 被截断为仅第一个引擎：
          - 主引擎失败立即上抛，不尝试后备引擎
          - 优点：快速暴露配置/网络问题，不产生误导性的"降级成功"日志

        enable_fallback=True 时保持原有行为：
          - 按 chain 顺序逐个重试，直到成功或耗尽
        """
        effective_chain = chain if self._enable_fallback else chain[:1]

        last_exc: Exception | None = None
        for i, alias in enumerate(effective_chain):
            spec = self._registry.get(alias)
            if spec is None:
                log.warning("router.alias_not_found", alias=alias, task=task)
                continue
            if self._circuits[alias].is_open():
                log.warning("router.circuit_open", alias=alias, task=task)
                continue

            t0 = time.monotonic()
            try:
                result = await fn(spec.engine)
                elapsed = (time.monotonic() - t0) * 1000
                self._record_success(alias, elapsed, fallback=(i > 0))
                if i > 0:
                    log.info("router.fallback_success", alias=alias,
                             task=task, attempt=i + 1)
                return result
            except Exception as exc:
                elapsed = (time.monotonic() - t0) * 1000
                self._record_failure(alias, elapsed)
                last_exc = exc
                remaining = len(effective_chain) - i - 1
                log.warning("router.engine_failed",
                            alias=alias, task=task, error=str(exc),
                            fallback_enabled=self._enable_fallback,
                            fallback_remaining=remaining)

        raise RuntimeError(
            f"Engine failed for task '{task}' "
            f"(fallback {'enabled' if self._enable_fallback else 'disabled'}). "
            f"Chain tried: {effective_chain}. Last error: {last_exc}"
        ) from last_exc

    async def _pick_healthy(self, chain: list[str]) -> ModelSpec:
        for alias in chain:
            spec = self._registry.get(alias)
            if spec and not self._circuits[alias].is_open():
                return spec
        raise RuntimeError(f"No healthy engine in chain: {chain}")

    def _record_success(self, alias: str, ms: float, fallback: bool) -> None:
        s = self._stats[alias]
        s.total    += 1
        s.success  += 1
        s.total_ms += ms
        if fallback:
            s.fallback += 1
        self._circuits[alias].record_success()

    def _record_failure(self, alias: str, ms: float) -> None:
        s = self._stats[alias]
        s.total    += 1
        s.errors   += 1
        s.total_ms += ms
        self._circuits[alias].record_failure()

    @staticmethod
    def _explicit_alias(config: AgentConfig) -> str | None:
        return getattr(config, "model_alias", None) or None


# ─────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────

def build_router_from_settings(settings=None) -> LLMRouter:
    """
    从 Settings 配置构建 LLMRouter。
    读取 ROUTER_* 环境变量定义多个引擎，TASK_* 变量定义任务路由。
    详见 .env.example 中的 [Multi-Model Router] 章节。
    """
    from utils.config import get_settings
    s = settings or get_settings()
    return s.build_router()


def single_engine_router(engine: Any, alias: str = "default") -> LLMRouter:
    """
    将单个引擎包装成 Router（兼容旧代码，零迁移成本）。
    所有任务类型都路由到同一个引擎。
    """
    registry = ModelRegistry()
    registry.register(alias, engine, supports_embed=True)
    task_router = TaskRouter(default=alias)
    return LLMRouter(registry, task_router)
