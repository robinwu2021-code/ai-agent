"""
core/container.py — 依赖注入容器 & Agent 工厂（完整版）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class AgentContainer:
    # ── 核心模块 ──────────────────────────────────────────────────
    # llm_router 是首选（支持多模型路由）
    # llm_engine 保留用于向后兼容：传入时自动包装成单引擎 Router
    llm_router:         Any = None   # LLMRouter
    llm_engine:         Any = None   # 向后兼容：单个 LLMEngine
    short_term_memory:  Any = None
    long_term_memory:   Any = None
    skill_registry:     Any = None
    mcp_hub:            Any = None
    context_manager:    Any = None
    orchestrator:       Any = None
    consolidator:       Any = None
    orchestrator_type:  str = "react"

    # ── 扩展模块（可选）─────────────────────────────────────────
    security_manager:   Any = None
    knowledge_base:     Any = None
    multimodal_proc:    Any = None
    hitl_manager:       Any = None
    cost_tracker:       Any = None
    quota_manager:      Any = None
    model_downgrader:   Any = None
    tenant_manager:     Any = None
    workspace_manager:  Any = None   # WorkspaceManager（多用户工作区）
    prompt_renderer:    Any = None
    feedback_store:     Any = None
    task_queue:         Any = None
    # ── 知识图谱 ──────────────────────────────────────────────────
    kg_store:           Any = None   # KGStore（SQLite / Neo4j）
    graph_builder:      Any = None   # GraphBuilder

    def _effective_router(self) -> Any:
        """返回 LLMRouter（若只设了 llm_engine，自动包装为单引擎 Router）。"""
        if self.llm_router is not None:
            return self.llm_router
        if self.llm_engine is not None:
            from llm.router import single_engine_router
            self.llm_router = single_engine_router(self.llm_engine)
            return self.llm_router
        raise RuntimeError("Container: neither llm_router nor llm_engine is set")

    # ── 多 Agent 编排专用字段 ─────────────────────────────────────
    agent_specs:        Any = None   # list[AgentSpec]，用于 multiagent 模式

    def build(self) -> "AgentContainer":
        from memory.consolidation import MemoryConsolidator
        from orchestrator.engines import ReactOrchestrator, PlanExecuteOrchestrator

        router = self._effective_router()

        if self.consolidator is None:
            self.consolidator = MemoryConsolidator(router, self.long_term_memory)

        if self.orchestrator is None:
            kwargs = dict(
                llm_engine=router,
                skill_registry=self.skill_registry,
                mcp_hub=self.mcp_hub,
                context_manager=self.context_manager,
                short_term_memory=self.short_term_memory,
                long_term_memory=self.long_term_memory,
            )
            if self.orchestrator_type == "plan_execute":
                self.orchestrator = PlanExecuteOrchestrator(**kwargs)
            elif self.orchestrator_type == "dag":
                from orchestrator.dag import DAGOrchestrator
                self.orchestrator = DAGOrchestrator(**kwargs)
            elif self.orchestrator_type == "multiagent":
                from multiagent.orchestrator import MultiAgentOrchestrator
                specs = self.agent_specs or []
                self.orchestrator = MultiAgentOrchestrator(
                    container_components=kwargs,
                    agent_specs=specs,
                )
            else:
                self.orchestrator = ReactOrchestrator(**kwargs)

        log.info("container.built", orchestrator=type(self.orchestrator).__name__)
        return self

    def agent(self) -> "Agent":
        return Agent(self)

    # ── Factory methods ───────────────────────────────────────────

    @classmethod
    def create_dev(cls, orchestrator_type: str = "react") -> "AgentContainer":
        from context.manager import PriorityContextManager
        from llm.engines import MockLLMEngine
        from memory.stores import InMemoryLongTermMemory, InMemoryShortTermMemory
        from mcp.hub import DefaultMCPHub
        from skills.registry import LocalSkillRegistry, PythonExecutorSkill, WebSearchSkill

        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(WebSearchSkill())

        c = cls(
            llm_engine=MockLLMEngine(),
            short_term_memory=InMemoryShortTermMemory(),
            long_term_memory=InMemoryLongTermMemory(),
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(),
            orchestrator_type=orchestrator_type,
        )
        return c.build()

    @classmethod
    def create_from_settings(cls, settings=None) -> "AgentContainer":
        """从 Settings 配置对象装配（推荐生产用法）。"""
        from utils.config import get_settings
        s = settings or get_settings()
        return s.build_container()

    @classmethod
    def create_with_router(cls, router, orchestrator_type: str = "react") -> "AgentContainer":
        """直接传入 LLMRouter 装配（适合自定义多模型路由场景）。"""
        from context.manager import PriorityContextManager
        from memory.stores import InMemoryLongTermMemory, InMemoryShortTermMemory
        from mcp.hub import DefaultMCPHub
        from skills.registry import LocalSkillRegistry, PythonExecutorSkill, WebSearchSkill

        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(WebSearchSkill())

        c = cls(
            llm_router=router,
            short_term_memory=InMemoryShortTermMemory(),
            long_term_memory=InMemoryLongTermMemory(),
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(),
            orchestrator_type=orchestrator_type,
        )
        return c.build()

    @classmethod
    def create_with_anthropic(
        cls,
        api_key: str | None = None,
        orchestrator_type: str = "react",
        redis_url: str | None = None,
        qdrant_url: str | None = None,
        enable_security: bool = True,
        enable_cost_tracking: bool = True,
        enable_prompt_mgr: bool = True,
    ) -> "AgentContainer":
        from context.manager import PriorityContextManager
        from llm.engines import AnthropicEngine
        from mcp.hub import DefaultMCPHub
        from skills.registry import (
            LocalSkillRegistry, PythonExecutorSkill,
            FileReadSkill, FileWriteSkill, WebSearchSkill,
        )

        _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        llm = AnthropicEngine(api_key=_api_key)

        if redis_url:
            from memory.stores import RedisShortTermMemory
            stm = RedisShortTermMemory(redis_url=redis_url)
        else:
            from memory.stores import InMemoryShortTermMemory
            stm = InMemoryShortTermMemory()

        if qdrant_url:
            from memory.stores import QdrantLongTermMemory
            ltm = QdrantLongTermMemory(url=qdrant_url, embed_fn=llm.embed)
        else:
            from memory.stores import InMemoryLongTermMemory
            ltm = InMemoryLongTermMemory()

        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(FileReadSkill())
        registry.register(FileWriteSkill())
        registry.register(WebSearchSkill())

        c = cls(
            llm_engine=llm,
            short_term_memory=stm,
            long_term_memory=ltm,
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(llm_engine=llm),
            orchestrator_type=orchestrator_type,
        )

        if enable_security:
            from security.middleware import SecurityManager
            c.security_manager = SecurityManager()

        if enable_cost_tracking:
            from utils.cost import CostTracker, QuotaManager, ModelDowngrader
            c.cost_tracker    = CostTracker()
            c.quota_manager   = QuotaManager(c.cost_tracker)
            c.model_downgrader = ModelDowngrader(c.quota_manager)

        if enable_prompt_mgr:
            from prompt_mgr.manager import PromptRegistry, ABTestRouter, PromptRenderer
            registry_pm = PromptRegistry()
            ab_router   = ABTestRouter(registry_pm)
            c.prompt_renderer = PromptRenderer(registry_pm, ab_router)

        from eval.framework import FeedbackStore
        c.feedback_store = FeedbackStore()

        from tenant.manager import TenantManager
        c.tenant_manager = TenantManager()

        from queue.scheduler import TaskQueue
        c.task_queue = TaskQueue()

        return c.build()


# ─────────────────────────────────────────────
# Agent — 顶层入口
# ─────────────────────────────────────────────

class Agent:
    def __init__(self, container: AgentContainer) -> None:
        self._c = container

    async def run(
        self,
        user_id: str,
        session_id: str,
        text: str,
        config: Any | None = None,
        files: list[dict] | None = None,
        tenant_id: str | None = None,
    ):
        from core.models import AgentConfig, AgentInput, AgentTask

        _config = config or AgentConfig()

        # ── 租户上下文解析 ────────────────────────────────────────
        c = self._c
        if c.tenant_manager and tenant_id:
            ctx = c.tenant_manager.get_or_default(tenant_id)
            user_id   = c.tenant_manager.resolve_user_id(tenant_id, user_id)
            _config   = _config.model_copy(update={
                "max_steps":   ctx.max_steps,
                "token_budget": ctx.token_budget,
            })

        # ── 成本/配额检查 ─────────────────────────────────────────
        if c.quota_manager:
            within, reason = c.quota_manager.check_daily_quota(user_id)
            if not within:
                yield {"type": "error", "message": f"Quota exceeded: {reason}"}
                return

        # ── 模型降级 ──────────────────────────────────────────────
        if c.model_downgrader and _config.model:
            _config = _config.model_copy(update={
                "model": c.model_downgrader.suggest_model(user_id, _config.model)
            })

        # ── 安全检查（输入） ──────────────────────────────────────
        if c.security_manager:
            safe, reason = await c.security_manager.check_request(user_id, text)
            if not safe:
                yield {"type": "error", "message": f"Security block: {reason}"}
                return

        # ── 多模态文件处理 ────────────────────────────────────────
        extra_context = ""
        if files and c.multimodal_proc:
            processed = await c.multimodal_proc.process_files(files)
            extra_context = c.multimodal_proc.to_context_text(processed)
            if extra_context:
                text = text + "\n\n" + extra_context

        # ── RAG 知识库检索 ────────────────────────────────────────
        if c.knowledge_base:
            chunks = await c.knowledge_base.query(text, top_k=3)
            if chunks:
                kb_ctx = c.knowledge_base.format_context(chunks)
                text   = text + "\n\n" + kb_ctx

        # ── 创建任务 ──────────────────────────────────────────────
        task = AgentTask(
            session_id=session_id,
            user_id=user_id,
            input=AgentInput(text=text, files=files or []),
        )

        existing = await c.short_term_memory.load_task(session_id)
        if existing:
            task.history = existing.history

        await c.short_term_memory.save_task(task)

        total_tokens = 0
        async for event in c.orchestrator.run(task, _config):
            # 输出安全过滤
            if event.get("type") == "delta" and c.security_manager:
                event = {**event, "text": c.security_manager.sanitize_output(event.get("text", ""))}
            # Token 计费
            if event.get("type") == "done" and c.cost_tracker:
                usage = event.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                c.cost_tracker.record(
                    user_id=user_id, session_id=session_id,
                    model=_config.model or "unknown",
                    prompt_tokens=usage.get("prompt_tokens", total_tokens // 2),
                    completion_tokens=usage.get("completion_tokens", total_tokens // 2),
                    task_id=task.id,
                )
            yield event

        # 任务结束后固化记忆
        if task.status.value in ("done", "error"):
            try:
                await c.consolidator.consolidate(task)
            except Exception as e:
                log.warning("consolidation.failed", error=str(e))
