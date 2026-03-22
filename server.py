"""
server.py — 轻量 HTTP 服务器

从 llm.yaml 读取模型配置，注册天气等 Skill，暴露以下接口：

    POST /chat          普通请求（等待完整回复后返回 JSON）
    POST /chat/stream   流式请求（SSE，逐 token 推送）
    GET  /skills        列出已注册的 Skill
    GET  /health        健康检查

启动方式：
    python server.py
    python server.py --host 0.0.0.0 --port 8080
    python server.py --engine ollama-qwen3   # 指定 llm.yaml 中的 alias
"""
from __future__ import annotations

import argparse
import json
import pathlib
import uuid
from typing import AsyncIterator

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────

app = FastAPI(
    title="AI Agent",
    version="1.0.0",
    description="基于 llm.yaml 配置，集成 Skill 与大模型的 HTTP 接口",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_container = None   # 全局容器，startup 时初始化


# ── 营销顾问系统提示词 ─────────────────────────────────────────────

_MARKETING_SYSTEM_PROMPT = """你是一位专业的营销顾问，专门服务于中小商家。

## 你的工作方式
- 通过友善、专业的对话，逐步了解商家的经营情况和营销需求
- 每次只问一个最关键的问题，避免让商家感到被"审问"
- 收集到足够信息后，调用 analyze_needs 评估完整度
- 完整度 >= 0.7 时，调用 suggest_activities 展示具体活动方案
- 商家确认偏好后，调用 create_plan 生成完整可执行方案

## 对话风格
- 亲切、接地气，像朋友聊天而非填表格
- 善用具体例子帮助商家理解方案
- 方案要实用、可落地，给出具体步骤和预期效果

## 重要提示
- 先问候商家，了解基本情况（经营类型和营销目标最重要）
- 不要一次性问很多问题
- 根据 analyze_needs 返回的 next_question 来决定下一个问题
- 展示活动建议时，用通俗易懂的语言说明每个活动的好处
"""

# ── 自定义上下文管理器（注入系统提示词前缀）────────────────────────

class _PrefixedContextManager:
    """包装 PriorityContextManager，将自定义系统提示词注入到 P0 消息前面。"""

    def __init__(self, base, prefix: str):
        self._base   = base
        self._prefix = prefix

    async def build(self, task, tools, long_term_memory, budget):
        from core.models import Message, MessageRole
        messages = await self._base.build(task, tools, long_term_memory, budget)
        # 在第一条 SYSTEM 消息前插入自定义前缀
        if messages and messages[0].role == MessageRole.SYSTEM:
            original = messages[0].content or ""
            messages[0] = Message(
                role=MessageRole.SYSTEM,
                content=f"{self._prefix}\n\n---\n\n{original}",
            )
        else:
            messages.insert(0, Message(role=MessageRole.SYSTEM, content=self._prefix))
        return messages


# ── Request / Response 模型 ────────────────────────────────────────

class ChatRequest(BaseModel):
    text:       str   = Field(..., description="用户输入文本")
    user_id:    str   = Field("default_user", description="用户 ID")
    session_id: str | None = Field(None, description="会话 ID，为空则自动生成")
    max_steps:  int   = Field(10, description="最大工具调用步数")
    skills:     list[str] | None = Field(
        None,
        description="限定本次请求可使用的 Skill 名称列表；为空则使用全部已注册 Skill",
    )
    mode:         str | None = Field(
        None,
        description="运行模式，\"marketing\" 启用营销顾问系统提示词",
    )
    system_prompt: str | None = Field(
        None,
        description="自定义系统提示词前缀；优先级高于 mode",
    )
    # ── Workspace 扩展 ─────────────────────────────────────────────
    workspace_id: str | None = Field(
        None,
        description="工作区 ID；设置后启用多层混合记忆和工作区配置",
    )
    project_id: str | None = Field(
        None,
        description="项目 ID（必须属于 workspace_id 下的项目）；设置后启用项目级记忆",
    )
    # ── Orchestrator 选择 ──────────────────────────────────────────
    orchestrator_type: str | None = Field(
        None,
        description=(
            "编排器类型：react（默认）| plan_execute | dag（DAG 并行任务拆分）"
            " | multiagent（多 Agent 协作）"
        ),
    )
    agent_specs: list[dict] | None = Field(
        None,
        description=(
            "多 Agent 规格列表（仅 orchestrator_type=multiagent 时生效）。"
            "每项格式：{name, description, skills, system_prompt, max_steps}"
        ),
    )


class ChatResponse(BaseModel):
    session_id: str
    reply:      str
    usage:      dict = {}
    steps:      list[dict] = []


# ── 容器初始化 ─────────────────────────────────────────────────────

def _build_container(engine_alias: str | None = None):
    """
    根据 llm.yaml 构建 AgentContainer。
    engine_alias：指定使用 llm.yaml 中某个引擎；为 None 时使用路由默认引擎。
    """
    from core.container import AgentContainer
    from context.manager import PriorityContextManager
    from memory.stores import InMemoryShortTermMemory, InMemoryLongTermMemory
    from mcp.hub import DefaultMCPHub
    from skills.registry import LocalSkillRegistry
    from skills.weather import WeatherCurrentSkill, WeatherForecastSkill
    from skills.builtins import CalculatorSkill, HttpRequestSkill
    from skills.registry import PythonExecutorSkill, WebSearchSkill
    from utils.llm_config import load_from_yaml
    from llm.router import LLMRouter, ModelRegistry, TaskRouter

    # ── 初始化 LLM 调用全链路日志 ─────────────────────────────────
    from utils.config import get_settings as _get_settings
    from llm.call_logger import init_call_logger as _init_call_logger
    _s = _get_settings()
    _init_call_logger(
        enabled      = _s.llm_call_log_enabled,
        log_level    = _s.llm_call_log_level,
        file_path    = _s.llm_call_log_file,
        max_bytes    = _s.llm_call_log_max_bytes,
        backup_count = _s.llm_call_log_backup_count,
        msg_preview  = _s.llm_call_log_msg_preview,
    )

    yaml_path = pathlib.Path(__file__).parent / "llm.yaml"
    configs, router_cfg = load_from_yaml(yaml_path)

    # 构建引擎注册表，跟踪成功加载的 alias
    registry       = ModelRegistry()
    loaded_aliases: list[str] = []
    for cfg in configs:
        try:
            engine = cfg.build_engine()
            registry.register(cfg.alias, engine, cost_tier=cfg.cost_tier)
            loaded_aliases.append(cfg.alias)
            log.info("server.engine_loaded", alias=cfg.alias, model=cfg.model)
        except Exception as exc:
            log.warning("server.engine_skip", alias=cfg.alias, reason=str(exc))

    if not loaded_aliases:
        raise RuntimeError("llm.yaml 中没有可用的引擎，请检查 SDK 安装和配置")

    # 确定默认 alias：优先使用命令行参数 > yaml router.default > 第一个成功加载的引擎
    preferred = engine_alias or router_cfg.default
    default_alias = preferred if preferred in loaded_aliases else loaded_aliases[0]
    if default_alias != preferred:
        log.warning("server.default_fallback",
                    preferred=preferred, actual=default_alias)

    def _resolve(alias: str | None) -> str:
        """若 alias 未加载则回退到 default_alias。"""
        return alias if alias in loaded_aliases else default_alias

    fallback = [a for a in (router_cfg.fallback or []) if a in loaded_aliases]

    task_router = TaskRouter(
        default     = default_alias,
        chat        = _resolve(router_cfg.chat),
        summarize   = _resolve(router_cfg.summarize),
        consolidate = _resolve(router_cfg.consolidate),
        fallback    = fallback,
    )
    llm_router = LLMRouter(registry=registry, task_router=task_router)

    # Skill 注册表：内置 Skill
    from skills.loader import SkillLoader
    skill_registry = LocalSkillRegistry()
    for builtin in [
        WeatherCurrentSkill(provider="wttr.in"),
        WeatherForecastSkill(provider="wttr.in"),
        CalculatorSkill(),
        HttpRequestSkill(),
    ]:
        skill_registry.register(builtin)
    for optional_cls in [PythonExecutorSkill, WebSearchSkill]:
        try:
            skill_registry.register(optional_cls())
        except Exception:
            pass

    # Hub Skill：自动发现并加载 skills/hub/ 下的所有 Skill 包
    for hub_skill in SkillLoader.load_all():
        try:
            skill_registry.register(hub_skill)
        except Exception as e:
            log.warning("server.hub_skill_skip", name=getattr(hub_skill, "descriptor", None) and hub_skill.descriptor.name, error=str(e))

    # ── Workspace 存储（SQLite，可通过 WORKSPACE_DB_URL 环境变量切换 MySQL）──
    import asyncio, os
    from workspace.store import create_store
    from workspace.manager import WorkspaceManager
    from workspace.memory import WorkspaceAwareLTM

    ws_db_url = os.environ.get("WORKSPACE_DB_URL", "sqlite:///workspace.db")
    ws_store  = create_store(ws_db_url)

    # 同步初始化（建表）——在进程启动时执行一次
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 若已在 async 上下文，用 run_in_executor 方式初始化
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, ws_store.initialize())
                future.result(timeout=10)
        else:
            loop.run_until_complete(ws_store.initialize())
    except Exception as exc:
        log.warning("server.workspace_store_init_failed", error=str(exc))

    workspace_ltm     = WorkspaceAwareLTM(ws_store)
    workspace_manager = WorkspaceManager(ws_store)

    # ── 知识图谱存储 & 构建器（与 workspace.db 同库）───────────────
    kg_store     = None
    graph_builder = None
    try:
        from rag.graph.store import SQLiteKGStore
        from rag.graph.extractor import TripleExtractor
        from rag.graph.resolver import EntityResolver
        from rag.graph.builder import GraphBuilder

        kg_db_url = os.environ.get("KG_STORE_URL", "sqlite:///workspace.db")
        kg_store  = SQLiteKGStore(kg_db_url.replace("sqlite:///", ""))
        try:
            loop2 = asyncio.get_event_loop()
            if loop2.is_running():
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, kg_store.initialize()).result(timeout=10)
            else:
                loop2.run_until_complete(kg_store.initialize())
        except Exception as exc:
            log.warning("server.kg_store_init_failed", error=str(exc))

        # embed_fn 从 llm_router 借用
        _embed_fn = lambda text: llm_router.embed(text)  # noqa: E731

        extractor     = TripleExtractor(llm_router)
        resolver      = EntityResolver(kg_store, _embed_fn)
        graph_builder = GraphBuilder(kg_store, extractor, resolver, _embed_fn)
        log.info("server.kg_ready")
    except Exception as exc:
        log.warning("server.kg_init_failed", error=str(exc))

    # ── 持久化知识库（与 workspace.db 同库）──────────────────────────
    pkb = None
    try:
        from rag.store import SQLiteKBStore
        from rag.persistent_kb import PersistentKnowledgeBase
        kb_store = SQLiteKBStore(ws_db_url.replace("sqlite:///", ""))
        # 同步初始化（建表）
        try:
            loop3 = asyncio.get_event_loop()
            if loop3.is_running():
                import concurrent.futures as _cf3
                with _cf3.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, kb_store.initialize()).result(timeout=10)
            else:
                loop3.run_until_complete(kb_store.initialize())
        except Exception as exc:
            log.warning("server.kb_store_init_failed", error=str(exc))
        _embed_fn2 = lambda text: llm_router.embed(text)  # noqa: E731
        pkb = PersistentKnowledgeBase(
            store=kb_store, embed_fn=_embed_fn2, kb_id="global"
        )
        # 启动时尝试恢复索引（若事件循环尚未运行则跳过，由首次查询时懒加载）
        try:
            loop4 = asyncio.get_event_loop()
            if not loop4.is_running():
                loop4.run_until_complete(pkb.initialize())
        except Exception:
            pass  # 在 FastAPI startup 事件中恢复
        log.info("server.pkb_ready")
    except Exception as exc:
        log.warning("server.pkb_init_failed", error=str(exc))

    container = AgentContainer(
        llm_router        = llm_router,
        short_term_memory = InMemoryShortTermMemory(),
        long_term_memory  = workspace_ltm,
        skill_registry    = skill_registry,
        mcp_hub           = DefaultMCPHub(),
        context_manager   = PriorityContextManager(),
        workspace_manager = workspace_manager,
        kg_store          = kg_store,
        graph_builder     = graph_builder,
        knowledge_base    = pkb,
    )
    return container.build()


# ── 核心：运行 Agent，收集事件 ─────────────────────────────────────

def _get_container(
    skills:             list[str] | None,
    system_prompt:      str | None = None,
    workspace_ctx:      Any        = None,   # WorkspaceContext | None
    orchestrator_type:  str | None = None,
    agent_specs:        list[dict] | None = None,
):
    """
    返回适合本次请求的容器副本。

    workspace_ctx 存在时：
      - 用工作区的 system_prompt（优先级：request.system_prompt > ws.system_prompt）
      - 用工作区的 allowed_skills（优先级：request.skills > ws.allowed_skills）
      - 使用 WorkspaceAwareLTM（若全局 LTM 支持）

    orchestrator_type 存在且与容器当前类型不同时，动态替换 orchestrator。
    """
    import dataclasses
    replacements: dict = {}

    # ── Skill 过滤 ─────────────────────────────────────────────────
    effective_skills = skills
    if not effective_skills and workspace_ctx and workspace_ctx.allowed_skills:
        effective_skills = workspace_ctx.allowed_skills

    if effective_skills:
        from skills.loader import FilteredSkillRegistry
        replacements["skill_registry"] = FilteredSkillRegistry(
            _container.skill_registry, set(effective_skills)
        )

    # ── System Prompt ──────────────────────────────────────────────
    effective_prompt = system_prompt
    if not effective_prompt and workspace_ctx and workspace_ctx.system_prompt:
        effective_prompt = workspace_ctx.system_prompt

    if effective_prompt:
        replacements["context_manager"] = _PrefixedContextManager(
            _container.context_manager, effective_prompt
        )

    # ── Orchestrator 动态替换 ──────────────────────────────────────
    otype = orchestrator_type
    if otype and otype != _container.orchestrator_type:
        sr = replacements.get("skill_registry", _container.skill_registry)
        kwargs = dict(
            llm_engine      = _container._effective_router(),
            skill_registry  = sr,
            mcp_hub         = _container.mcp_hub,
            context_manager = replacements.get("context_manager", _container.context_manager),
            short_term_memory = _container.short_term_memory,
            long_term_memory  = _container.long_term_memory,
        )
        if otype == "dag":
            from orchestrator.dag import DAGOrchestrator
            replacements["orchestrator"] = DAGOrchestrator(**kwargs)
        elif otype == "multiagent":
            from multiagent.orchestrator import MultiAgentOrchestrator, AgentSpec
            specs: list[AgentSpec] = []
            for s in (agent_specs or []):
                specs.append(AgentSpec(
                    name          = s.get("name", "agent"),
                    description   = s.get("description", ""),
                    skills        = s.get("skills", []),
                    system_prompt = s.get("system_prompt", ""),
                    max_steps     = s.get("max_steps", 8),
                ))
            replacements["orchestrator"] = MultiAgentOrchestrator(
                container_components=kwargs,
                agent_specs=specs,
            )
        elif otype == "plan_execute":
            from orchestrator.engines import PlanExecuteOrchestrator
            replacements["orchestrator"] = PlanExecuteOrchestrator(**kwargs)
        else:  # react
            from orchestrator.engines import ReactOrchestrator
            replacements["orchestrator"] = ReactOrchestrator(**kwargs)
        replacements["orchestrator_type"] = otype

    return dataclasses.replace(_container, **replacements) if replacements else _container


async def _resolve_workspace_ctx(req: ChatRequest) -> Any:
    """解析工作区上下文（若 workspace_id 存在）。"""
    if not req.workspace_id or not _container:
        return None
    wm = getattr(_container, "workspace_manager", None)
    if not wm:
        return None
    try:
        return await wm.get_context(
            user_id=req.user_id,
            workspace_id=req.workspace_id,
            project_id=req.project_id,
        )
    except Exception as exc:
        log.warning("server.workspace_ctx_failed", error=str(exc))
        return None


async def _run_agent(req: ChatRequest):
    """返回 (session_id, reply, usage, steps) 四元组。"""
    from core.models import AgentConfig

    prompt       = req.system_prompt or (_MARKETING_SYSTEM_PROMPT if req.mode == "marketing" else None)
    ws_ctx       = await _resolve_workspace_ctx(req)
    container    = _get_container(req.skills, prompt, ws_ctx,
                                  req.orchestrator_type, req.agent_specs)
    session_id   = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    max_steps    = req.max_steps
    if ws_ctx:
        max_steps = min(max_steps, ws_ctx.max_steps)
    config       = AgentConfig(stream=False, max_steps=max_steps)

    # 记录 Session 与工作区关联
    if ws_ctx and hasattr(_container, "workspace_manager") and _container.workspace_manager:
        try:
            store = _container.workspace_manager._store
            await store.save_session(
                session_id, req.user_id, req.workspace_id, req.project_id
            )
        except Exception:
            pass

    reply_parts: list[str] = []
    steps:       list[dict] = []
    usage:       dict       = {}

    async for ev in container.agent().run(
        user_id    = req.user_id,
        session_id = session_id,
        text       = req.text,
        config     = config,
    ):
        t = ev.get("type")
        if t == "delta":
            reply_parts.append(ev.get("text", ""))
        elif t == "step":
            steps.append({
                "tool":   ev.get("tool"),
                "status": ev.get("status"),
                "error":  ev.get("error"),
            })
        elif t == "done":
            usage = ev.get("usage", {})
            if ev.get("status") not in ("done", None):
                raise HTTPException(
                    status_code=500,
                    detail=f"Agent 执行异常: {ev.get('status')}",
                )

    return session_id, "".join(reply_parts), usage, steps


async def _stream_agent(req: ChatRequest) -> AsyncIterator[str]:
    """生成 SSE 事件流。"""
    from core.models import AgentConfig

    prompt     = req.system_prompt or (_MARKETING_SYSTEM_PROMPT if req.mode == "marketing" else None)
    ws_ctx     = await _resolve_workspace_ctx(req)
    container  = _get_container(req.skills, prompt, ws_ctx,
                                req.orchestrator_type, req.agent_specs)
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    max_steps  = req.max_steps
    if ws_ctx:
        max_steps = min(max_steps, ws_ctx.max_steps)
    config     = AgentConfig(stream=True, max_steps=max_steps)

    # 在 SSE 流开头推送 workspace 信息
    meta: dict = {"type": "session", "session_id": session_id}
    if ws_ctx:
        meta["workspace_id"] = ws_ctx.workspace.workspace_id
        meta["project_id"]   = ws_ctx.project.project_id if ws_ctx.project else None
    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    async for ev in container.agent().run(
        user_id    = req.user_id,
        session_id = session_id,
        text       = req.text,
        config     = config,
    ):
        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


# ── HTTP 端点 ──────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, summary="发送消息并等待完整回复")
async def chat(req: ChatRequest):
    """
    发送一条消息，等待 Agent 调用 Skill 与大模型后返回完整回复。

    示例：
    ```json
    { "text": "今天成都的天气怎么样？" }
    ```
    """
    if _container is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    session_id, reply, usage, steps = await _run_agent(req)
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        usage=usage,
        steps=steps,
    )


@app.get("/chat", response_model=ChatResponse, summary="发送消息并等待完整回复（GET）")
async def chat_get(
    text:       str,
    user_id:    str   = "default_user",
    session_id: str | None = None,
    max_steps:  int   = 10,
):
    """
    GET 版本的 /chat，通过 Query 参数传入文本。

    示例：
    ```
    GET /chat?text=今天成都的天气怎么样？
    ```
    """
    if _container is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    req = ChatRequest(text=text, user_id=user_id,
                      session_id=session_id, max_steps=max_steps)
    sid, reply, usage, steps = await _run_agent(req)
    return ChatResponse(session_id=sid, reply=reply, usage=usage, steps=steps)


@app.post("/chat/stream", summary="发送消息并以 SSE 流式接收回复")
async def chat_stream(req: ChatRequest):
    """
    发送一条消息，以 Server-Sent Events 流式推送 Agent 的思考过程与最终回复。

    事件格式（每行 `data: <JSON>`）：
    - `{"type":"session","session_id":"..."}` — 会话 ID
    - `{"type":"thinking","step":1}` — 正在推理
    - `{"type":"step","tool":"weather_current","status":"running"}` — 工具调用中
    - `{"type":"step","tool":"weather_current","status":"done"}` — 工具完成
    - `{"type":"delta","text":"..."}` — 回复文本片段
    - `{"type":"done","status":"done","usage":{...}}` — 全部完成
    - `{"type":"planning"}` — DAG 编排器正在规划
    - `{"type":"plan","steps":[...]}` — DAG 规划结果
    - `{"type":"parallel_start","step_ids":[...],"goals":[...]}` — 并行步骤启动
    - `{"type":"step_start","step_id":"...","goal":"..."}` — 单步启动
    - `{"type":"step_done","step_id":"...","result":"..."}` — 单步完成
    - `{"type":"step_failed","step_id":"...","error":"..."}` — 单步失败
    - `{"type":"orchestrating","agents":[...]}` — 多 Agent 编排开始
    - `{"type":"subtask_assign","agent":"...","goal":"...","depends":[...]}` — 子任务分配
    - `{"type":"agent_start","agent":"...","subtask_id":"..."}` — 子 Agent 启动
    - `{"type":"agent_done","agent":"...","subtask_id":"...","tokens":N}` — 子 Agent 完成
    - `{"type":"agent_error","agent":"...","error":"..."}` — 子 Agent 失败
    """
    if _container is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    return StreamingResponse(
        _stream_agent(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/skills", summary="列出已注册的 Skill")
async def list_skills():
    """返回当前已注册的所有 Skill 列表（含 hub 扩展 Skill）。"""
    if _container is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    from skills.loader import SkillLoader, HUB_DIR
    hub_names = set()
    for m in SkillLoader.manifests():
        hub_names.add(m.get("name", ""))
    return {
        "skills": [
            {
                "name":        d.name,
                "description": d.description,
                "tags":        d.tags,
                "source":      "hub" if d.name in hub_names else "builtin",
            }
            for d in _container.skill_registry.list_descriptors()
        ]
    }


# ══════════════════════════════════════════════════════════════════════
# Workspace REST API
# ══════════════════════════════════════════════════════════════════════

from pydantic import BaseModel as _BM

class _CreateWorkspaceReq(_BM):
    name:           str
    # 同时接受 creator_id（后端标准）和 created_by（前端惯例），二者皆可
    creator_id:     str   = ""
    created_by:     str   = ""   # alias for creator_id
    description:    str   = ""
    system_prompt:  str   = ""
    allowed_skills: list[str] = []
    token_budget:   int   = 16_000
    max_steps:      int   = 20

    @property
    def resolved_creator(self) -> str:
        return self.creator_id or self.created_by or "anonymous"

class _CreateProjectReq(_BM):
    name:           str
    creator_id:     str   = ""
    created_by:     str   = ""   # alias for creator_id
    description:    str   = ""
    system_prompt:  str   = ""
    allowed_skills: list[str] = []

    @property
    def resolved_creator(self) -> str:
        return self.creator_id or self.created_by or "anonymous"
    token_budget:   int   = 12_000
    max_steps:      int   = 20

class _AddMemberReq(_BM):
    operator_id: str
    user_id:     str
    role:        str = "member"   # admin | member | viewer

class _ShareMemoryReq(_BM):
    entry_id:         str
    to_project_ids:   list[str]
    shared_by:        str
    permission:       str = "read"
    note:             str = ""


def _wm():
    """获取 WorkspaceManager，未初始化时抛 503。"""
    if _container is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    wm = getattr(_container, "workspace_manager", None)
    if not wm:
        raise HTTPException(status_code=501, detail="工作区功能未启用（workspace_manager 未配置）")
    return wm


@app.post("/workspaces", summary="创建工作区", tags=["Workspace"])
async def create_workspace(req: _CreateWorkspaceReq):
    ws = await _wm().create_workspace(
        name=req.name, creator_id=req.resolved_creator,
        description=req.description, system_prompt=req.system_prompt,
        allowed_skills=req.allowed_skills,
        token_budget=req.token_budget, max_steps=req.max_steps,
    )
    return {
        "workspace_id": ws.workspace_id,
        "name":         ws.name,
        "description":  ws.description,
        "member_count": len(ws.members),
        "project_count": len(ws.projects),
    }


@app.get("/workspaces", summary="列出工作区", tags=["Workspace"])
async def list_workspaces(user_id: str | None = None):
    workspaces = await _wm().list_workspaces(user_id)
    return {"workspaces": [
        {
            "workspace_id":  w.workspace_id,
            "name":          w.name,
            "description":   w.description,
            "member_count":  len(w.members),
            "project_count": len(w.projects),
        }
        for w in workspaces
    ]}


@app.get("/workspaces/{workspace_id}", summary="获取工作区详情", tags=["Workspace"])
async def get_workspace(workspace_id: str):
    ws = await _wm().get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="工作区不存在")
    return {
        "workspace_id": ws.workspace_id, "name": ws.name,
        "description": ws.description,
        "system_prompt": ws.system_prompt,
        "allowed_skills": ws.allowed_skills,
        "token_budget": ws.token_budget, "max_steps": ws.max_steps,
        "members": [{"user_id": m.user_id, "role": m.role.value} for m in ws.members],
        "projects": [{"project_id": p.project_id, "name": p.name} for p in ws.projects],
    }


@app.post("/workspaces/{workspace_id}/members", summary="添加工作区成员", tags=["Workspace"])
async def add_workspace_member(workspace_id: str, req: _AddMemberReq):
    from workspace.models import WorkspaceRole
    await _wm().add_member(
        workspace_id=workspace_id, operator_id=req.operator_id,
        user_id=req.user_id, role=WorkspaceRole(req.role),
    )
    return {"ok": True}


@app.delete("/workspaces/{workspace_id}/members/{user_id}", summary="移除工作区成员", tags=["Workspace"])
async def remove_workspace_member(workspace_id: str, user_id: str, operator_id: str):
    await _wm().remove_member(workspace_id, operator_id, user_id)
    return {"ok": True}


@app.post("/workspaces/{workspace_id}/projects", summary="创建项目", tags=["Workspace"])
async def create_project(workspace_id: str, req: _CreateProjectReq):
    proj = await _wm().create_project(
        workspace_id=workspace_id, creator_id=req.resolved_creator,
        name=req.name, description=req.description,
        system_prompt=req.system_prompt, allowed_skills=req.allowed_skills,
        token_budget=req.token_budget, max_steps=req.max_steps,
    )
    return {
        "project_id":   proj.project_id,
        "workspace_id": proj.workspace_id,
        "name":         proj.name,
        "description":  proj.description,
        "member_count": len(proj.members),
    }


@app.get("/workspaces/{workspace_id}/projects", summary="列出项目", tags=["Workspace"])
async def list_projects(workspace_id: str, user_id: str | None = None):
    projects = await _wm().list_projects(workspace_id, user_id)
    return {"projects": [
        {
            "project_id":   p.project_id,
            "workspace_id": p.workspace_id,
            "name":         p.name,
            "description":  p.description,
            "member_count": len(p.members),
        }
        for p in projects
    ]}


@app.get("/workspaces/{workspace_id}/projects/{project_id}", summary="获取项目详情", tags=["Workspace"])
async def get_project(workspace_id: str, project_id: str):
    proj = await _wm().get_project(project_id)
    if not proj or proj.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="项目不存在")
    return {
        "project_id": proj.project_id, "workspace_id": proj.workspace_id,
        "name": proj.name, "description": proj.description,
        "system_prompt": proj.system_prompt,
        "allowed_skills": proj.allowed_skills,
        "token_budget": proj.token_budget, "max_steps": proj.max_steps,
        "members": [{"user_id": m.user_id, "role": m.role.value} for m in proj.members],
    }


@app.post("/workspaces/{workspace_id}/projects/{project_id}/members",
          summary="添加项目成员", tags=["Workspace"])
async def add_project_member(workspace_id: str, project_id: str, req: _AddMemberReq):
    from workspace.models import ProjectRole
    await _wm().add_project_member(
        project_id=project_id, workspace_id=workspace_id,
        operator_id=req.operator_id, user_id=req.user_id,
        role=ProjectRole(req.role),
    )
    return {"ok": True}


@app.post("/workspaces/memory/share", summary="跨项目共享记忆", tags=["Workspace"])
async def share_memory(req: _ShareMemoryReq):
    wm = _wm()
    if not hasattr(wm, "_store"):
        raise HTTPException(status_code=501, detail="WorkspaceAwareLTM 未启用")
    ltm = getattr(_container, "long_term_memory", None)
    if not ltm or not hasattr(ltm, "share_to_projects"):
        raise HTTPException(status_code=501, detail="WorkspaceAwareLTM 未启用")
    shares = await ltm.share_to_projects(
        entry_id=req.entry_id,
        to_project_ids=req.to_project_ids,
        shared_by=req.shared_by,
        permission=req.permission,
        note=req.note,
    )
    return {"shared": len(shares), "entry_id": req.entry_id}


# ══════════════════════════════════════════════════════════════════════
# Knowledge Base REST API  /kb/*
# ══════════════════════════════════════════════════════════════════════

from fastapi import UploadFile, File as _File, BackgroundTasks
import tempfile, os as _os

class _KBAskReq(_BM):
    query:      str
    kb_id:      str  = "global"
    top_k:      int  = 5
    mode:       str  = "hybrid"   # hybrid | vector | bm25
    with_graph: bool = False       # also run graph search and merge

class _KBAddTextReq(_BM):
    text:     str
    kb_id:    str = "global"
    source:   str = "inline"
    filename: str = ""


def _pkb():
    """获取 PersistentKnowledgeBase，未初始化时抛 503 / 501。"""
    if _container is None:
        raise HTTPException(503, "服务未初始化")
    pkb = getattr(_container, "knowledge_base", None)
    if not pkb or not hasattr(pkb, "list_documents"):
        raise HTTPException(501, "持久化知识库未启用")
    return pkb


@app.post("/kb/documents/upload", summary="上传文件到知识库", tags=["KnowledgeBase"])
async def kb_upload_file(
    file:  UploadFile = _File(...),
    kb_id: str        = "global",
):
    """
    上传文档（PDF / TXT / MD / DOCX / HTML）到知识库，后台同步分块 + 向量化。
    返回 KBDocument 元数据（含 doc_id、状态等）。
    """
    pkb = _pkb()
    allowed_suffixes = {".pdf", ".txt", ".md", ".docx", ".html"}
    suffix = _os.path.splitext(file.filename or "doc.txt")[1].lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(400, f"不支持的文件类型: {suffix}，支持: {allowed_suffixes}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        doc = await pkb.add_file(tmp_path, kb_id=kb_id)
    finally:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass

    import dataclasses
    return {**dataclasses.asdict(doc), "filename": file.filename or doc.filename}


@app.post("/kb/documents/text", summary="直接添加文本到知识库", tags=["KnowledgeBase"])
async def kb_add_text(req: _KBAddTextReq):
    """将纯文本内容直接入库（无需上传文件），同步分块 + 向量化后返回文档元数据。"""
    pkb  = _pkb()
    import dataclasses
    doc  = await pkb.add_text(
        text=req.text, source=req.source, kb_id=req.kb_id, filename=req.filename
    )
    return dataclasses.asdict(doc)


@app.get("/kb/documents", summary="列出知识库文档", tags=["KnowledgeBase"])
async def kb_list_documents(kb_id: str = "global"):
    """返回指定 kb_id 下所有文档的元数据列表（含状态、分块数等）。"""
    pkb  = _pkb()
    import dataclasses
    docs = await pkb.list_documents(kb_id=kb_id)
    return {"documents": [dataclasses.asdict(d) for d in docs], "total": len(docs)}


@app.get("/kb/documents/{doc_id}", summary="获取文档详情", tags=["KnowledgeBase"])
async def kb_get_document(doc_id: str):
    """获取单个文档的元数据及分块数量。"""
    pkb  = _pkb()
    store = pkb._store
    import dataclasses
    doc = await store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    chunks = await store.list_chunks(doc_id)
    result = dataclasses.asdict(doc)
    result["chunk_count"] = len(chunks)
    return result


@app.delete("/kb/documents/{doc_id}", summary="删除知识库文档", tags=["KnowledgeBase"])
async def kb_delete_document(doc_id: str):
    """删除文档及其所有分块（同时从内存索引中移除）。"""
    pkb = _pkb()
    await pkb.delete_document(doc_id)
    return {"ok": True, "doc_id": doc_id}


@app.post("/kb/ask", summary="知识库 RAG 问答", tags=["KnowledgeBase"])
async def kb_ask(req: _KBAskReq):
    """
    基于知识库的检索增强问答（RAG）。

    返回字段：
    - chunks:    命中的分块列表
    - context:   格式化后可注入 Prompt 的上下文文本
    - citations: 引用信息列表，每项含 index/source/filename/text_preview
    """
    pkb = _pkb()

    # Vector / hybrid retrieval
    kb_chunks = await pkb.query(req.query, kb_id=req.kb_id, top_k=req.top_k)

    # Optional: merge graph search results
    if req.with_graph:
        try:
            store = getattr(_container, "kg_store", None)
            if store:
                router   = _container._effective_router()
                embed_fn = lambda t: router.embed(t)  # noqa: E731
                from rag.graph.retriever import GraphRetriever
                gr       = GraphRetriever(store=store, embed_fn=embed_fn, llm_engine=router)
                subgraph = await gr.local_search(req.query, req.kb_id, hops=2)
                if subgraph.context_text:
                    from rag.store import KBChunk as _KBChunk
                    graph_chunk = _KBChunk(
                        chunk_id    = "kg_context",
                        doc_id      = "knowledge_graph",
                        kb_id       = req.kb_id,
                        chunk_index = 0,
                        text        = subgraph.context_text,
                        meta        = {"source": "knowledge_graph", "filename": "知识图谱"},
                        created_at  = 0.0,
                    )
                    kb_chunks = [graph_chunk] + kb_chunks
        except Exception as kg_exc:
            log.warning("kb_ask.kg_merge_failed", error=str(kg_exc))

    context    = pkb.format_context(kb_chunks)
    citations  = [
        {
            "index":        i + 1,
            "source":       c.meta.get("source", c.doc_id),
            "filename":     c.meta.get("filename", c.doc_id),
            "text_preview": c.text[:200],
        }
        for i, c in enumerate(kb_chunks)
    ]
    import dataclasses
    return {
        "chunks":    [dataclasses.asdict(c) for c in kb_chunks],
        "context":   context,
        "citations": citations,
    }


@app.get("/kb/stats", summary="知识库统计", tags=["KnowledgeBase"])
async def kb_stats(kb_id: str = "global"):
    """返回知识库的文档数、分块数、就绪文档数、总字符数等统计信息。"""
    pkb   = _pkb()
    stats = await pkb.get_stats(kb_id=kb_id)
    return {"kb_id": kb_id, **stats}


@app.post("/kb/build-graph/{doc_id}", summary="为 KB 文档触发图谱构建", tags=["KnowledgeBase"])
async def kb_build_graph(doc_id: str, kb_id: str = "global"):
    """
    从知识库文档的分块数据触发知识图谱构建任务。
    需要 GraphBuilder 已初始化（启用了 KG 功能）。
    """
    pkb = _pkb()
    gb  = getattr(_container, "graph_builder", None)
    if not gb:
        raise HTTPException(501, "GraphBuilder 未配置")

    store  = pkb._store
    chunks = await store.list_chunks(doc_id)
    if not chunks:
        raise HTTPException(404, f"文档 {doc_id} 不存在或没有分块数据")

    chunk_dicts = [
        {"text": c.text, "doc_id": c.doc_id, "chunk_id": c.chunk_id}
        for c in chunks
    ]
    job_id = gb.start_build_job(chunks=chunk_dicts, kb_id=kb_id, doc_id=doc_id)
    return {"job_id": job_id, "doc_id": doc_id, "chunks": len(chunk_dicts)}


# ══════════════════════════════════════════════════════════════════════
# Knowledge Graph REST API  /kg/*
# ══════════════════════════════════════════════════════════════════════

class _KGBuildTextReq(_BM):
    text:   str
    kb_id:  str  = "global"
    doc_id: str  = ""
    source: str  = "inline"

class _KGQueryReq(_BM):
    query:  str
    kb_id:  str  = "global"
    mode:   str  = "local"   # local | global | path
    hops:   int  = 2

class _KGSubgraphReq(_BM):
    node_id: str
    kb_id:   str = "global"
    hops:    int = 2

class _KGPathReq(_BM):
    query:   str
    kb_id:   str = "global"

class _KGEntityPatchReq(_BM):
    name:        str | None = None
    description: str | None = None
    node_type:   str | None = None

class _KGAddEdgeReq(_BM):
    kb_id:    str
    src_name: str
    relation: str
    dst_name: str
    context:  str = ""
    weight:   float = 1.0


def _kg():
    """获取 kg_store，未初始化时抛 503。"""
    if _container is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    store = getattr(_container, "kg_store", None)
    if not store:
        raise HTTPException(status_code=501, detail="知识图谱功能未启用（kg_store 未配置）")
    return store

def _gb():
    """获取 graph_builder，未初始化时抛 503。"""
    if _container is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    gb = getattr(_container, "graph_builder", None)
    if not gb:
        raise HTTPException(status_code=501, detail="GraphBuilder 未配置")
    return gb

def _gr():
    """懒建 GraphRetriever（按需）。"""
    store = _kg()
    router = _container._effective_router()
    embed_fn = lambda t: router.embed(t)  # noqa: E731
    from rag.graph.retriever import GraphRetriever
    return GraphRetriever(store=store, embed_fn=embed_fn, llm_engine=router)


# ── 文档入库 / 构建 ────────────────────────────────────────────────

@app.post("/kg/build/text", summary="从文本构建图谱", tags=["KnowledgeGraph"])
async def kg_build_text(req: _KGBuildTextReq, background_tasks: BackgroundTasks):
    """将纯文本内容异步构建为知识图谱，返回 job_id 供查询进度。"""
    gb = _gb()
    import uuid as _uuid
    doc_id = req.doc_id or f"doc_{_uuid.uuid4().hex[:8]}"
    job_id = gb.start_build_text_job(
        text=req.text, kb_id=req.kb_id,
        doc_id=doc_id, source=req.source,
    )
    return {"job_id": job_id, "doc_id": doc_id, "kb_id": req.kb_id}


@app.post("/kg/build/file", summary="上传文件并构建图谱", tags=["KnowledgeGraph"])
async def kg_build_file(
    kb_id: str = "global",
    file: UploadFile = _File(...),
):
    """上传文档（PDF/TXT/MD/DOCX/HTML），后台异步提取知识图谱。"""
    gb = _gb()
    suffix = _os.path.splitext(file.filename or "doc.txt")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        from rag.knowledge_base import DocumentIngester, TextChunker
        ingester = DocumentIngester()
        chunker  = TextChunker()
        doc      = ingester.ingest_file(tmp_path)
        doc.metadata["original_filename"] = file.filename
        chunks   = chunker.chunk(doc)
    finally:
        _os.unlink(tmp_path)

    chunk_dicts = [
        {"text": c.text, "doc_id": doc.id, "chunk_id": c.id}
        for c in chunks
    ]
    job_id = gb.start_build_job(
        chunks=chunk_dicts,
        kb_id=kb_id, doc_id=doc.id,
    )
    return {
        "job_id": job_id, "doc_id": doc.id,
        "kb_id": kb_id, "chunks": len(chunks),
        "filename": file.filename,
    }


@app.get("/kg/build/status/{job_id}", summary="查询构建任务状态", tags=["KnowledgeGraph"])
async def kg_build_status(job_id: str):
    """查询后台图谱构建任务的进度和结果。"""
    gb = _gb()
    status = gb.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return status


@app.get("/kg/build/jobs", summary="列出所有构建任务", tags=["KnowledgeGraph"])
async def kg_build_jobs():
    """列出所有图谱构建任务（含状态）。"""
    gb = _gb()
    return {"jobs": gb.list_jobs()}


# ── 社区摘要 ────────────────────────────────────────────────────────

@app.post("/kg/communities/rebuild", summary="重建社区摘要", tags=["KnowledgeGraph"])
async def kg_rebuild_communities(kb_id: str = "global"):
    """对指定知识库重新执行社区检测并用 LLM 生成摘要（后台任务）。"""
    gb = _gb()
    router = _container._effective_router()
    embed_fn = lambda t: router.embed(t)  # noqa: E731
    import asyncio as _aio
    _aio.create_task(_async_rebuild(gb, kb_id, router))
    return {"ok": True, "kb_id": kb_id, "message": "社区摘要重建已启动"}

async def _async_rebuild(gb, kb_id, llm_engine):
    try:
        result = await gb.rebuild_communities(kb_id=kb_id, llm_engine=llm_engine)
        log.info("kg.communities_rebuilt", **result)
    except Exception as exc:
        log.error("kg.communities_rebuild_failed", error=str(exc))


# ── 图谱查询 ────────────────────────────────────────────────────────

@app.post("/kg/query", summary="知识图谱问答", tags=["KnowledgeGraph"])
async def kg_query(req: _KGQueryReq):
    """
    基于知识图谱的问答接口，返回答案上下文 + 实体 + 关系 + 推理链。

    mode:
    - local  — 实体链接 + 子图展开（默认，适合具体问题）
    - global — 社区摘要召回（适合宏观概括性问题）
    - path   — 两实体路径推理（适合"A和B有什么关系"）
    """
    gr = _gr()
    if req.mode == "global":
        subgraph = await gr.global_search(req.query, req.kb_id)
    elif req.mode == "path":
        subgraph = await gr.path_search(req.query, req.kb_id)
    else:
        subgraph = await gr.local_search(req.query, req.kb_id, hops=req.hops)

    from rag.graph.retriever import GraphRetriever
    context = GraphRetriever.format_for_prompt(subgraph)
    return {
        "context":        context,
        "subgraph":       subgraph.to_dict(),
        "mode":           req.mode,
        "nodes_found":    len(subgraph.nodes),
        "edges_found":    len(subgraph.edges),
    }


@app.post("/kg/search/entities", summary="实体语义搜索", tags=["KnowledgeGraph"])
async def kg_search_entities(query: str, kb_id: str = "global", limit: int = 20):
    """在知识图谱中搜索与查询相关的实体节点（文本 + 向量双路）。"""
    store = _kg()
    router = _container._effective_router()

    text_hits = await store.search_nodes_by_text(query, kb_id, limit=limit)
    try:
        emb  = await router.embed(query)
        vec_hits = await store.search_nodes_by_embedding(emb, kb_id, limit=limit)
    except Exception:
        vec_hits = []

    # 合并去重
    seen: set[str] = set()
    merged = []
    for n in text_hits + vec_hits:
        if n.id not in seen:
            seen.add(n.id)
            merged.append(n)

    return {"entities": [
        {"id": n.id, "name": n.name, "type": n.node_type.value,
         "description": n.description, "degree": n.degree}
        for n in merged[:limit]
    ]}


@app.post("/kg/subgraph", summary="获取实体子图", tags=["KnowledgeGraph"])
async def kg_subgraph(req: _KGSubgraphReq):
    """获取指定实体节点的 N 跳邻居子图（用于前端可视化）。"""
    store = _kg()
    nodes, edges = await store.get_neighbors(req.node_id, req.kb_id, hops=req.hops)
    center = await store.get_node(req.node_id)
    if center and center not in nodes:
        nodes.insert(0, center)
    return {
        "nodes": [{"id": n.id, "name": n.name, "type": n.node_type.value,
                   "description": n.description, "degree": n.degree} for n in nodes],
        "edges": [{"id": e.id, "src_id": e.src_id, "dst_id": e.dst_id,
                   "relation": e.relation, "weight": e.weight, "context": e.context}
                  for e in edges],
    }


@app.post("/kg/path", summary="两实体路径查询", tags=["KnowledgeGraph"])
async def kg_path(req: _KGPathReq):
    """在图谱中查找两个实体间的最短路径，返回推理链。"""
    gr = _gr()
    subgraph = await gr.path_search(req.query, req.kb_id)
    return {
        "subgraph":        subgraph.to_dict(),
        "reasoning_chain": subgraph.reasoning_chain,
        "context":         subgraph.context_text,
    }


# ── 实体 / 边 CRUD ─────────────────────────────────────────────────

@app.get("/kg/entities", summary="列出实体", tags=["KnowledgeGraph"])
async def kg_list_entities(
    kb_id:     str        = "global",
    node_type: str | None = None,
    limit:     int        = 100,
    offset:    int        = 0,
):
    """列出知识图谱中的实体节点，支持类型过滤和分页。"""
    store = _kg()
    from rag.graph.models import NodeType
    nt = NodeType(node_type) if node_type else None
    nodes = await store.list_nodes(kb_id, node_type=nt, limit=limit, offset=offset)
    return {"entities": [
        {"id": n.id, "name": n.name, "type": n.node_type.value,
         "description": n.description, "degree": n.degree, "aliases": n.aliases}
        for n in nodes
    ], "total": len(nodes)}


@app.get("/kg/entities/{entity_id}", summary="实体详情", tags=["KnowledgeGraph"])
async def kg_get_entity(entity_id: str):
    """获取单个实体详情，含所有关联边。"""
    store = _kg()
    node = await store.get_node(entity_id)
    if not node:
        raise HTTPException(status_code=404, detail="实体不存在")
    edges = await store.list_edges(node.kb_id, src_id=entity_id)
    edges += await store.list_edges(node.kb_id, dst_id=entity_id)
    return {
        "id": node.id, "name": node.name, "type": node.node_type.value,
        "description": node.description, "aliases": node.aliases,
        "doc_ids": node.doc_ids, "degree": node.degree,
        "edges": [
            {"id": e.id, "src_id": e.src_id, "dst_id": e.dst_id,
             "relation": e.relation, "weight": e.weight, "context": e.context}
            for e in edges
        ],
    }


@app.patch("/kg/entities/{entity_id}", summary="修正实体", tags=["KnowledgeGraph"])
async def kg_patch_entity(entity_id: str, req: _KGEntityPatchReq):
    """人工修正实体名称、描述或类型。"""
    store = _kg()
    node = await store.get_node(entity_id)
    if not node:
        raise HTTPException(status_code=404, detail="实体不存在")
    from rag.graph.models import NodeType
    if req.name:        node.name = req.name
    if req.description: node.description = req.description
    if req.node_type:   node.node_type = NodeType(req.node_type)
    await store.upsert_node(node)
    return {"ok": True, "id": node.id, "name": node.name}


@app.delete("/kg/entities/{entity_id}", summary="删除实体", tags=["KnowledgeGraph"])
async def kg_delete_entity(entity_id: str):
    """删除实体及其所有关联边。"""
    store = _kg()
    await store.delete_node(entity_id)
    return {"ok": True}


@app.post("/kg/edges", summary="手动添加关系边", tags=["KnowledgeGraph"])
async def kg_add_edge(req: _KGAddEdgeReq):
    """手动在两个实体之间添加关系边（实体不存在时自动创建）。"""
    store = _kg()
    import uuid as _uuid, time as _time
    from rag.graph.models import Node, Edge, NodeType

    async def _get_or_create(name: str) -> Node:
        n = await store.get_node_by_name(name, req.kb_id)
        if not n:
            n = Node(
                id=f"n_{_uuid.uuid4().hex[:12]}",
                kb_id=req.kb_id, name=name,
                node_type=NodeType.OTHER,
                created_at=_time.time(),
            )
            await store.upsert_node(n)
        return n

    src_node = await _get_or_create(req.src_name)
    dst_node = await _get_or_create(req.dst_name)
    edge = Edge(
        id=f"e_{_uuid.uuid4().hex[:12]}",
        kb_id=req.kb_id,
        src_id=src_node.id,
        dst_id=dst_node.id,
        relation=req.relation,
        weight=req.weight,
        context=req.context,
        created_at=_time.time(),
    )
    await store.upsert_edge(edge)
    return {
        "ok": True, "edge_id": edge.id,
        "src": src_node.name, "dst": dst_node.name, "relation": req.relation,
    }


@app.delete("/kg/edges/{edge_id}", summary="删除关系边", tags=["KnowledgeGraph"])
async def kg_delete_edge(edge_id: str, kb_id: str = "global"):
    """删除指定关系边。"""
    store = _kg()
    await store.delete_edge(edge_id)
    return {"ok": True}


# ── 可视化数据 ─────────────────────────────────────────────────────

@app.get("/kg/graph", summary="获取完整图谱数据", tags=["KnowledgeGraph"])
async def kg_full_graph(kb_id: str = "global", limit: int = 500):
    """
    获取完整图谱的节点和边（用于前端可视化）。
    limit 控制最大节点数（按 degree 降序取前 N 个）。
    """
    store = _kg()
    nodes, edges = await store.get_full_graph(kb_id, limit=limit)
    node_ids = {n.id for n in nodes}
    # 过滤掉引用了不在节点集中的悬空边
    edges = [e for e in edges if e.src_id in node_ids and e.dst_id in node_ids]
    return {
        "nodes": [
            {"id": n.id, "name": n.name, "type": n.node_type.value,
             "description": n.description, "degree": n.degree}
            for n in nodes
        ],
        "edges": [
            {"id": e.id, "src_id": e.src_id, "dst_id": e.dst_id,
             "relation": e.relation, "weight": e.weight}
            for e in edges
        ],
        "kb_id": kb_id,
    }


@app.get("/kg/stats", summary="图谱统计信息", tags=["KnowledgeGraph"])
async def kg_stats(kb_id: str = "global"):
    """返回知识图谱的节点数、边数、社区数等统计信息。"""
    store = _kg()
    stats = await store.get_stats(kb_id)
    return {"kb_id": kb_id, **stats}


@app.delete("/kg/documents/{doc_id}", summary="删除文档图谱数据", tags=["KnowledgeGraph"])
async def kg_delete_document(doc_id: str, kb_id: str = "global"):
    """删除指定文档产生的所有节点和边（节点若被其他文档共享则保留）。"""
    store = _kg()
    await store.delete_by_doc(doc_id, kb_id)
    return {"ok": True, "doc_id": doc_id}


@app.post("/kg/reindex/{doc_id}", summary="重建文档图谱", tags=["KnowledgeGraph"])
async def kg_reindex_document(doc_id: str, kb_id: str = "global"):
    """先删除文档旧图谱数据，再触发重建（需要原始文本仍在 KnowledgeBase 中）。"""
    store = _kg()
    await store.delete_by_doc(doc_id, kb_id)
    return {"ok": True, "message": "旧数据已清除，请重新调用 /kg/build/text 或 /kg/build/file 触发构建"}


@app.on_event("startup")
async def _startup():
    """
    FastAPI startup hook — finish PKB index recovery once the event loop is live.
    _container is set in main() before uvicorn starts, so it is available here.
    """
    if _container is not None:
        pkb = getattr(_container, "knowledge_base", None)
        if pkb and hasattr(pkb, "initialize"):
            try:
                await pkb.initialize()
                log.info("server.pkb_initialized")
            except Exception as exc:
                log.warning("server.pkb_startup_init_failed", error=str(exc))


@app.get("/health", summary="健康检查")
async def health():
    """返回服务状态与已加载组件信息。"""
    if _container is None:
        return {"status": "initializing"}
    skills = _container.skill_registry.list_descriptors() if _container.skill_registry else []
    return {
        "status": "ok",
        "skills": len(skills),
        "skill_names": [d.name for d in skills],
    }


# ── 启动入口 ───────────────────────────────────────────────────────

def main():
    global _container

    parser = argparse.ArgumentParser(description="AI Agent HTTP 服务器")
    parser.add_argument("--host",   default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port",   default=8000, type=int, help="监听端口（默认 8000）")
    parser.add_argument("--engine", default=None, help="指定 llm.yaml 中的引擎 alias")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载（暂不支持）")
    args = parser.parse_args()

    print("正在初始化 Agent 容器（加载 llm.yaml）…")
    _container = _build_container(args.engine)

    from skills.loader import SkillLoader
    hub_names  = {m.get("name") for m in SkillLoader.manifests()}
    all_skills = _container.skill_registry.list_descriptors()
    builtin    = [d.name for d in all_skills if d.name not in hub_names]
    hub        = [d.name for d in all_skills if d.name in hub_names]
    print(f"  内置 Skill ({len(builtin)}): {builtin}")
    print(f"  Hub  Skill ({len(hub)}):  {hub}")
    print(f"\n服务地址:")
    print(f"  普通请求(POST): http://{args.host}:{args.port}/chat")
    print(f"  普通请求(GET):  http://{args.host}:{args.port}/chat?text=你好")
    print(f"  流式请求:       http://{args.host}:{args.port}/chat/stream")
    print(f"  Skill 列表:     http://{args.host}:{args.port}/skills")
    print(f"  健康检查:       http://{args.host}:{args.port}/health")
    print(f"  API 文档:       http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
