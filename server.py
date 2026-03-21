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

    container = AgentContainer(
        llm_router        = llm_router,
        short_term_memory = InMemoryShortTermMemory(),
        long_term_memory  = workspace_ltm,     # 替换为 workspace 感知 LTM
        skill_registry    = skill_registry,
        mcp_hub           = DefaultMCPHub(),
        context_manager   = PriorityContextManager(),
        workspace_manager = workspace_manager,
    )
    return container.build()


# ── 核心：运行 Agent，收集事件 ─────────────────────────────────────

def _get_container(
    skills:        list[str] | None,
    system_prompt: str | None = None,
    workspace_ctx: Any        = None,   # WorkspaceContext | None
):
    """
    返回适合本次请求的容器副本。

    workspace_ctx 存在时：
      - 用工作区的 system_prompt（优先级：request.system_prompt > ws.system_prompt）
      - 用工作区的 allowed_skills（优先级：request.skills > ws.allowed_skills）
      - 使用 WorkspaceAwareLTM（若全局 LTM 支持）
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
    container    = _get_container(req.skills, prompt, ws_ctx)
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
    container  = _get_container(req.skills, prompt, ws_ctx)
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
    creator_id:     str
    description:    str   = ""
    system_prompt:  str   = ""
    allowed_skills: list[str] = []
    token_budget:   int   = 16_000
    max_steps:      int   = 20

class _CreateProjectReq(_BM):
    name:           str
    creator_id:     str
    description:    str   = ""
    system_prompt:  str   = ""
    allowed_skills: list[str] = []
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
        name=req.name, creator_id=req.creator_id,
        description=req.description, system_prompt=req.system_prompt,
        allowed_skills=req.allowed_skills,
        token_budget=req.token_budget, max_steps=req.max_steps,
    )
    return {"workspace_id": ws.workspace_id, "name": ws.name}


@app.get("/workspaces", summary="列出工作区", tags=["Workspace"])
async def list_workspaces(user_id: str | None = None):
    workspaces = await _wm().list_workspaces(user_id)
    return {"workspaces": [
        {"workspace_id": w.workspace_id, "name": w.name,
         "description": w.description,
         "members": len(w.members), "projects": len(w.projects)}
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
        workspace_id=workspace_id, creator_id=req.creator_id,
        name=req.name, description=req.description,
        system_prompt=req.system_prompt, allowed_skills=req.allowed_skills,
        token_budget=req.token_budget, max_steps=req.max_steps,
    )
    return {"project_id": proj.project_id, "name": proj.name}


@app.get("/workspaces/{workspace_id}/projects", summary="列出项目", tags=["Workspace"])
async def list_projects(workspace_id: str, user_id: str | None = None):
    projects = await _wm().list_projects(workspace_id, user_id)
    return {"projects": [
        {"project_id": p.project_id, "name": p.name,
         "description": p.description, "members": len(p.members)}
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
