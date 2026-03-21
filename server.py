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

    container = AgentContainer(
        llm_router        = llm_router,
        short_term_memory = InMemoryShortTermMemory(),
        long_term_memory  = InMemoryLongTermMemory(),
        skill_registry    = skill_registry,
        mcp_hub           = DefaultMCPHub(),
        context_manager   = PriorityContextManager(),
    )
    return container.build()


# ── 核心：运行 Agent，收集事件 ─────────────────────────────────────

def _get_container(skills: list[str] | None, system_prompt: str | None = None):
    """返回适合本次请求的容器：若指定了 skills/system_prompt 则创建轻量副本。"""
    import dataclasses
    replacements: dict = {}

    if skills:
        from skills.loader import FilteredSkillRegistry
        replacements["skill_registry"] = FilteredSkillRegistry(
            _container.skill_registry, set(skills)
        )

    if system_prompt:
        replacements["context_manager"] = _PrefixedContextManager(
            _container.context_manager, system_prompt
        )

    return dataclasses.replace(_container, **replacements) if replacements else _container


async def _run_agent(req: ChatRequest):
    """返回 (session_id, reply, usage, steps) 四元组。"""
    from core.models import AgentConfig

    prompt     = req.system_prompt or (_MARKETING_SYSTEM_PROMPT if req.mode == "marketing" else None)
    container  = _get_container(req.skills, prompt)
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    config     = AgentConfig(stream=False, max_steps=req.max_steps)

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
    container  = _get_container(req.skills, prompt)
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    config     = AgentConfig(stream=True, max_steps=req.max_steps)

    yield f"data: {json.dumps({'type': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"

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
