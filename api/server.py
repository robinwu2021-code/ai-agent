"""
api/server.py — FastAPI 完整 REST 接口（含所有扩展模块端点）
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = structlog.get_logger(__name__)

app = FastAPI(
    title="AI Agent API",
    version="2.0.0",
    description="Modular AI Agent with Skill, MCP, Memory, RAG, HITL, Multi-Agent support",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_container = None

def get_container():
    if _container is None:
        raise RuntimeError("Container not initialized. Call init_app() first.")
    return _container

def init_app(container) -> None:
    global _container
    _container = container


# ── Request/Response Models ────────────────────────────────────────

class RunRequest(BaseModel):
    user_id:    str
    session_id: str | None = None
    input:      dict[str, Any]
    config:     dict[str, Any] = {}
    tenant_id:  str | None = None

class FeedbackRequest(BaseModel):
    user_id:    str
    session_id: str
    task_id:    str
    rating:     int           # 1-5
    comment:    str = ""
    category:   str = ""

class HITLResolveRequest(BaseModel):
    approved:      bool
    modified_args: dict | None = None
    feedback:      str = ""

class KBAddRequest(BaseModel):
    text:   str
    source: str = "api"

class SkillRegisterRequest(BaseModel):
    module_path: str
    class_name:  str
    init_kwargs: dict[str, Any] = {}

class TenantCreateRequest(BaseModel):
    tenant_id:      str
    name:           str
    system_prompt:  str = ""
    allowed_skills: list[str] = []
    max_steps:      int = 20
    daily_cost_limit: float = 10.0

class PromptRegisterRequest(BaseModel):
    name:        str
    version:     str
    content:     str
    description: str = ""

class QuotaSetRequest(BaseModel):
    daily_token_limit:  int   = 500_000
    daily_cost_limit:   float = 5.0
    monthly_cost_limit: float = 50.0
    allow_downgrade:    bool  = True


# ── Timing middleware ──────────────────────────────────────────────

@app.middleware("http")
async def add_timing(request: Request, call_next):
    start    = time.monotonic()
    response = await call_next(request)
    duration = round((time.monotonic() - start) * 1000)
    response.headers["X-Response-Time-Ms"] = str(duration)
    return response


# ════════════════════════════════════════════════════════
# AGENT — Core Execution
# ════════════════════════════════════════════════════════

@app.post("/v1/agent/run")
async def run_agent(req: RunRequest):
    """Execute an agent task with optional SSE streaming."""
    from core.models import AgentConfig

    c          = get_container()
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    config     = AgentConfig(**req.config) if req.config else AgentConfig()

    async def event_stream() -> AsyncIterator[str]:
        agent = c.agent()
        async for event in agent.run(
            user_id=req.user_id,
            session_id=session_id,
            text=req.input.get("text", ""),
            config=config,
            files=req.input.get("files"),
            tenant_id=req.tenant_id,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    if config.stream:
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"X-Session-Id": session_id},
        )

    events = []
    agent  = c.agent()
    async for event in agent.run(
        user_id=req.user_id, session_id=session_id,
        text=req.input.get("text", ""), config=config,
        files=req.input.get("files"), tenant_id=req.tenant_id,
    ):
        events.append(event)
    return {"session_id": session_id, "events": events}


# ════════════════════════════════════════════════════════
# MEMORY
# ════════════════════════════════════════════════════════

@app.get("/v1/memory/{user_id}/search")
async def search_memory(user_id: str, q: str, top_k: int = 5):
    c = get_container()
    results = await c.long_term_memory.search(user_id=user_id, query=q, top_k=top_k)
    return {"results": [
        {"id": e.id, "text": e.text, "type": e.type.value,
         "importance": e.importance, "created_at": e.created_at.isoformat()}
        for e in results
    ]}

@app.get("/v1/memory/{user_id}/profile")
async def get_profile(user_id: str):
    c = get_container()
    return {"user_id": user_id, "profile": await c.long_term_memory.get_profile(user_id)}

@app.delete("/v1/memory/{user_id}")
async def clear_memory(user_id: str):
    c      = get_container()
    pruned = await c.long_term_memory.prune(user_id=user_id, max_items=0, score_threshold=1.1)
    return {"pruned": pruned}


# ════════════════════════════════════════════════════════
# SKILLS & MCP
# ════════════════════════════════════════════════════════

@app.get("/v1/skills")
async def list_skills():
    c = get_container()
    return {"skills": [
        {"name": d.name, "description": d.description,
         "source": d.source, "tags": d.tags, "permission": d.permission}
        for d in c.skill_registry.list_descriptors()
    ]}

@app.post("/v1/skills/register")
async def register_skill(req: SkillRegisterRequest):
    import importlib
    c = get_container()
    try:
        module = importlib.import_module(req.module_path)
        cls    = getattr(module, req.class_name)
        skill  = cls(**req.init_kwargs)
        c.skill_registry.register(skill)
        return {"registered": skill.descriptor.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/mcp/servers")
async def list_mcp_servers():
    c = get_container()
    return {"tools": [
        {"name": d.name, "description": d.description, "tags": d.tags}
        for d in c.mcp_hub.list_descriptors()
    ]}


# ════════════════════════════════════════════════════════
# RAG — Knowledge Base
# ════════════════════════════════════════════════════════

@app.post("/v1/kb/add")
async def kb_add_text(req: KBAddRequest):
    c = get_container()
    if not c.knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not configured")
    n = await c.knowledge_base.add_text(req.text, source=req.source)
    return {"chunks_added": n, "total_docs": c.knowledge_base.doc_count,
            "total_chunks": c.knowledge_base.chunk_count}

@app.get("/v1/kb/search")
async def kb_search(q: str, top_k: int = 5):
    c = get_container()
    if not c.knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not configured")
    chunks = await c.knowledge_base.query(q, top_k=top_k)
    return {"results": [
        {"id": ch.id, "text": ch.text, "score": round(ch.score, 4),
         "source": ch.metadata.get("source", "")}
        for ch in chunks
    ]}

@app.get("/v1/kb/stats")
async def kb_stats():
    c = get_container()
    if not c.knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not configured")
    return {"doc_count": c.knowledge_base.doc_count,
            "chunk_count": c.knowledge_base.chunk_count}


# ════════════════════════════════════════════════════════
# HITL — Human-in-the-loop
# ════════════════════════════════════════════════════════

@app.get("/v1/hitl/pending")
async def hitl_pending():
    c = get_container()
    if not c.hitl_manager:
        return {"pending": []}
    return {"pending": [vars(r) for r in c.hitl_manager.get_pending()]}

@app.post("/v1/hitl/{request_id}/resolve")
async def hitl_resolve(request_id: str, req: HITLResolveRequest):
    c = get_container()
    if not c.hitl_manager:
        raise HTTPException(status_code=503, detail="HITL not configured")
    ok = c.hitl_manager.resolve(
        request_id, req.approved,
        modified_args=req.modified_args,
        feedback=req.feedback,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    return {"resolved": request_id, "approved": req.approved}

@app.get("/v1/hitl/history")
async def hitl_history():
    c = get_container()
    if not c.hitl_manager:
        return {"history": []}
    return {"history": c.hitl_manager.get_history()}


# ════════════════════════════════════════════════════════
# COST & QUOTA
# ════════════════════════════════════════════════════════

@app.get("/v1/cost/{user_id}/daily")
async def cost_daily(user_id: str):
    c = get_container()
    if not c.cost_tracker:
        raise HTTPException(status_code=503, detail="Cost tracking not configured")
    return {"user_id": user_id, "usage": c.cost_tracker.get_daily_usage(user_id)}

@app.get("/v1/cost/report")
async def cost_report():
    from utils.cost import CostReport
    c = get_container()
    if not c.cost_tracker:
        raise HTTPException(status_code=503, detail="Cost tracking not configured")
    report = CostReport(c.cost_tracker)
    return {"daily": report.daily_report(), "by_model": report.model_breakdown()}

@app.post("/v1/quota/{user_id}")
async def set_quota(user_id: str, req: QuotaSetRequest):
    from utils.cost import QuotaConfig
    c = get_container()
    if not c.quota_manager:
        raise HTTPException(status_code=503, detail="Quota manager not configured")
    c.quota_manager.set_quota(user_id, QuotaConfig(
        daily_token_limit=req.daily_token_limit,
        daily_cost_limit=req.daily_cost_limit,
        monthly_cost_limit=req.monthly_cost_limit,
        allow_downgrade=req.allow_downgrade,
    ))
    return {"user_id": user_id, "quota_set": True}

@app.get("/v1/quota/{user_id}")
async def get_quota(user_id: str):
    c = get_container()
    if not c.quota_manager:
        raise HTTPException(status_code=503, detail="Quota manager not configured")
    quota = c.quota_manager.get_quota(user_id)
    usage = c.cost_tracker.get_daily_usage(user_id) if c.cost_tracker else {}
    return {"user_id": user_id, "quota": vars(quota), "today_usage": usage}


# ════════════════════════════════════════════════════════
# SECURITY & AUDIT
# ════════════════════════════════════════════════════════

@app.get("/v1/audit")
async def audit_log(user_id: str | None = None, limit: int = 100):
    c = get_container()
    if not c.security_manager:
        return {"entries": []}
    return {"entries": c.security_manager.audit.get_entries(user_id=user_id, limit=limit)}


# ════════════════════════════════════════════════════════
# FEEDBACK & EVAL
# ════════════════════════════════════════════════════════

@app.post("/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    c = get_container()
    if not c.feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not configured")
    fb = c.feedback_store.submit(
        user_id=req.user_id, session_id=req.session_id,
        task_id=req.task_id, rating=req.rating,
        comment=req.comment, category=req.category,
    )
    return {"feedback_id": fb.id, "recorded": True}

@app.get("/v1/feedback/stats")
async def feedback_stats(days: int = 7):
    c = get_container()
    if not c.feedback_store:
        return {"stats": {}}
    return {"stats": c.feedback_store.get_stats(days=days)}


# ════════════════════════════════════════════════════════
# TENANT
# ════════════════════════════════════════════════════════

@app.post("/v1/tenants")
async def create_tenant(req: TenantCreateRequest):
    from tenant.manager import TenantContext
    c = get_container()
    if not c.tenant_manager:
        raise HTTPException(status_code=503, detail="Tenant manager not configured")
    ctx = TenantContext(
        tenant_id=req.tenant_id, name=req.name,
        system_prompt=req.system_prompt,
        allowed_skills=req.allowed_skills,
        max_steps=req.max_steps,
        daily_cost_limit=req.daily_cost_limit,
    )
    c.tenant_manager.register(ctx)
    return {"tenant_id": req.tenant_id, "created": True}

@app.get("/v1/tenants")
async def list_tenants():
    c = get_container()
    if not c.tenant_manager:
        return {"tenants": []}
    return {"tenants": c.tenant_manager.list_tenants()}


# ════════════════════════════════════════════════════════
# PROMPT MANAGEMENT
# ════════════════════════════════════════════════════════

@app.post("/v1/prompts")
async def register_prompt(req: PromptRegisterRequest):
    from prompt_mgr.manager import PromptTemplate
    import uuid as _uuid
    c = get_container()
    if not c.prompt_renderer:
        raise HTTPException(status_code=503, detail="Prompt manager not configured")
    t = PromptTemplate(
        id=f"api_{_uuid.uuid4().hex[:6]}",
        name=req.name, version=req.version,
        content=req.content, description=req.description,
    )
    c.prompt_renderer._registry.register(t)
    return {"name": req.name, "version": req.version, "registered": True}

@app.get("/v1/prompts/{name}/versions")
async def list_prompt_versions(name: str):
    c = get_container()
    if not c.prompt_renderer:
        raise HTTPException(status_code=503, detail="Prompt manager not configured")
    return {"name": name, "versions": c.prompt_renderer._registry.list_versions(name)}

@app.post("/v1/prompts/{name}/activate/{version}")
async def activate_prompt(name: str, version: str):
    c = get_container()
    if not c.prompt_renderer:
        raise HTTPException(status_code=503, detail="Prompt manager not configured")
    ok = c.prompt_renderer._registry.set_active(name, version)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")
    return {"name": name, "active_version": version}

@app.post("/v1/prompts/{name}/rollback")
async def rollback_prompt(name: str):
    c = get_container()
    if not c.prompt_renderer:
        raise HTTPException(status_code=503, detail="Prompt manager not configured")
    ok = c.prompt_renderer._registry.rollback(name)
    return {"name": name, "rolled_back": ok}


# ════════════════════════════════════════════════════════
# TASK QUEUE
# ════════════════════════════════════════════════════════

@app.get("/v1/queue/stats")
async def queue_stats():
    c = get_container()
    if not c.task_queue:
        return {"stats": {}}
    return {"stats": c.task_queue.stats}

@app.get("/v1/queue/jobs")
async def list_jobs(status: str | None = None, limit: int = 50):
    from queue.scheduler import JobStatus
    c = get_container()
    if not c.task_queue:
        return {"jobs": []}
    st   = JobStatus(status) if status else None
    jobs = c.task_queue.list_jobs(status=st, limit=limit)
    return {"jobs": [vars(j) for j in jobs]}

@app.delete("/v1/queue/jobs/{job_id}")
async def cancel_job(job_id: str):
    c  = get_container()
    ok = c.task_queue.cancel(job_id) if c.task_queue else False
    return {"job_id": job_id, "cancelled": ok}


# ════════════════════════════════════════════════════════
# HEALTH & METRICS
# ════════════════════════════════════════════════════════

@app.get("/v1/health")
async def health():
    c = get_container()
    return {
        "status":  "ok",
        "version": "2.0.0",
        "modules": {
            "llm":       c.llm_engine is not None,
            "memory":    c.short_term_memory is not None,
            "skills":    len(c.skill_registry.list_descriptors()) if c.skill_registry else 0,
            "mcp":       len(c.mcp_hub.list_descriptors()) if c.mcp_hub else 0,
            "security":  c.security_manager is not None,
            "rag":       c.knowledge_base is not None,
            "hitl":      c.hitl_manager is not None,
            "cost":      c.cost_tracker is not None,
            "tenant":    c.tenant_manager is not None,
            "prompts":   c.prompt_renderer is not None,
            "queue":     c.task_queue is not None,
        },
    }

@app.get("/v1/metrics")
async def metrics():
    from utils.observability import metrics as m
    return m.summary()
