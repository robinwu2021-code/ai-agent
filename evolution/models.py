"""
evolution/models.py — 进化系统数据模型

所有事件、信号、动作的基础数据结构。
模块内部流转：
  Event（原始事件）→ Signal（已持久化的信号）→ Action（执行结果）
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ── 基础事件 ─────────────────────────────────────────────────────────────────

@dataclass
class BaseEvent:
    """所有进化事件的基类，由功能模块触发，经 EventBus 路由到各 Collector。"""
    event_id:   str   = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)
    session_id: str   = ""
    user_id:    str   = ""


# ── RAG 相关事件 ──────────────────────────────────────────────────────────────

@dataclass
class RagQueryEvent(BaseEvent):
    """RAG 检索完成后触发。"""
    event_type:         str        = "rag_query"
    kb_id:              str        = ""
    query_text:         str        = ""
    retrieved_chunks:   list[dict] = field(default_factory=list)  # [{chunk_id, score, rank}]
    answer_text:        str        = ""
    latency_ms:         int        = 0
    retrieval_strategy: str        = ""   # hybrid | vector | keyword | graph


@dataclass
class FeedbackEvent(BaseEvent):
    """用户显式反馈（👍/👎 或评分）。"""
    event_type: str   = "feedback"
    query_id:   str   = ""    # 关联的 rag_query / bi_query event_id
    source:     str   = ""    # "rag" | "bi"
    score:      float = 0.0   # 1-5 或 0/1
    comment:    str   = ""


# ── BI 相关事件 ───────────────────────────────────────────────────────────────

@dataclass
class BiQueryEvent(BaseEvent):
    """AgentBiSkill.execute() 完成后触发。"""
    event_type:   str        = "bi_query"
    user_text:    str        = ""    # 用户原始自然语言（由 server 注入）
    bra_id:       str        = ""
    mode:         str        = ""    # default | dish_top | low_stock | ...
    api_payload:  dict       = field(default_factory=dict)
    result_rows:  int        = 0
    no_data:      bool       = False
    error:        str        = ""
    latency_ms:   int        = 0
    retry_count:  int        = 0     # 本次会话中此工具已重试次数


# ── 信号（已持久化的分析单元）──────────────────────────────────────────────────

@dataclass
class Signal:
    """经 Collector 处理后写入 evolution.db 的标准化信号。"""
    signal_id:   str        = field(default_factory=lambda: uuid.uuid4().hex[:16])
    signal_type: str        = ""    # "rag_query" | "bi_query" | "feedback" | ...
    source_id:   str        = ""    # 原始 event_id
    kb_id:       str        = ""
    bra_id:      str        = ""
    session_id:  str        = ""
    payload:     dict       = field(default_factory=dict)
    quality:     float      = -1.0  # -1 = 未评分
    created_at:  float      = field(default_factory=time.time)


# ── 执行动作结果 ──────────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Actor 执行结果记录，写入 evolution.db。"""
    action_id:   str   = field(default_factory=lambda: uuid.uuid4().hex[:16])
    actor_name:  str   = ""
    target_type: str   = ""    # "chunk" | "kb_config" | "bi_template" | "store_profile" | "prompt"
    target_id:   str   = ""
    description: str   = ""
    success:     bool  = True
    error:       str   = ""
    created_at:  float = field(default_factory=time.time)
