"""
evolution/store.py — 进化数据持久化层

独立数据库 evolution.db（SQLite），与业务库完全隔离。
表结构：
  signals       — 所有采集到的信号（rag_query / bi_query / feedback）
  bi_store_profile  — 门店数据可用性画像
  bi_templates  — BI 成功查询模板
  kb_chunk_stats — chunk 命中统计
  evolution_actions — 已执行的进化动作记录
  evolution_config  — 动态配置 KV（覆盖 YAML）
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("evolution.store")

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- 原始信号
CREATE TABLE IF NOT EXISTS signals (
    signal_id   TEXT PRIMARY KEY,
    signal_type TEXT NOT NULL,          -- rag_query | bi_query | feedback
    source_id   TEXT NOT NULL,          -- 原始 event_id
    kb_id       TEXT DEFAULT '',
    bra_id      TEXT DEFAULT '',
    session_id  TEXT DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '{}',  -- JSON
    quality     REAL DEFAULT -1,        -- -1=未评分, 0-1=质量分
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_type       ON signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_session    ON signals(session_id);
CREATE INDEX IF NOT EXISTS idx_signals_kb         ON signals(kb_id);
CREATE INDEX IF NOT EXISTS idx_signals_braid      ON signals(bra_id);
CREATE INDEX IF NOT EXISTS idx_signals_created    ON signals(created_at);

-- BI 门店数据画像
CREATE TABLE IF NOT EXISTS bi_store_profile (
    bra_id         TEXT NOT NULL,
    metric         TEXT NOT NULL DEFAULT '*',  -- '*' 表示整体
    no_data_dates  TEXT DEFAULT '[]',           -- JSON 字符串列表
    first_data_ts  REAL DEFAULT 0,
    last_data_ts   REAL DEFAULT 0,
    total_queries  INTEGER DEFAULT 0,
    no_data_count  INTEGER DEFAULT 0,
    updated_at     REAL NOT NULL,
    PRIMARY KEY (bra_id, metric)
);

-- BI 成功查询模板
CREATE TABLE IF NOT EXISTS bi_templates (
    template_id     TEXT PRIMARY KEY,
    intent_text     TEXT NOT NULL,    -- 代表性的用户意图描述
    intent_embedding TEXT DEFAULT '',  -- JSON 向量（可选，用于相似匹配）
    canonical_params TEXT NOT NULL,   -- JSON: 成功的 API 参数
    bra_id          TEXT DEFAULT '',
    hit_count       INTEGER DEFAULT 1,
    success_rate    REAL DEFAULT 1.0,
    last_used_at    REAL NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_braid ON bi_templates(bra_id);

-- RAG chunk 命中统计
CREATE TABLE IF NOT EXISTS kb_chunk_stats (
    chunk_id        TEXT PRIMARY KEY,
    kb_id           TEXT NOT NULL,
    hit_count       INTEGER DEFAULT 0,    -- 被检索命中次数
    used_count      INTEGER DEFAULT 0,    -- 被 LLM 答案引用次数
    feedback_sum    REAL DEFAULT 0,
    feedback_count  INTEGER DEFAULT 0,
    last_hit_at     REAL DEFAULT 0,
    last_used_at    REAL DEFAULT 0,
    quality_score   REAL DEFAULT -1
);
CREATE INDEX IF NOT EXISTS idx_chunk_stats_kb ON kb_chunk_stats(kb_id);
CREATE INDEX IF NOT EXISTS idx_chunk_stats_quality ON kb_chunk_stats(quality_score);

-- 进化动作日志
CREATE TABLE IF NOT EXISTS evolution_actions (
    action_id    TEXT PRIMARY KEY,
    actor_name   TEXT NOT NULL,
    target_type  TEXT NOT NULL,
    target_id    TEXT DEFAULT '',
    description  TEXT DEFAULT '',
    success      INTEGER DEFAULT 1,
    error        TEXT DEFAULT '',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_actor ON evolution_actions(actor_name);

-- 动态配置覆盖（优先级高于 YAML）
CREATE TABLE IF NOT EXISTS evolution_config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,  -- JSON
    updated_at REAL NOT NULL
);
"""


class EvolutionStore:
    """Evolution SQLite 数据库访问层（同步操作包装在 asyncio.to_thread 中）。"""

    def __init__(self, db_path: str = "./data/evolution.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """建库建表。"""
        await asyncio.to_thread(self._sync_initialize)
        log.info("evolution.store.initialized", db=str(self._db_path))

    def _sync_initialize(self) -> None:
        conn = self._get_conn()
        conn.executescript(_DDL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def close(self) -> None:
        if self._conn:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    # ── 信号写入 ──────────────────────────────────────────────────────────────

    async def save_signal(
        self,
        signal_id:   str,
        signal_type: str,
        source_id:   str,
        payload:     dict,
        kb_id:       str = "",
        bra_id:      str = "",
        session_id:  str = "",
        quality:     float = -1.0,
    ) -> None:
        sql = """
            INSERT OR REPLACE INTO signals
            (signal_id, signal_type, source_id, kb_id, bra_id, session_id,
             payload, quality, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """
        args = (signal_id, signal_type, source_id, kb_id, bra_id, session_id,
                json.dumps(payload, ensure_ascii=False), quality, time.time())
        await asyncio.to_thread(self._exec, sql, args)

    async def update_signal_quality(self, signal_id: str, quality: float) -> None:
        await asyncio.to_thread(
            self._exec,
            "UPDATE signals SET quality=? WHERE signal_id=?",
            (quality, signal_id),
        )

    async def recent_signals(
        self, signal_type: str, hours: int = 24, limit: int = 1000
    ) -> list[dict]:
        since = time.time() - hours * 3600
        sql = """SELECT * FROM signals WHERE signal_type=? AND created_at>=?
                 ORDER BY created_at DESC LIMIT ?"""
        return await asyncio.to_thread(self._fetchall, sql, (signal_type, since, limit))

    # ── BI 门店画像 ───────────────────────────────────────────────────────────

    async def upsert_store_profile(
        self,
        bra_id:        str,
        metric:        str = "*",
        no_data_date:  str | None = None,
        has_data_ts:   float | None = None,
    ) -> None:
        """更新门店画像：追加无数据日期 或 更新数据范围。"""
        def _sync():
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM bi_store_profile WHERE bra_id=? AND metric=?",
                (bra_id, metric)
            ).fetchone()
            now = time.time()
            if row is None:
                no_dates = json.dumps([no_data_date] if no_data_date else [])
                conn.execute(
                    """INSERT INTO bi_store_profile
                       (bra_id, metric, no_data_dates, first_data_ts, last_data_ts,
                        total_queries, no_data_count, updated_at)
                       VALUES (?,?,?,?,?,1,?,?)""",
                    (bra_id, metric, no_dates,
                     has_data_ts or 0, has_data_ts or 0,
                     1 if no_data_date else 0, now),
                )
            else:
                # 追加无数据日期
                no_dates: list = json.loads(row["no_data_dates"] or "[]")
                if no_data_date and no_data_date not in no_dates:
                    no_dates.append(no_data_date)
                    if len(no_dates) > 365:   # 最多保留 365 天
                        no_dates = no_dates[-365:]
                first_ts = row["first_data_ts"] or 0
                last_ts  = row["last_data_ts"] or 0
                if has_data_ts:
                    first_ts = min(first_ts, has_data_ts) if first_ts else has_data_ts
                    last_ts  = max(last_ts,  has_data_ts)
                conn.execute(
                    """UPDATE bi_store_profile SET
                       no_data_dates=?, first_data_ts=?, last_data_ts=?,
                       total_queries=total_queries+1,
                       no_data_count=no_data_count+?,
                       updated_at=?
                       WHERE bra_id=? AND metric=?""",
                    (json.dumps(no_dates), first_ts, last_ts,
                     1 if no_data_date else 0, now, bra_id, metric),
                )
            conn.commit()
        await asyncio.to_thread(_sync)

    async def get_store_profile(self, bra_id: str) -> dict:
        rows = await asyncio.to_thread(
            self._fetchall,
            "SELECT * FROM bi_store_profile WHERE bra_id=?",
            (bra_id,),
        )
        return {r["metric"]: r for r in rows}

    async def is_known_no_data(self, bra_id: str, date_str: str) -> bool:
        """快速判断某门店某天是否已知无数据。"""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT no_data_dates FROM bi_store_profile WHERE bra_id=? AND metric='*'",
            (bra_id,),
        )
        if not row:
            return False
        no_dates: list = json.loads(row["no_data_dates"] or "[]")
        return date_str in no_dates

    # ── BI 查询模板 ───────────────────────────────────────────────────────────

    async def upsert_bi_template(
        self,
        template_id:     str,
        intent_text:     str,
        canonical_params: dict,
        bra_id:          str = "",
    ) -> None:
        now = time.time()
        def _sync():
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT * FROM bi_templates WHERE template_id=?",
                (template_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO bi_templates
                       (template_id, intent_text, canonical_params, bra_id,
                        hit_count, success_rate, last_used_at, created_at)
                       VALUES (?,?,?,?,1,1.0,?,?)""",
                    (template_id, intent_text,
                     json.dumps(canonical_params, ensure_ascii=False),
                     bra_id, now, now),
                )
            else:
                conn.execute(
                    """UPDATE bi_templates SET
                       hit_count=hit_count+1, last_used_at=? WHERE template_id=?""",
                    (now, template_id),
                )
            conn.commit()
        await asyncio.to_thread(_sync)

    async def get_bi_templates(self, bra_id: str = "", limit: int = 200) -> list[dict]:
        if bra_id:
            sql = """SELECT * FROM bi_templates WHERE bra_id=? OR bra_id=''
                     ORDER BY hit_count DESC LIMIT ?"""
            return await asyncio.to_thread(self._fetchall, sql, (bra_id, limit))
        sql = "SELECT * FROM bi_templates ORDER BY hit_count DESC LIMIT ?"
        return await asyncio.to_thread(self._fetchall, sql, (limit,))

    # ── Chunk 统计 ────────────────────────────────────────────────────────────

    async def record_chunk_hit(self, chunk_id: str, kb_id: str, used: bool = False) -> None:
        def _sync():
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO kb_chunk_stats (chunk_id, kb_id, hit_count, used_count, last_hit_at)
                   VALUES (?,?,1,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     hit_count=hit_count+1,
                     used_count=used_count+?,
                     last_hit_at=excluded.last_hit_at""",
                (chunk_id, kb_id, 1 if used else 0, time.time(), 1 if used else 0),
            )
            conn.commit()
        await asyncio.to_thread(_sync)

    async def record_chunk_feedback(
        self, chunk_ids: list[str], score: float
    ) -> None:
        """将用户反馈分摊到关联的 chunks。"""
        def _sync():
            conn = self._get_conn()
            for cid in chunk_ids:
                conn.execute(
                    """INSERT INTO kb_chunk_stats (chunk_id, kb_id, feedback_sum, feedback_count)
                       VALUES (?,'',?,1)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         feedback_sum=feedback_sum+?,
                         feedback_count=feedback_count+1""",
                    (cid, score, score),
                )
            conn.commit()
        await asyncio.to_thread(_sync)

    async def get_low_quality_chunks(
        self, kb_id: str, threshold: float = 0.3, limit: int = 100
    ) -> list[dict]:
        sql = """SELECT * FROM kb_chunk_stats
                 WHERE kb_id=? AND quality_score >= 0 AND quality_score < ?
                 ORDER BY quality_score ASC LIMIT ?"""
        return await asyncio.to_thread(self._fetchall, sql, (kb_id, threshold, limit))

    async def update_chunk_quality(self, chunk_id: str, score: float) -> None:
        await asyncio.to_thread(
            self._exec,
            """UPDATE kb_chunk_stats SET quality_score=? WHERE chunk_id=?""",
            (score, chunk_id),
        )

    async def get_chunk_stats(self, kb_id: str) -> list[dict]:
        sql = "SELECT * FROM kb_chunk_stats WHERE kb_id=?"
        return await asyncio.to_thread(self._fetchall, sql, (kb_id,))

    # ── 进化动作日志 ──────────────────────────────────────────────────────────

    async def log_action(
        self,
        action_id:   str,
        actor_name:  str,
        target_type: str,
        target_id:   str = "",
        description: str = "",
        success:     bool = True,
        error:       str = "",
    ) -> None:
        sql = """INSERT OR IGNORE INTO evolution_actions
                 (action_id, actor_name, target_type, target_id,
                  description, success, error, created_at)
                 VALUES (?,?,?,?,?,?,?,?)"""
        await asyncio.to_thread(
            self._exec, sql,
            (action_id, actor_name, target_type, target_id,
             description, int(success), error, time.time()),
        )

    async def recent_actions(self, hours: int = 24) -> list[dict]:
        since = time.time() - hours * 3600
        sql = "SELECT * FROM evolution_actions WHERE created_at>=? ORDER BY created_at DESC"
        return await asyncio.to_thread(self._fetchall, sql, (since,))

    # ── 动态配置 ──────────────────────────────────────────────────────────────

    async def get_config(self, key: str, default: Any = None) -> Any:
        row = await asyncio.to_thread(
            self._fetchone, "SELECT value FROM evolution_config WHERE key=?", (key,)
        )
        if row is None:
            return default
        return json.loads(row["value"])

    async def set_config(self, key: str, value: Any) -> None:
        await asyncio.to_thread(
            self._exec,
            """INSERT OR REPLACE INTO evolution_config (key, value, updated_at)
               VALUES (?,?,?)""",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )

    # ── 通用 helpers ──────────────────────────────────────────────────────────

    def _exec(self, sql: str, args: tuple = ()) -> None:
        conn = self._get_conn()
        conn.execute(sql, args)
        conn.commit()

    def _fetchone(self, sql: str, args: tuple = ()) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _fetchall(self, sql: str, args: tuple = ()) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]
