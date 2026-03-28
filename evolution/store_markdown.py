"""
evolution/store_markdown.py — Markdown 存储后端

画像类数据（bi_store_profile / bi_templates / qa_pairs / config / actions）
→ 本地 Markdown 文件，人类可读可编辑，方便版本管理

信号类数据（signals / kb_chunk_stats）→ 轻量 SQLite（高频写入，需要索引查询）

目录结构：
  {markdown_dir}/
  ├── bi_profiles/          # 门店数据画像，每门店一个 .md
  │   └── {bra_id}.md
  ├── qa_pairs/             # 提取的 Q&A 对
  │   ├── pending/          # 待注入知识库
  │   └── injected/         # 已注入
  ├── bi_templates/         # BI 查询模板，每门店（或全局）一个 .md
  │   └── {bra_id|_global}.md
  ├── config/               # 动态配置 KV
  │   └── {key_sanitized}.md
  └── actions.md            # 进化动作日志（追加）
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("evolution.store.markdown")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _ts_to_date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""


def _ts_to_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


def _now_dt() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_key(key: str) -> str:
    """将任意字符串转为合法文件名（保留字母数字点横杠，其余替换为下划线）。"""
    return re.sub(r"[^\w\-.]", "_", key)[:120]


def _parse_frontmatter(text: str) -> dict:
    """解析 YAML front-matter（--- 包围的块）。"""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    fm_text = text[3:end].strip()
    try:
        import yaml
        return yaml.safe_load(fm_text) or {}
    except Exception as exc:
        log.warning("markdown_store.parse_frontmatter_failed", error=str(exc))
        return {}


# ── MarkdownEvolutionStore ────────────────────────────────────────────────────

class MarkdownEvolutionStore:
    """
    Markdown 存储后端。

    与 EvolutionStore（SQLite）接口完全一致，可通过工厂函数透明切换。
    """

    def __init__(
        self,
        markdown_dir: str = "./data/evolution",
        signals_db_path: str = "./data/evolution_signals.db",
    ) -> None:
        self._root = Path(markdown_dir)
        self._signals_db_path = signals_db_path
        self._signal_store: Any = None  # EvolutionStore 懒加载

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        for sub in ["bi_profiles", "qa_pairs/pending", "qa_pairs/injected",
                    "bi_templates", "config"]:
            (self._root / sub).mkdir(parents=True, exist_ok=True)
        await self._sig().initialize()
        log.info("evolution.store.markdown.initialized", root=str(self._root))

    async def close(self) -> None:
        if self._signal_store is not None:
            await self._signal_store.close()

    def _sig(self):
        """获取信号/Chunk 用的 SQLite 后端（懒加载）。"""
        if self._signal_store is None:
            from evolution.store import EvolutionStore
            self._signal_store = EvolutionStore(self._signals_db_path)
        return self._signal_store

    # ── 信号（委托 SQLite）────────────────────────────────────────────────────

    async def save_signal(
        self, signal_id: str, signal_type: str, source_id: str,
        payload: dict, kb_id: str = "", bra_id: str = "",
        session_id: str = "", quality: float = -1.0,
    ) -> None:
        await self._sig().save_signal(
            signal_id, signal_type, source_id, payload,
            kb_id=kb_id, bra_id=bra_id,
            session_id=session_id, quality=quality,
        )
        # Q&A 对额外写入 Markdown 文件
        if signal_type == "qa_pair":
            await asyncio.to_thread(
                self._write_qa_pair, signal_id, payload, quality
            )

    async def update_signal_quality(self, signal_id: str, quality: float) -> None:
        await self._sig().update_signal_quality(signal_id, quality)

    async def recent_signals(
        self, signal_type: str, hours: int = 24, limit: int = 1000
    ) -> list[dict]:
        return await self._sig().recent_signals(signal_type, hours, limit)

    # ── Q&A 对 Markdown ───────────────────────────────────────────────────────

    def _write_qa_pair(self, signal_id: str, payload: dict, quality: float) -> None:
        short_id = signal_id[:8]
        date_str = datetime.now().strftime("%Y-%m-%d")
        fname = f"{date_str}_{short_id}.md"
        p = self._root / "qa_pairs" / "pending" / fname
        query = payload.get("query", "")
        answer = payload.get("answer", "")
        content = (
            f"---\n"
            f"id: \"{signal_id}\"\n"
            f"kb_id: \"{payload.get('kb_id', '')}\"\n"
            f"session_id: \"{payload.get('session_id', '')}\"\n"
            f"quality: {quality:.3f}\n"
            f"created_at: \"{_now_dt()}\"\n"
            f"injected: false\n"
            f"---\n\n"
            f"## 问题\n\n{query}\n\n"
            f"## 回答\n\n{answer}\n"
        )
        p.write_text(content, encoding="utf-8")

    def _mark_qa_injected(self, signal_id: str) -> None:
        """将 qa_pair 文件从 pending/ 移动到 injected/。"""
        short_id = signal_id[:8]
        for pending in (self._root / "qa_pairs" / "pending").glob(f"*_{short_id}.md"):
            dest = self._root / "qa_pairs" / "injected" / pending.name
            pending.rename(dest)
            break

    # ── BI 门店画像 ───────────────────────────────────────────────────────────

    async def upsert_store_profile(
        self,
        bra_id: str,
        metric: str = "*",
        no_data_date: str | None = None,
        has_data_ts: float | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._sync_upsert_profile, bra_id, no_data_date, has_data_ts
        )

    def _sync_upsert_profile(
        self, bra_id: str,
        no_data_date: str | None,
        has_data_ts: float | None,
    ) -> None:
        p = self._root / "bi_profiles" / f"{bra_id}.md"
        data = _parse_frontmatter(p.read_text(encoding="utf-8")) if p.exists() else {}

        data.setdefault("bra_id", bra_id)
        data["updated_at"] = time.time()
        data["total_queries"] = data.get("total_queries", 0) + 1

        no_dates: list = data.get("no_data_dates") or []
        if no_data_date and no_data_date not in no_dates:
            no_dates.append(no_data_date)
            no_dates = sorted(set(no_dates))[-365:]
            data["no_data_count"] = data.get("no_data_count", 0) + 1
        data["no_data_dates"] = no_dates

        if has_data_ts:
            fts = data.get("first_data_ts") or 0
            lts = data.get("last_data_ts") or 0
            data["first_data_ts"] = min(fts, has_data_ts) if fts else has_data_ts
            data["last_data_ts"] = max(lts, has_data_ts)

        self._render_profile(p, data)

    def _render_profile(self, p: Path, d: dict) -> None:
        bra_id    = d.get("bra_id", "")
        total     = d.get("total_queries", 0)
        no_cnt    = d.get("no_data_count", 0)
        no_dates  = d.get("no_data_dates") or []
        first_dt  = _ts_to_date(d.get("first_data_ts") or 0)
        last_dt   = _ts_to_date(d.get("last_data_ts") or 0)
        updated   = _ts_to_dt(d.get("updated_at") or time.time())
        pct       = f"{no_cnt / max(total, 1) * 100:.1f}%"

        recent_30 = sorted(no_dates[-30:], reverse=True) if no_dates else []
        dates_md  = "\n".join(f"- {dt}" for dt in recent_30) or "_（无）_"

        # YAML front-matter（机器可读）
        fm_lines = [
            "---",
            f'bra_id: "{bra_id}"',
            f'updated_at: "{updated}"',
            f"total_queries: {total}",
            f"no_data_count: {no_cnt}",
            f"first_data_ts: {d.get('first_data_ts') or 0}",
            f"last_data_ts: {d.get('last_data_ts') or 0}",
        ]
        if no_dates:
            fm_lines.append("no_data_dates:")
            for dt in no_dates:
                fm_lines.append(f'  - "{dt}"')
        else:
            fm_lines.append("no_data_dates: []")
        fm_lines.append("---")

        body = (
            f"\n# 门店数据画像 — {bra_id}\n\n"
            f"**更新时间：** {updated}  \n"
            f"**查询统计：** 累计 {total} 次，无数据 {no_cnt} 次（{pct}）  \n"
            f"**数据范围：** {first_dt or '未知'} 至 {last_dt or '未知'}\n\n"
            f"## 近期无数据日期（最近 30 天）\n\n{dates_md}\n\n"
            f"## 历史无数据日期（共 {len(no_dates)} 天）\n\n"
            + (", ".join(sorted(no_dates)) if no_dates else "_（无）_")
            + "\n"
        )
        p.write_text("\n".join(fm_lines) + body, encoding="utf-8")

    async def get_store_profile(self, bra_id: str) -> dict:
        p = self._root / "bi_profiles" / f"{bra_id}.md"
        if not p.exists():
            return {}
        data = _parse_frontmatter(p.read_text(encoding="utf-8"))
        return {"*": data} if data else {}

    async def all_store_profiles(self) -> list[dict]:
        result = []
        for f in (self._root / "bi_profiles").glob("*.md"):
            d = _parse_frontmatter(f.read_text(encoding="utf-8"))
            if d:
                result.append(d)
        return result

    async def is_known_no_data(self, bra_id: str, date_str: str) -> bool:
        p = self._root / "bi_profiles" / f"{bra_id}.md"
        if not p.exists():
            return False
        d = _parse_frontmatter(p.read_text(encoding="utf-8"))
        return date_str in (d.get("no_data_dates") or [])

    # ── BI 查询模板 ───────────────────────────────────────────────────────────

    async def upsert_bi_template(
        self,
        template_id: str,
        intent_text: str,
        canonical_params: dict,
        bra_id: str = "",
    ) -> None:
        await asyncio.to_thread(
            self._sync_upsert_template,
            template_id, intent_text, canonical_params, bra_id,
        )

    def _sync_upsert_template(
        self, template_id: str, intent_text: str,
        canonical_params: dict, bra_id: str,
    ) -> None:
        key = bra_id or "_global"
        p = self._root / "bi_templates" / f"{key}.md"
        templates = self._read_templates_file(p)
        now = time.time()
        if template_id in templates:
            t = templates[template_id]
            t["hit_count"] = t.get("hit_count", 1) + 1
            t["last_used_at"] = now
        else:
            templates[template_id] = {
                "template_id": template_id,
                "intent_text": intent_text,
                "canonical_params": canonical_params,
                "bra_id": bra_id,
                "hit_count": 1,
                "success_rate": 1.0,
                "last_used_at": now,
                "created_at": now,
            }
        self._render_templates(p, bra_id, templates)

    def _read_templates_file(self, p: Path) -> dict:
        if not p.exists():
            return {}
        d = _parse_frontmatter(p.read_text(encoding="utf-8"))
        result = {}
        for t in d.get("templates") or []:
            tid = t.get("template_id")
            if not tid:
                continue
            # canonical_params 存为 JSON 字符串
            cp = t.get("canonical_params", "{}")
            if isinstance(cp, str):
                try:
                    cp = json.loads(cp)
                except Exception:
                    cp = {}
            t["canonical_params"] = cp
            result[tid] = t
        return result

    def _render_templates(self, p: Path, bra_id: str, templates: dict) -> None:
        items = sorted(templates.values(), key=lambda x: -(x.get("hit_count") or 0))

        # YAML front-matter（机器可读）
        fm_lines = [
            "---",
            f'bra_id: "{bra_id or "全局"}"',
            f'updated_at: "{_now_dt()}"',
            f"total_templates: {len(items)}",
            "templates:",
        ]
        for t in items:
            params_json = json.dumps(t.get("canonical_params") or {}, ensure_ascii=False)
            intent = str(t.get("intent_text", ""))[:100].replace('"', "'")
            fm_lines += [
                f'  - template_id: "{t["template_id"]}"',
                f'    intent_text: "{intent}"',
                f'    bra_id: "{t.get("bra_id", "")}"',
                f'    hit_count: {t.get("hit_count", 1)}',
                f'    success_rate: {t.get("success_rate", 1.0):.2f}',
                f'    last_used_at: {t.get("last_used_at", 0)}',
                f'    created_at: {t.get("created_at", 0)}',
                f"    canonical_params: '{params_json}'",
            ]
        fm_lines.append("---")

        # 人类可读主体
        body_lines = [
            f"\n# BI 查询模板库 — {bra_id or '全局'}",
            f"\n**更新时间：** {_now_dt()}　　**模板数：** {len(items)}\n",
        ]
        for i, t in enumerate(items, 1):
            intent = t.get("intent_text", "")[:60]
            params = t.get("canonical_params") or {}
            body_lines.append(f"## 模板 {i:03d} — {intent}\n")
            for k, v in params.items():
                body_lines.append(f"- **{k}：** `{v}`")
            body_lines += [
                f"- **使用次数：** {t.get('hit_count', 1)}",
                f"- **成功率：** {t.get('success_rate', 1.0):.0%}",
                f"- **最后使用：** {_ts_to_dt(t.get('last_used_at') or 0)}",
                "",
            ]

        p.write_text("\n".join(fm_lines) + "\n".join(body_lines), encoding="utf-8")

    async def get_bi_templates(self, bra_id: str = "", limit: int = 200) -> list[dict]:
        tmpl_dir = self._root / "bi_templates"
        results: list[dict] = []
        if bra_id:
            for key in [bra_id, "_global"]:
                p = tmpl_dir / f"{key}.md"
                results.extend(self._read_templates_file(p).values())
        else:
            for p in tmpl_dir.glob("*.md"):
                results.extend(self._read_templates_file(p).values())
        results.sort(key=lambda x: -(x.get("hit_count") or 0))
        return results[:limit]

    # ── Chunk 统计（委托 SQLite）──────────────────────────────────────────────

    async def record_chunk_hit(self, chunk_id: str, kb_id: str, used: bool = False) -> None:
        await self._sig().record_chunk_hit(chunk_id, kb_id, used)

    async def record_chunk_feedback(self, chunk_ids: list[str], score: float) -> None:
        await self._sig().record_chunk_feedback(chunk_ids, score)

    async def get_low_quality_chunks(
        self, kb_id: str, threshold: float = 0.3, limit: int = 100
    ) -> list[dict]:
        return await self._sig().get_low_quality_chunks(kb_id, threshold, limit)

    async def update_chunk_quality(self, chunk_id: str, score: float) -> None:
        await self._sig().update_chunk_quality(chunk_id, score)

    async def get_chunk_stats(self, kb_id: str) -> list[dict]:
        return await self._sig().get_chunk_stats(kb_id)

    # ── 进化动作日志 ──────────────────────────────────────────────────────────

    async def log_action(
        self,
        action_id: str,
        actor_name: str,
        target_type: str,
        target_id: str = "",
        description: str = "",
        success: bool = True,
        error: str = "",
    ) -> None:
        await asyncio.to_thread(
            self._sync_log_action,
            action_id, actor_name, target_type, target_id, description, success, error,
        )

    def _sync_log_action(
        self, action_id: str, actor_name: str, target_type: str,
        target_id: str, description: str, success: bool, error: str,
    ) -> None:
        p = self._root / "actions.md"
        if not p.exists():
            p.write_text("# 进化动作日志\n\n---\n", encoding="utf-8")

        status = "✅ 成功" if success else "❌ 失败"
        lines = [
            f"\n## {_now_dt()} — {actor_name} [{status}]",
            f"",
            f"| 字段 | 值 |",
            f"|------|----|",
            f"| 动作ID | `{action_id[:16]}` |",
            f"| 目标类型 | {target_type} |",
            f"| 目标ID | {target_id or '-'} |",
        ]
        if description:
            lines += ["", description]
        if error:
            lines += ["", f"> ⚠ 错误：{error}"]
        lines += ["", "---", ""]

        with open(p, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    async def recent_actions(self, hours: int = 24) -> list[dict]:
        p = self._root / "actions.md"
        if not p.exists():
            return []
        return await asyncio.to_thread(self._parse_recent_actions, p, hours)

    def _parse_recent_actions(self, p: Path, hours: int) -> list[dict]:
        since = time.time() - hours * 3600
        text = p.read_text(encoding="utf-8")
        results = []
        for block in text.split("\n## ")[1:]:
            first_line = block.split("\n")[0].strip()
            # "2024-01-15 02:00:00 — actor_name [status]"
            dt_part = first_line.split(" — ")[0].strip()
            try:
                ts = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S").timestamp()
                if ts >= since:
                    results.append({"header": first_line, "ts": ts})
            except Exception:
                pass
        return results

    # ── 动态配置 ──────────────────────────────────────────────────────────────

    async def get_config(self, key: str, default: Any = None) -> Any:
        p = self._root / "config" / f"{_sanitize_key(key)}.md"
        if not p.exists():
            return default
        d = _parse_frontmatter(p.read_text(encoding="utf-8"))
        return d.get("value", default)

    async def set_config(self, key: str, value: Any) -> None:
        await asyncio.to_thread(self._sync_set_config, key, value)

    def _sync_set_config(self, key: str, value: Any) -> None:
        p = self._root / "config" / f"{_sanitize_key(key)}.md"
        value_str = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, indent=2
        )
        # 判断是否为提示词（prompt_ 前缀），用 text block 展示
        is_prompt = "prompt_" in key
        lang = "" if is_prompt else "json"

        fm_lines = [
            "---",
            f'key: "{key}"',
            f'updated_at: "{_now_dt()}"',
        ]
        # value 在 YAML 中序列化
        if isinstance(value, str):
            escaped = value.replace('"', '\\"')
            fm_lines.append(f'value: "{escaped}"')
        else:
            fm_lines.append(f"value: {json.dumps(value, ensure_ascii=False)}")
        fm_lines.append("---")

        body = (
            f"\n# 配置项：`{key}`\n\n"
            f"**更新时间：** {_now_dt()}\n\n"
            f"## 值\n\n"
            f"```{lang}\n{value_str}\n```\n"
        )
        p.write_text("\n".join(fm_lines) + body, encoding="utf-8")


# ── HybridEvolutionStore ──────────────────────────────────────────────────────

class HybridEvolutionStore:
    """
    混合模式：SQLite 处理全部数据（保证性能和查询能力）
    + Markdown 同步镜像画像类数据（人类可读、可版本管理）。

    信号/Chunk 仅写 SQLite（高频，Markdown 无意义）。
    画像/模板/配置/动作同时写入两侧（双写）。
    """

    def __init__(
        self,
        sqlite_store: Any,           # EvolutionStore
        markdown_store: "MarkdownEvolutionStore",
    ) -> None:
        self._sq = sqlite_store
        self._md = markdown_store

    async def initialize(self) -> None:
        await self._sq.initialize()
        await self._md.initialize()

    async def close(self) -> None:
        await self._sq.close()
        await self._md.close()

    # ── 信号（仅 SQLite）──────────────────────────────────────────────────────

    async def save_signal(self, signal_id, signal_type, source_id, payload,
                          kb_id="", bra_id="", session_id="", quality=-1.0):
        await self._sq.save_signal(
            signal_id, signal_type, source_id, payload,
            kb_id=kb_id, bra_id=bra_id,
            session_id=session_id, quality=quality,
        )
        if signal_type == "qa_pair":
            await asyncio.to_thread(self._md._write_qa_pair, signal_id, payload, quality)

    async def update_signal_quality(self, signal_id, quality):
        await self._sq.update_signal_quality(signal_id, quality)

    async def recent_signals(self, signal_type, hours=24, limit=1000):
        return await self._sq.recent_signals(signal_type, hours, limit)

    # ── Chunk 统计（仅 SQLite）────────────────────────────────────────────────

    async def record_chunk_hit(self, chunk_id, kb_id, used=False):
        await self._sq.record_chunk_hit(chunk_id, kb_id, used)

    async def record_chunk_feedback(self, chunk_ids, score):
        await self._sq.record_chunk_feedback(chunk_ids, score)

    async def get_low_quality_chunks(self, kb_id, threshold=0.3, limit=100):
        return await self._sq.get_low_quality_chunks(kb_id, threshold, limit)

    async def update_chunk_quality(self, chunk_id, score):
        await self._sq.update_chunk_quality(chunk_id, score)

    async def get_chunk_stats(self, kb_id):
        return await self._sq.get_chunk_stats(kb_id)

    # ── 门店画像（双写）──────────────────────────────────────────────────────

    async def upsert_store_profile(self, bra_id, metric="*",
                                   no_data_date=None, has_data_ts=None):
        await self._sq.upsert_store_profile(bra_id, metric, no_data_date, has_data_ts)
        await self._md.upsert_store_profile(bra_id, metric, no_data_date, has_data_ts)

    async def get_store_profile(self, bra_id):
        return await self._sq.get_store_profile(bra_id)

    async def all_store_profiles(self):
        return await self._sq.all_store_profiles() if hasattr(self._sq, "all_store_profiles") \
            else await self._md.all_store_profiles()

    async def is_known_no_data(self, bra_id, date_str):
        return await self._sq.is_known_no_data(bra_id, date_str)

    # ── BI 模板（双写）───────────────────────────────────────────────────────

    async def upsert_bi_template(self, template_id, intent_text, canonical_params, bra_id=""):
        await self._sq.upsert_bi_template(template_id, intent_text, canonical_params, bra_id)
        await self._md.upsert_bi_template(template_id, intent_text, canonical_params, bra_id)

    async def get_bi_templates(self, bra_id="", limit=200):
        return await self._sq.get_bi_templates(bra_id, limit)

    # ── 动作日志（双写）──────────────────────────────────────────────────────

    async def log_action(self, action_id, actor_name, target_type,
                         target_id="", description="", success=True, error=""):
        await self._sq.log_action(
            action_id, actor_name, target_type, target_id, description, success, error
        )
        await self._md.log_action(
            action_id, actor_name, target_type, target_id, description, success, error
        )

    async def recent_actions(self, hours=24):
        return await self._sq.recent_actions(hours)

    # ── 动态配置（双写）──────────────────────────────────────────────────────

    async def get_config(self, key, default=None):
        return await self._sq.get_config(key, default)

    async def set_config(self, key, value):
        await self._sq.set_config(key, value)
        await self._md.set_config(key, value)
