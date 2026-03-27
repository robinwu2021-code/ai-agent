"""
evolution/actors/kb_injector.py — 知识库 Q&A 自动注入执行器

从 signals 表中取出未注入的 qa_pair 信号，
将 Q&A 对以文档形式写入对应知识库，标记已注入。
"""
from __future__ import annotations

import json
import time
import uuid

import structlog

from evolution.actors.base import BaseActor
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution.actors.kb_injector")


class KbInjectorActor(BaseActor):
    """
    初始化时需传入 kb（PersistentKnowledgeBase 实例），
    通过 **kwargs 方式接收，保持基类签名兼容。
    """

    def __init__(self, store: EvolutionStore, config: dict | None = None,
                 kb=None, **kwargs) -> None:
        super().__init__(store, config, **kwargs)
        self._kb = kb

    async def act(self) -> dict:
        if self._kb is None:
            log.warning("evolution.kb_injector.no_kb")
            return {"injected": 0, "skipped": 0, "reason": "kb not configured"}

        # 取出所有未注入的 qa_pair
        pending = await self._store._fetchall_by(
            """SELECT * FROM signals WHERE signal_type='qa_pair'
               AND json_extract(payload, '$.injected') = 0
               ORDER BY quality DESC LIMIT 50"""
        )

        injected = 0
        failed   = 0

        for row in pending:
            try:
                payload = json.loads(row["payload"])
                kb_id   = row.get("kb_id") or payload.get("kb_id") or ""
                query   = payload.get("query", "")
                answer  = payload.get("answer", "")

                if not kb_id or not query or not answer:
                    continue

                # 构造注入内容
                doc_text = (
                    f"问：{query}\n\n答：{answer}\n\n"
                    f"[自动生成 | session: {payload.get('session_id','')} | "
                    f"质量分: {row.get('quality', 0):.2f}]"
                )

                # 调用知识库 API 注入
                # 适配不同的 KB 接口（ingest_text / add_document）
                if hasattr(self._kb, "ingest_text"):
                    await self._kb.ingest_text(
                        kb_id    = kb_id,
                        text     = doc_text,
                        metadata = {
                            "source":     "auto_evolution",
                            "qa_pair_id": row["signal_id"],
                            "query":      query[:200],
                        },
                    )
                elif hasattr(self._kb, "add_text"):
                    await self._kb.add_text(kb_id=kb_id, text=doc_text)
                else:
                    log.warning("evolution.kb_injector.unsupported_kb_api",
                                kb_type=type(self._kb).__name__)
                    break

                # 标记已注入
                payload["injected"]    = True
                payload["injected_at"] = time.time()
                await self._store._exec(
                    "UPDATE signals SET payload=? WHERE signal_id=?",
                    (json.dumps(payload, ensure_ascii=False), row["signal_id"]),
                )

                await self._store.log_action(
                    action_id   = uuid.uuid4().hex[:16],
                    actor_name  = "kb_injector",
                    target_type = "chunk",
                    target_id   = row["signal_id"],
                    description = f"Injected Q&A into kb={kb_id}: {query[:60]}",
                )
                injected += 1

            except Exception as exc:
                log.error("evolution.kb_injector.failed",
                          signal_id=row.get("signal_id"), error=str(exc))
                failed += 1

        summary = {"injected": injected, "failed": failed, "pending": len(pending)}
        log.info("evolution.kb_injector.done", **summary)
        return summary
