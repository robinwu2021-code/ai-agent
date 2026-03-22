"""
llm/call_logger.py — LLM 调用全链路日志

每次 LLM API 调用（chat / stream_chat / embed）都会产生三类事件：
  llm.<task>.request   — 调用前：请求参数（模型、消息数、工具列表、配置等）
  llm.<task>.response  — 成功后：响应参数（内容预览、token 用量、耗时等）
  llm.<task>.error     — 异常时：错误类型、错误信息、耗时

输出目标（可同时启用）：
  1. structlog / 控制台  — 由 LOG_LEVEL 控制可见性
  2. 本地 JSONL 文件     — RotatingFileHandler，每行一个 JSON 对象
     默认路径：logs/llm_calls.jsonl
     可通过 LLM_CALL_LOG_FILE='' 禁用文件写入

配置项（全部可通过 .env 或环境变量覆盖）：
  LLM_CALL_LOG_ENABLED      = true
  LLM_CALL_LOG_LEVEL        = DEBUG   # 控制台输出级别（DEBUG/INFO/WARNING）
  LLM_CALL_LOG_FILE         = logs/llm_calls.jsonl
  LLM_CALL_LOG_MAX_BYTES    = 10485760   # 单文件最大字节（默认 10 MB）
  LLM_CALL_LOG_BACKUP_COUNT = 5          # 保留的历史文件数
  LLM_CALL_LOG_MSG_PREVIEW  = 500        # 消息内容最多记录字符数
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# LLMCallLogger
# ─────────────────────────────────────────────────────────────────

class LLMCallLogger:
    """
    结构化日志记录器，同时写入 structlog（控制台）和本地 JSONL 文件。
    通过 init_call_logger() 初始化后，在引擎层调用各 log_* 方法。
    """

    def __init__(
        self,
        enabled:      bool = True,
        log_level:    str  = "DEBUG",
        file_path:    str  = "logs/llm_calls.jsonl",
        max_bytes:    int  = 10 * 1024 * 1024,
        backup_count: int  = 5,
        msg_preview:  int  = 500,
    ):
        self.enabled     = enabled
        self._log_level  = log_level.upper()
        self._msg_preview = msg_preview
        self._file_logger: logging.Logger | None = None

        if enabled and file_path:
            self._setup_file_logger(file_path, max_bytes, backup_count)
            log.info(
                "llm_call_logger.ready",
                file=file_path,
                console_level=log_level,
                msg_preview=msg_preview,
                max_bytes=max_bytes,
                backup_count=backup_count,
            )
        elif enabled:
            log.info("llm_call_logger.ready", file="(disabled)", console_level=log_level)

    # ── file handler setup ─────────────────────────────────────────

    def _setup_file_logger(self, path: str, max_bytes: int, backup_count: int) -> None:
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)

        handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("llm.call_file")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False  # 不向 root logger 传播，避免重复输出
        self._file_logger = logger

    # ── internal helpers ───────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _clip(self, text: str | None) -> str:
        if not text:
            return ""
        s = str(text)
        return s[:self._msg_preview] + ("…" if len(s) > self._msg_preview else "")

    def _emit(self, record: dict) -> None:
        """Write to structlog (console) + JSONL file."""
        if not self.enabled:
            return

        event    = record.get("event", "llm.event")
        _level   = record.pop("_level", "debug")
        _payload = {k: v for k, v in record.items() if k != "event"}

        # ── structlog (console) ──────────────────────────────────
        getattr(log, _level, log.debug)(event, **_payload)

        # ── JSONL file ───────────────────────────────────────────
        if self._file_logger:
            line = json.dumps({"event": event, **_payload}, ensure_ascii=False, default=str)
            self._file_logger.debug(line)

    # ── message extraction helpers ─────────────────────────────────

    @staticmethod
    def _extract_messages(messages: list) -> dict:
        """Extract loggable fields from a list of Message objects."""
        system_preview = ""
        last_user      = ""
        all_roles: list[str] = []

        for m in messages:
            role = getattr(m, "role", None)
            role_val = role.value if hasattr(role, "value") else str(role)
            all_roles.append(role_val)

            if role_val == "system":
                content = getattr(m, "content", "") or ""
                system_preview = str(content)
            elif role_val == "user":
                content = getattr(m, "content", "") or ""
                last_user = str(content)

        return {
            "message_count": len(messages),
            "roles":         all_roles,
            "system_preview": system_preview,
            "last_user_msg": last_user,
        }

    @staticmethod
    def _extract_tools(tools: list) -> dict:
        names = [getattr(t, "name", str(t)) for t in (tools or [])]
        return {"tool_count": len(names), "tool_names": names}

    @staticmethod
    def _extract_config(config: Any) -> dict:
        if config is None:
            return {}
        return {
            "temperature": getattr(config, "temperature", None),
            "max_tokens":  getattr(config, "max_tokens", None),
            "model_hint":  getattr(config, "model", None),
            "node_id":     getattr(config, "node_id", None),
        }

    # ── public logging API ─────────────────────────────────────────

    def log_request(
        self,
        engine:   str,
        alias:    str,
        model:    str,
        task:     str,
        messages: list,
        tools:    list,
        config:   Any,
    ) -> float:
        """Log a chat/plan/eval request. Returns t0 for elapsed calculation."""
        t0 = time.perf_counter()
        if not self.enabled:
            return t0

        msg_info  = self._extract_messages(messages)
        tool_info = self._extract_tools(tools)
        cfg_info  = self._extract_config(config)

        self._emit({
            "ts":            self._ts(),
            "event":         f"llm.{task}.request",
            "_level":        self._log_level.lower(),
            "engine":        engine,
            "alias":         alias,
            "model":         model,
            "task":          task,
            **msg_info,
            **tool_info,
            "config":        cfg_info,
            # clip long previews for file output
            "system_preview":  self._clip(msg_info["system_preview"]),
            "last_user_msg":   self._clip(msg_info["last_user_msg"]),
        })
        return t0

    def log_response(
        self,
        engine:   str,
        alias:    str,
        model:    str,
        task:     str,
        t0:       float,
        response: Any,
    ) -> None:
        """Log a successful chat/plan/eval response."""
        if not self.enabled:
            return

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        content    = getattr(response, "content", None) or ""
        tcs_raw    = getattr(response, "tool_calls", []) or []
        usage      = getattr(response, "usage", {}) or {}
        resp_model = getattr(response, "model", model) or model

        tc_list = [
            {
                "tool_name":   getattr(tc, "tool_name", str(tc)),
                "args_preview": self._clip(str(getattr(tc, "arguments", {}))),
            }
            for tc in tcs_raw
        ]

        self._emit({
            "ts":              self._ts(),
            "event":           f"llm.{task}.response",
            "_level":          self._log_level.lower(),
            "engine":          engine,
            "alias":           alias,
            "model":           resp_model,
            "task":            task,
            "elapsed_ms":      elapsed_ms,
            "content_len":     len(content),
            "content_preview": self._clip(content),
            "tool_calls":      tc_list,
            "usage":           usage,
        })

    def log_stream_start(
        self,
        engine: str,
        alias:  str,
        model:  str,
        task:   str,
        messages: list,
        tools:    list,
        config:   Any,
    ) -> float:
        """Log a stream_chat request start (also logs request params)."""
        t0 = time.perf_counter()
        if not self.enabled:
            return t0

        msg_info  = self._extract_messages(messages)
        tool_info = self._extract_tools(tools)
        cfg_info  = self._extract_config(config)

        self._emit({
            "ts":             self._ts(),
            "event":          f"llm.{task}.request",
            "_level":         self._log_level.lower(),
            "engine":         engine,
            "alias":          alias,
            "model":          model,
            "task":           task,
            **msg_info,
            **tool_info,
            "config":         cfg_info,
            "system_preview": self._clip(msg_info["system_preview"]),
            "last_user_msg":  self._clip(msg_info["last_user_msg"]),
        })
        return t0

    def log_stream_end(
        self,
        engine:      str,
        alias:       str,
        model:       str,
        task:        str,
        t0:          float,
        total_chars: int,
    ) -> None:
        """Log successful stream completion."""
        if not self.enabled:
            return
        self._emit({
            "ts":          self._ts(),
            "event":       f"llm.{task}.response",
            "_level":      self._log_level.lower(),
            "engine":      engine,
            "alias":       alias,
            "model":       model,
            "task":        task,
            "elapsed_ms":  round((time.perf_counter() - t0) * 1000, 1),
            "total_chars": total_chars,
            "streaming":   True,
        })

    def log_error(
        self,
        engine: str,
        alias:  str,
        model:  str,
        task:   str,
        t0:     float,
        exc:    Exception,
    ) -> None:
        """Log a failed call (always at WARNING level regardless of log_level)."""
        if not self.enabled:
            return
        self._emit({
            "ts":         self._ts(),
            "event":      f"llm.{task}.error",
            "_level":     "warning",
            "engine":     engine,
            "alias":      alias,
            "model":      model,
            "task":       task,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error":      str(exc),
            "error_type": type(exc).__name__,
        })


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────

_instance: LLMCallLogger | None = None


def get_call_logger() -> LLMCallLogger:
    """Return the module-level singleton (safe default: disabled)."""
    global _instance
    if _instance is None:
        _instance = LLMCallLogger(enabled=False)
    return _instance


def init_call_logger(
    enabled:      bool = True,
    log_level:    str  = "DEBUG",
    file_path:    str  = "logs/llm_calls.jsonl",
    max_bytes:    int  = 10 * 1024 * 1024,
    backup_count: int  = 5,
    msg_preview:  int  = 500,
) -> LLMCallLogger:
    """
    Initialize (or re-initialize) the global LLMCallLogger.
    Call once at server startup, e.g. from server.py or _build_container().
    """
    global _instance
    _instance = LLMCallLogger(
        enabled=enabled,
        log_level=log_level,
        file_path=file_path,
        max_bytes=max_bytes,
        backup_count=backup_count,
        msg_preview=msg_preview,
    )
    return _instance
