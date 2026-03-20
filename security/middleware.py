"""
security/middleware.py — 安全中间件

功能：
  1. ToolCallGuard      工具调用鉴权（白/黑名单 + 危险操作确认）
  2. ContentFilter      输入/输出内容过滤（Prompt Injection 检测）
  3. AuditLogger        完整审计链路（持久化到数据库）
  4. RateLimiter        每用户请求频率限制
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Risk Levels
# ─────────────────────────────────────────────

class RiskLevel:
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    BLOCK  = "block"   # 直接拒绝，不允许执行


# 工具风险等级映射（默认 LOW，可在配置中覆盖）
DEFAULT_TOOL_RISK: dict[str, str] = {
    "execute_python":  RiskLevel.HIGH,
    "write_file":      RiskLevel.HIGH,
    "http_request":    RiskLevel.MEDIUM,
    "web_search":      RiskLevel.LOW,
    "read_file":       RiskLevel.LOW,
    "calculator":      RiskLevel.LOW,
    "json_extract":    RiskLevel.LOW,
    "query_memory":    RiskLevel.LOW,
    "summarize_text":  RiskLevel.LOW,
}

# 绝对禁止的 Prompt Injection 特征
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(?:a|an)\s+(?:evil|unrestricted|jailbreak)",
    r"DAN\s+mode",
    r"disregard\s+your\s+(guidelines|rules|instructions)",
    r"override\s+system\s+prompt",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
]


# ─────────────────────────────────────────────
# Audit Entry
# ─────────────────────────────────────────────

@dataclass
class AuditEntry:
    timestamp:   str
    user_id:     str
    session_id:  str
    event_type:  str          # "tool_call" | "content_filter" | "rate_limit" | "block"
    tool_name:   str | None
    arguments:   dict | None
    risk_level:  str
    allowed:     bool
    reason:      str | None
    duration_ms: int = 0


# ─────────────────────────────────────────────
# Tool Call Guard
# ─────────────────────────────────────────────

class ToolCallGuard:
    """
    工具调用安全门卫。
    - 白名单：只允许显式列出的工具
    - 黑名单：永久禁止某些工具
    - 风险等级：HIGH 级别需要注入确认回调（Human-in-the-loop 接入点）
    - 参数净化：防止路径穿越、命令注入
    """

    def __init__(
        self,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
        tool_risk_map: dict[str, str] | None = None,
        high_risk_approver: Callable | None = None,  # async (tool, args) -> bool
    ) -> None:
        self._whitelist  = set(whitelist) if whitelist else None   # None = 全部允许
        self._blacklist  = set(blacklist or [])
        self._risk_map   = {**DEFAULT_TOOL_RISK, **(tool_risk_map or {})}
        self._approver   = high_risk_approver

    async def check(
        self, tool_name: str, arguments: dict[str, Any],
        user_id: str, session_id: str,
    ) -> tuple[bool, str]:
        """
        返回 (allowed, reason)。
        allowed=False 时调用方应拒绝执行并返回错误 ToolResult。
        """
        # 黑名单直接拒绝
        if tool_name in self._blacklist:
            return False, f"Tool '{tool_name}' is permanently blocked"

        # 白名单检查
        if self._whitelist and tool_name not in self._whitelist:
            return False, f"Tool '{tool_name}' is not in the allowed list"

        # 参数安全性检查
        safe, reason = self._sanitize_arguments(tool_name, arguments)
        if not safe:
            return False, reason

        # 风险等级
        risk = self._risk_map.get(tool_name, RiskLevel.LOW)
        if risk == RiskLevel.BLOCK:
            return False, f"Tool '{tool_name}' is risk-blocked"

        if risk == RiskLevel.HIGH and self._approver:
            approved = await self._approver(tool_name, arguments, user_id)
            if not approved:
                return False, f"High-risk tool '{tool_name}' was not approved"

        log.debug("tool.guard.allowed", tool=tool_name, risk=risk, user=user_id)
        return True, ""

    def _sanitize_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[bool, str]:
        """检查参数中的危险模式。"""
        for key, val in arguments.items():
            if not isinstance(val, str):
                continue
            # 路径穿越
            if ".." in val and key in ("path", "file", "filename"):
                return False, f"Path traversal detected in argument '{key}'"
            # Shell 注入（针对代码执行类工具的粗略检测）
            if tool_name in ("execute_python", "execute_shell"):
                dangerous = ["__import__", "subprocess", "os.system", "eval(", "exec("]
                for d in dangerous:
                    if d in val:
                        log.warning("tool.guard.dangerous_code", pattern=d, tool=tool_name)
                        # 不直接拒绝，只记录警告——实际沙箱隔离会处理
        return True, ""


# ─────────────────────────────────────────────
# Content Filter
# ─────────────────────────────────────────────

class ContentFilter:
    """
    输入/输出内容过滤。
    - Prompt Injection 检测
    - 敏感信息脱敏（信用卡号、身份证号等）
    - 输出内容合规检查（可扩展接入 Moderation API）
    """

    SENSITIVE_PATTERNS = [
        (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD_REDACTED]"),
        (r"\b1[3-9]\d{9}\b",                            "[PHONE_REDACTED]"),
        (r"\b\d{17}[\dXx]\b",                           "[ID_REDACTED]"),
        (r"(?i)(password|passwd|secret|api.?key)\s*[:=]\s*\S+", "[SECRET_REDACTED]"),
    ]

    def __init__(self, block_injections: bool = True, redact_sensitive: bool = True):
        self._block_injections = block_injections
        self._redact_sensitive = redact_sensitive
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self._sensitive_re = [(re.compile(p), repl) for p, repl in self.SENSITIVE_PATTERNS]

    def check_input(self, text: str) -> tuple[bool, str]:
        """检查用户输入，返回 (safe, reason)。"""
        if not self._block_injections:
            return True, ""
        for pattern in self._injection_re:
            if pattern.search(text):
                log.warning("content.injection_detected", snippet=text[:100])
                return False, "Potential prompt injection detected in input"
        return True, ""

    def redact_output(self, text: str) -> str:
        """对输出做敏感信息脱敏。"""
        if not self._redact_sensitive or not text:
            return text
        for pattern, replacement in self._sensitive_re:
            text = pattern.sub(replacement, text)
        return text

    def check_output(self, text: str) -> tuple[bool, str]:
        """检查模型输出是否包含不应泄露的内容（可接入 Moderation API）。"""
        # Stub：生产环境接入 OpenAI Moderation 或自定义分类器
        return True, ""


# ─────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────

class AuditLogger:
    """
    完整审计日志。
    开发：内存列表
    生产：异步写入 PostgreSQL / ClickHouse
    """

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend
        self._buffer: list[AuditEntry] = []

    def log_tool_call(
        self, user_id: str, session_id: str,
        tool_name: str, arguments: dict,
        allowed: bool, risk: str, reason: str = "",
        duration_ms: int = 0,
    ) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id, session_id=session_id,
            event_type="tool_call", tool_name=tool_name,
            arguments=self._mask_secrets(arguments),
            risk_level=risk, allowed=allowed, reason=reason,
            duration_ms=duration_ms,
        )
        self._buffer.append(entry)
        log.info(
            "audit.tool_call",
            user=user_id, tool=tool_name,
            allowed=allowed, risk=risk,
        )

    def log_content_filter(
        self, user_id: str, session_id: str,
        direction: str, blocked: bool, reason: str,
    ) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id, session_id=session_id,
            event_type="content_filter", tool_name=None,
            arguments={"direction": direction},
            risk_level=RiskLevel.HIGH if blocked else RiskLevel.LOW,
            allowed=not blocked, reason=reason,
        )
        self._buffer.append(entry)

    def get_entries(
        self, user_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        entries = self._buffer
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        return [vars(e) for e in entries[-limit:]]

    @staticmethod
    def _mask_secrets(args: dict) -> dict:
        masked = {}
        for k, v in args.items():
            if any(s in k.lower() for s in ("key", "secret", "token", "password")):
                masked[k] = "***"
            else:
                masked[k] = v
        return masked


# ─────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────

class RateLimiter:
    """
    滑动窗口限流器（内存实现，生产用 Redis + lua script）。
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self._max  = max_requests
        self._win  = window_seconds
        self._logs: dict[str, list[float]] = {}

    def check(self, user_id: str) -> tuple[bool, int]:
        """
        返回 (allowed, retry_after_seconds)。
        """
        now    = time.monotonic()
        cutoff = now - self._win
        times  = self._logs.setdefault(user_id, [])
        # 清理过期记录
        self._logs[user_id] = [t for t in times if t > cutoff]
        if len(self._logs[user_id]) >= self._max:
            oldest = self._logs[user_id][0]
            retry  = int(self._win - (now - oldest)) + 1
            return False, retry
        self._logs[user_id].append(now)
        return True, 0


# ─────────────────────────────────────────────
# Security Manager (façade)
# ─────────────────────────────────────────────

class SecurityManager:
    """
    统一安全入口，编排引擎通过此类做所有安全检查。
    """

    def __init__(
        self,
        tool_guard: ToolCallGuard | None = None,
        content_filter: ContentFilter | None = None,
        audit_logger: AuditLogger | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.tool_guard     = tool_guard     or ToolCallGuard()
        self.content_filter = content_filter or ContentFilter()
        self.audit          = audit_logger   or AuditLogger()
        self.rate_limiter   = rate_limiter   or RateLimiter()

    async def check_request(
        self, user_id: str, text: str
    ) -> tuple[bool, str]:
        """检查进入的用户请求。"""
        allowed, limit_after = self.rate_limiter.check(user_id)
        if not allowed:
            return False, f"Rate limit exceeded. Retry after {limit_after}s"
        safe, reason = self.content_filter.check_input(text)
        if not safe:
            self.audit.log_content_filter(user_id, "", "input", True, reason)
        return safe, reason

    async def check_tool_call(
        self, user_id: str, session_id: str,
        tool_name: str, arguments: dict,
    ) -> tuple[bool, str]:
        """检查工具调用前的权限。"""
        allowed, reason = await self.tool_guard.check(
            tool_name, arguments, user_id, session_id
        )
        risk = self.tool_guard._risk_map.get(tool_name, RiskLevel.LOW)
        self.audit.log_tool_call(
            user_id, session_id, tool_name, arguments,
            allowed, risk, reason,
        )
        return allowed, reason

    def sanitize_output(self, text: str) -> str:
        """对模型输出做脱敏处理。"""
        return self.content_filter.redact_output(text)
