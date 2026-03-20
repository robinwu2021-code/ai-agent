"""
hitl/checkpoint.py — Human-in-the-loop 检查点系统

功能：
  - CheckpointManager   在高风险操作前暂停任务等待人工确认
  - ApprovalRequest     待确认请求的数据结构
  - ConsoleApprover     终端交互式确认（开发用）
  - APIApprover         通过 HTTP 回调等待确认（生产用）

使用方式（在编排引擎中集成）：
    hitl = CheckpointManager(approver=ConsoleApprover())

    # 在工具调用前
    approved = await hitl.request_approval(
        task_id=task.id,
        user_id=task.user_id,
        tool_name="write_file",
        arguments={"path": "/etc/hosts", "content": "..."},
        reason="写入系统文件是高风险操作",
    )
    if not approved:
        raise PermissionError("User rejected the operation")
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import structlog

log = structlog.get_logger(__name__)


class ApprovalStatus(str, Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED  = "expired"
    MODIFIED = "modified"   # 人工修改了参数后批准


@dataclass
class ApprovalRequest:
    id:           str = field(default_factory=lambda: f"appr_{uuid.uuid4().hex[:8]}")
    task_id:      str = ""
    user_id:      str = ""
    tool_name:    str = ""
    arguments:    dict = field(default_factory=dict)
    reason:       str = ""
    status:       ApprovalStatus = ApprovalStatus.PENDING
    modified_args: dict | None = None   # 人工修改后的参数
    feedback:     str = ""
    created_at:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at:  str | None = None
    timeout_sec:  int = 300   # 5 分钟超时


@dataclass
class CheckpointEvent:
    """编排引擎 yield 给调用方的暂停事件。"""
    type:       str = "checkpoint"
    request_id: str = ""
    task_id:    str = ""
    tool_name:  str = ""
    arguments:  dict = field(default_factory=dict)
    reason:     str = ""
    message:    str = "等待人工确认，请通过 API 提交批准或拒绝"


# ─────────────────────────────────────────────
# Approver Protocol
# ─────────────────────────────────────────────

class ConsoleApprover:
    """终端交互式审批，开发和调试用。"""

    async def request(self, req: ApprovalRequest) -> ApprovalRequest:
        print(f"\n{'='*50}")
        print(f"  [HITL] 需要人工确认")
        print(f"  工具: {req.tool_name}")
        print(f"  原因: {req.reason}")
        print(f"  参数: {req.arguments}")
        print(f"  输入 y/n/m(修改参数): ", end="", flush=True)

        try:
            loop    = asyncio.get_event_loop()
            answer  = await asyncio.wait_for(
                loop.run_in_executor(None, input), timeout=req.timeout_sec
            )
            answer  = answer.strip().lower()
        except asyncio.TimeoutError:
            req.status = ApprovalStatus.EXPIRED
            log.warning("hitl.timeout", request_id=req.id)
            return req

        if answer == "y":
            req.status = ApprovalStatus.APPROVED
        elif answer == "m":
            print("  请输入修改后的 JSON 参数: ", end="", flush=True)
            try:
                import json
                modified = await asyncio.wait_for(
                    loop.run_in_executor(None, input), timeout=60
                )
                req.modified_args = json.loads(modified)
                req.status = ApprovalStatus.MODIFIED
            except Exception:
                req.status = ApprovalStatus.REJECTED
        else:
            req.status = ApprovalStatus.REJECTED

        req.resolved_at = datetime.now(timezone.utc).isoformat()
        log.info("hitl.resolved", request_id=req.id, status=req.status.value)
        return req


class APIApprover:
    """
    通过 HTTP 回调等待人工审批（生产用）。
    工作流：
      1. 创建 ApprovalRequest 并存储
      2. 向 webhook_url 发送通知
      3. 轮询或等待 resolve() 被调用
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        self._webhook      = webhook_url
        self._poll_interval = poll_interval
        self._pending:  dict[str, ApprovalRequest] = {}
        self._resolved: dict[str, asyncio.Event]   = {}

    async def request(self, req: ApprovalRequest) -> ApprovalRequest:
        self._pending[req.id]  = req
        self._resolved[req.id] = asyncio.Event()

        # 发送 webhook 通知
        if self._webhook:
            await self._notify(req)

        log.info("hitl.waiting", request_id=req.id, tool=req.tool_name)

        try:
            await asyncio.wait_for(
                self._resolved[req.id].wait(),
                timeout=req.timeout_sec,
            )
        except asyncio.TimeoutError:
            req.status     = ApprovalStatus.EXPIRED
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            log.warning("hitl.expired", request_id=req.id)

        self._resolved.pop(req.id, None)
        return self._pending.pop(req.id, req)

    def resolve(
        self, request_id: str,
        approved: bool,
        modified_args: dict | None = None,
        feedback: str = "",
    ) -> bool:
        """由 API 端点调用，解除等待。"""
        req = self._pending.get(request_id)
        if not req:
            return False
        if modified_args:
            req.status = ApprovalStatus.MODIFIED
            req.modified_args = modified_args
        elif approved:
            req.status = ApprovalStatus.APPROVED
        else:
            req.status = ApprovalStatus.REJECTED
        req.feedback    = feedback
        req.resolved_at = datetime.now(timezone.utc).isoformat()

        if ev := self._resolved.get(request_id):
            ev.set()
        log.info("hitl.resolved_via_api", request_id=request_id, status=req.status.value)
        return True

    def list_pending(self) -> list[ApprovalRequest]:
        return list(self._pending.values())

    async def _notify(self, req: ApprovalRequest) -> None:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(self._webhook, json={
                    "request_id": req.id,
                    "task_id":    req.task_id,
                    "user_id":    req.user_id,
                    "tool":       req.tool_name,
                    "arguments":  req.arguments,
                    "reason":     req.reason,
                }, timeout=5)
        except Exception as e:
            log.warning("hitl.webhook_failed", error=str(e))


# ─────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────

class CheckpointManager:
    """
    编排引擎集成入口。
    与 SecurityManager.ToolCallGuard 配合：
      - HIGH 风险工具 → 触发 checkpoint
      - 返回最终参数（人工可能修改过）或 None（拒绝）
    """

    HIGH_RISK_TOOLS = {
        "write_file", "execute_python", "http_request",
        "send_email", "delete_file", "deploy",
    }

    def __init__(
        self,
        approver: Any = None,
        auto_approve_low_risk: bool = True,
        high_risk_tools: set[str] | None = None,
    ) -> None:
        self._approver = approver or ConsoleApprover()
        self._auto_approve = auto_approve_low_risk
        self._high_risk    = high_risk_tools or self.HIGH_RISK_TOOLS
        self._history: list[ApprovalRequest] = []

    async def maybe_checkpoint(
        self,
        task_id: str,
        user_id: str,
        tool_name: str,
        arguments: dict,
        risk_level: str = "low",
    ) -> tuple[bool, dict]:
        """
        决定是否需要人工确认。
        返回 (proceed, final_arguments)。
        proceed=False 表示操作被拒绝，final_arguments 可能是修改后的版本。
        """
        if tool_name not in self._high_risk and risk_level != "high":
            return True, arguments

        reason = self._build_reason(tool_name, arguments)
        req    = ApprovalRequest(
            task_id=task_id, user_id=user_id,
            tool_name=tool_name, arguments=arguments,
            reason=reason,
        )

        resolved = await self._approver.request(req)
        self._history.append(resolved)

        if resolved.status == ApprovalStatus.APPROVED:
            return True, arguments
        elif resolved.status == ApprovalStatus.MODIFIED:
            return True, resolved.modified_args or arguments
        else:
            return False, arguments

    def get_pending(self) -> list[ApprovalRequest]:
        if isinstance(self._approver, APIApprover):
            return self._approver.list_pending()
        return []

    def resolve(self, request_id: str, approved: bool, **kwargs) -> bool:
        if isinstance(self._approver, APIApprover):
            return self._approver.resolve(request_id, approved, **kwargs)
        return False

    def get_history(self) -> list[dict]:
        return [vars(r) for r in self._history]

    @staticmethod
    def _build_reason(tool_name: str, arguments: dict) -> str:
        reasons = {
            "write_file":      f"将写入文件: {arguments.get('path', '?')}",
            "execute_python":  "将执行 Python 代码（可访问系统资源）",
            "http_request":    f"将发送 HTTP 请求到: {arguments.get('url', '?')}",
            "send_email":      f"将发送邮件给: {arguments.get('to', '?')}",
            "delete_file":     f"将删除文件: {arguments.get('path', '?')}",
        }
        return reasons.get(tool_name, f"将执行高风险操作: {tool_name}")
