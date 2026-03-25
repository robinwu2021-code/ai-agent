"""
rag/files/models.py — 文件全生命周期数据模型

三层结构：
  KBFile          — 文件当前状态快照（一个文件一条记录）
  KBFileVersion   — 版本历史（每次上传/修改一条，含摘要和差异描述）
  KBFileAuditLog  — 操作审计日志（append-only，不可变）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FileStatus(str, Enum):
    UPLOADING  = "uploading"    # 上传中
    PROCESSING = "processing"   # 解析/索引中
    INDEXED    = "indexed"      # 已完成索引
    ERROR      = "error"        # 处理失败
    DELETED    = "deleted"      # 已软删除


class ChangeType(str, Enum):
    CREATED          = "created"           # 首次上传
    CONTENT_UPDATED  = "content_updated"   # 内容变更（hash 不同）
    METADATA_UPDATED = "metadata_updated"  # 仅元数据变更（无需重索引）
    REINDEXED        = "reindexed"         # 强制重新索引（内容不变）
    RESTORED         = "restored"          # 从历史版本恢复


class OperationType(str, Enum):
    UPLOAD            = "upload"
    UPDATE            = "update"
    DELETE            = "delete"
    RESTORE           = "restore"
    REINDEX           = "reindex"
    MOVE              = "move"
    PERMISSION_CHANGE = "permission_change"
    SHARE             = "share"


@dataclass
class OperatorContext:
    """操作人上下文，注入到每次文件操作。"""
    user_id: str
    user_name: str = ""
    ip_address: str = ""
    user_agent: str = ""


# ── 文件注册表（当前状态快照）────────────────────────────────────────────────

@dataclass
class KBFile:
    id: str                           # UUID
    workspace_id: str
    kb_id: str
    directory_id: str                 # 所属目录

    name: str                         # 用户可见名称（可重命名）
    original_name: str                # 首次上传时锁定的原始文件名
    mime_type: str = ""
    file_size: int = 0
    file_hash: str = ""               # SHA-256，用于判断内容是否变更

    status: FileStatus = FileStatus.UPLOADING
    current_version_id: str | None = None
    version_count: int = 1

    # 冗余最新版本数据，避免频繁 JOIN
    summary: str = ""
    chunk_count: int = 0
    token_count: int = 0

    created_at: float = 0.0
    created_by: str = ""              # 上传人 user_id
    updated_at: float = 0.0
    updated_by: str = ""              # 最后修改人 user_id

    metadata: dict = field(default_factory=dict)


# ── 版本历史（每次上传/修改一条）────────────────────────────────────────────

@dataclass
class KBFileVersion:
    id: str                           # UUID
    file_id: str
    version_number: int               # 1, 2, 3 … 递增

    file_path: str = ""               # 该版本原始文件存储路径（用于恢复）
    file_size: int = 0
    file_hash: str = ""

    change_type: ChangeType = ChangeType.CREATED
    change_description: str = ""     # 用户填写的变更说明（可选）

    # LLM 自动生成
    summary: str = ""                # 本版本文档内容摘要
    diff_summary: str = ""           # 与上一版本的差异摘要

    chunk_count: int = 0
    token_count: int = 0
    indexing_duration_ms: int = 0
    processing_error: str = ""

    created_at: float = 0.0
    created_by: str = ""             # 本次操作的 user_id


# ── 审计日志（append-only，不可变）──────────────────────────────────────────

@dataclass
class KBFileAuditLog:
    id: str                           # UUID
    file_id: str
    version_id: str | None = None

    operation: OperationType = OperationType.UPLOAD
    operator_id: str = ""
    operator_name: str = ""

    detail: dict = field(default_factory=dict)
    # 示例：
    #   UPLOAD: {file_name, size}
    #   UPDATE: {change_type, description}
    #   MOVE:   {from_directory, to_directory}
    #   PERMISSION_CHANGE: {added: [...], removed: [...]}

    ip_address: str = ""
    user_agent: str = ""
    created_at: float = 0.0
