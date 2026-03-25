"""
rag/files/manager.py — 文件全生命周期管理

职责：上传、更新、删除、版本恢复、元数据修改、历史查询。
不感知索引细节——索引由 IngestionPipeline 完成后回调更新版本记录。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path

import structlog

from rag.files.models import (
    ChangeType, FileStatus, KBFile, KBFileAuditLog,
    KBFileVersion, OperationType, OperatorContext,
)
from rag.files.store import KBFileStore

log = structlog.get_logger(__name__)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _mime_type(path: str) -> str:
    import mimetypes
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


class KBFileManager:
    """文件全生命周期管理器。"""

    def __init__(
        self,
        store: KBFileStore,
        storage_base: str = "./data/files",
        keep_versions: int = 10,
    ) -> None:
        self._store = store
        self._storage_base = Path(storage_base)
        self._keep_versions = keep_versions

    def _storage_path(self, file_id: str, version: int, original_name: str) -> Path:
        """计算某版本原始文件的存储路径。"""
        ext = Path(original_name).suffix
        return self._storage_base / file_id / f"v{version}{ext}"

    def _copy_to_storage(self, src: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    async def _write_audit(
        self,
        file_id: str,
        version_id: str | None,
        operation: OperationType,
        operator: OperatorContext,
        detail: dict,
    ) -> None:
        entry = KBFileAuditLog(
            id=uuid.uuid4().hex,
            file_id=file_id,
            version_id=version_id,
            operation=operation,
            operator_id=operator.user_id,
            operator_name=operator.user_name,
            detail=detail,
            ip_address=operator.ip_address,
            user_agent=operator.user_agent,
            created_at=time.time(),
        )
        await self._store.append_audit(entry)

    # ── 上传新文件 ────────────────────────────────────────────────────

    async def upload(
        self,
        file_path: str,
        workspace_id: str,
        kb_id: str,
        directory_id: str,
        operator: OperatorContext,
        change_description: str = "",
        file_id: str | None = None,
    ) -> KBFile:
        """
        上传新文件。
        返回 KBFile（status=PROCESSING），由 IngestionPipeline 完成后更新为 INDEXED。
        """
        path = Path(file_path)
        file_hash = await asyncio.to_thread(_sha256, file_path)
        fid = file_id or uuid.uuid4().hex

        # 复制到持久化存储目录
        dst = self._storage_path(fid, 1, path.name)
        await asyncio.to_thread(self._copy_to_storage, file_path, dst)

        now = time.time()
        kb_file = KBFile(
            id=fid,
            workspace_id=workspace_id,
            kb_id=kb_id,
            directory_id=directory_id,
            name=path.stem,
            original_name=path.name,
            mime_type=_mime_type(file_path),
            file_size=path.stat().st_size,
            file_hash=file_hash,
            status=FileStatus.PROCESSING,
            version_count=1,
            created_at=now,
            created_by=operator.user_id,
            updated_at=now,
            updated_by=operator.user_id,
        )
        await self._store.save_file(kb_file)

        version = KBFileVersion(
            id=uuid.uuid4().hex,
            file_id=fid,
            version_number=1,
            file_path=str(dst),
            file_size=kb_file.file_size,
            file_hash=file_hash,
            change_type=ChangeType.CREATED,
            change_description=change_description,
            created_at=now,
            created_by=operator.user_id,
        )
        await self._store.save_version(version)
        kb_file.current_version_id = version.id
        await self._store.save_file(kb_file)

        await self._write_audit(
            fid, version.id, OperationType.UPLOAD, operator,
            {"file_name": path.name, "size": kb_file.file_size},
        )
        log.info("file_manager.uploaded", file_id=fid, name=path.name, kb_id=kb_id)
        return kb_file

    # ── 更新文件内容 ──────────────────────────────────────────────────

    async def update(
        self,
        file_id: str,
        new_file_path: str,
        operator: OperatorContext,
        change_description: str = "",
    ) -> KBFile:
        """
        更新文件内容。
        - hash 相同 → REINDEXED（内容未变，仅重跑索引）
        - hash 不同 → CONTENT_UPDATED（新版本）
        """
        kb_file = await self._store.get_file(file_id)
        if kb_file is None:
            raise ValueError(f"文件不存在: {file_id!r}")

        new_hash = await asyncio.to_thread(_sha256, new_file_path)
        change_type = (
            ChangeType.REINDEXED if new_hash == kb_file.file_hash
            else ChangeType.CONTENT_UPDATED
        )

        new_version_num = kb_file.version_count + 1
        dst = self._storage_path(file_id, new_version_num, kb_file.original_name)
        await asyncio.to_thread(self._copy_to_storage, new_file_path, dst)

        now = time.time()
        kb_file.file_hash = new_hash
        kb_file.file_size = Path(new_file_path).stat().st_size
        kb_file.status = FileStatus.PROCESSING
        kb_file.version_count = new_version_num
        kb_file.updated_at = now
        kb_file.updated_by = operator.user_id
        await self._store.save_file(kb_file)

        version = KBFileVersion(
            id=uuid.uuid4().hex,
            file_id=file_id,
            version_number=new_version_num,
            file_path=str(dst),
            file_size=kb_file.file_size,
            file_hash=new_hash,
            change_type=change_type,
            change_description=change_description,
            created_at=now,
            created_by=operator.user_id,
        )
        await self._store.save_version(version)
        kb_file.current_version_id = version.id
        await self._store.save_file(kb_file)

        await self._write_audit(
            file_id, version.id, OperationType.UPDATE, operator,
            {"change_type": change_type.value, "description": change_description},
        )
        await self._prune_old_versions(file_id)
        return kb_file

    # ── 更新元数据（不触发重索引）────────────────────────────────────

    async def update_metadata(
        self,
        file_id: str,
        updates: dict,               # 可更新: name, directory_id, metadata
        operator: OperatorContext,
    ) -> KBFile:
        kb_file = await self._store.get_file(file_id)
        if kb_file is None:
            raise ValueError(f"文件不存在: {file_id!r}")

        allowed = {"name", "directory_id", "metadata"}
        old_vals = {k: getattr(kb_file, k) for k in updates if k in allowed}
        for k, v in updates.items():
            if k in allowed:
                setattr(kb_file, k, v)

        now = time.time()
        kb_file.updated_at = now
        kb_file.updated_by = operator.user_id
        await self._store.save_file(kb_file)

        version = KBFileVersion(
            id=uuid.uuid4().hex,
            file_id=file_id,
            version_number=kb_file.version_count,
            file_hash=kb_file.file_hash,
            change_type=ChangeType.METADATA_UPDATED,
            created_at=now,
            created_by=operator.user_id,
        )
        await self._store.save_version(version)
        await self._write_audit(
            file_id, version.id, OperationType.UPDATE, operator,
            {"changed_fields": list(old_vals.keys()), "old": old_vals,
             "new": {k: updates[k] for k in old_vals}},
        )
        return kb_file

    # ── 软删除 ────────────────────────────────────────────────────────

    async def delete(self, file_id: str, operator: OperatorContext) -> None:
        await self._store.soft_delete_file(file_id)
        await self._write_audit(file_id, None, OperationType.DELETE, operator, {})
        log.info("file_manager.deleted", file_id=file_id)

    # ── 版本恢复 ──────────────────────────────────────────────────────

    async def restore_version(
        self,
        file_id: str,
        version_id: str,
        operator: OperatorContext,
    ) -> KBFile:
        target = await self._store.get_version(version_id)
        if target is None:
            raise ValueError(f"版本不存在: {version_id!r}")
        await self._write_audit(
            file_id, version_id, OperationType.RESTORE, operator,
            {"restored_version": target.version_number},
        )
        return await self.update(
            file_id, target.file_path, operator,
            change_description=f"恢复到版本 {target.version_number}",
        )

    # ── 版本完成回调（由 IngestionPipeline 调用）─────────────────────

    async def on_version_indexed(
        self,
        file_id: str,
        version_id: str,
        summary: str,
        diff_summary: str,
        chunk_count: int,
        token_count: int,
        indexing_duration_ms: int,
        error: str = "",
    ) -> None:
        """流水线完成索引后回调，更新版本记录和文件快照。"""
        version = await self._store.get_version(version_id)
        if version is None:
            return

        version.summary = summary
        version.diff_summary = diff_summary
        version.chunk_count = chunk_count
        version.token_count = token_count
        version.indexing_duration_ms = indexing_duration_ms
        version.processing_error = error
        await self._store.save_version(version)

        kb_file = await self._store.get_file(file_id)
        if kb_file is None:
            return

        kb_file.status = FileStatus.ERROR if error else FileStatus.INDEXED
        kb_file.summary = summary
        kb_file.chunk_count = chunk_count
        kb_file.token_count = token_count
        kb_file.updated_at = time.time()
        await self._store.save_file(kb_file)

    # ── 查询接口 ──────────────────────────────────────────────────────

    async def get_history(
        self, file_id: str, limit: int = 20, offset: int = 0
    ) -> list[KBFileVersion]:
        return await self._store.list_versions(file_id, limit, offset)

    async def get_audit_log(
        self,
        file_id: str | None = None,
        kb_id: str | None = None,
        operator_id: str | None = None,
        operation: OperationType | None = None,
        since: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KBFileAuditLog]:
        return await self._store.query_audit(
            file_id=file_id, kb_id=kb_id,
            operator_id=operator_id, operation=operation,
            since=since, limit=limit, offset=offset,
        )

    async def list_files(
        self,
        kb_id: str | None = None,
        directory_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[KBFile]:
        return await self._store.list_files(
            kb_id=kb_id, directory_id=directory_id, workspace_id=workspace_id
        )

    # ── 清理旧版本文件 ────────────────────────────────────────────────

    async def _prune_old_versions(self, file_id: str) -> None:
        if self._keep_versions <= 0:
            return
        versions = await self._store.list_versions(file_id, limit=200)
        to_delete = versions[self._keep_versions:]
        for v in to_delete:
            p = Path(v.file_path)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


# asyncio import needed for to_thread calls in upload/update
import asyncio
