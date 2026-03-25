"""
workspace — 多用户工作区系统

模块说明：
  models.py   数据模型（Workspace / Project / Member / MemoryEntry）
  store.py    持久化（SQLiteWorkspaceStore / MySQLWorkspaceStore）
  manager.py  业务逻辑（CRUD + 权限校验）
  memory.py   WorkspaceAwareLTM（4 层混合记忆读写）

快速启动（SQLite）：
    from workspace.store   import SQLiteWorkspaceStore
    from workspace.manager import WorkspaceManager
    from workspace.memory  import WorkspaceAwareLTM

    store   = SQLiteWorkspaceStore("workspace.db")
    await store.initialize()
    manager = WorkspaceManager(store)
    ltm     = WorkspaceAwareLTM(store)

    ws   = await manager.create_workspace("我的公司", creator_id="alice")
    proj = await manager.create_project(ws.workspace_id, "alice", "营销项目")

    # 写入 3 种范围的记忆
    await ltm.write_personal("alice", "偏好简洁文案风格")
    await ltm.write_project(ws.workspace_id, proj.project_id, "alice", "Q3活动ROI=3.2")
    await ltm.write_workspace(ws.workspace_id, "alice", "品牌配色：主色橙#FF6B35")

    # 4 层混合召回
    results = await ltm.search_mixed(
        query="活动方案", user_id="alice",
        workspace_id=ws.workspace_id, project_id=proj.project_id,
        top_k=8,
    )
    print(WorkspaceAwareLTM.format_for_context(results))

切换到 MySQL：
    store = MySQLWorkspaceStore(host="db", user="u", password="pw", database="agent")
    # 其余代码完全相同
"""

from workspace.models import (
    MemoryScope,
    MemoryShare,
    Project,
    ProjectMember,
    ProjectRole,
    RankedEntry,
    Workspace,
    WorkspaceContext,
    WorkspaceMember,
    WorkspaceMemoryEntry,
    WorkspaceRole,
)
from workspace.store import (
    MySQLWorkspaceStore,
    SQLiteWorkspaceStore,
    WorkspaceStore,
    create_store,
)
from workspace.manager import WorkspaceManager
from workspace.memory import WorkspaceAwareLTM

__all__ = [
    # models
    "MemoryScope", "MemoryShare", "Project", "ProjectMember", "ProjectRole",
    "RankedEntry", "Workspace", "WorkspaceContext", "WorkspaceMember",
    "WorkspaceMemoryEntry", "WorkspaceRole",
    # store
    "WorkspaceStore", "SQLiteWorkspaceStore", "MySQLWorkspaceStore", "create_store",
    # manager
    "WorkspaceManager",
    # memory
    "WorkspaceAwareLTM",
]
