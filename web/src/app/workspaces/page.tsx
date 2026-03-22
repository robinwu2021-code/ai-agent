'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useApp } from '@/contexts/AppContext'
import type { WorkspaceInfo, ProjectInfo } from '@/contexts/AppContext'
import { cn } from '@/lib/utils'

// ─── Role badge ───────────────────────────────────────────────────────────────

function RoleBadge({ role }: { role?: string }) {
  if (!role) return null
  const cfg: Record<string, { label: string; color: string; bg: string }> = {
    ADMIN:  { label: 'ADMIN',  color: '#86efac', bg: 'rgba(34,197,94,0.12)'  },
    MEMBER: { label: 'MEMBER', color: '#93c5fd', bg: 'rgba(37,99,235,0.12)'  },
    VIEWER: { label: 'VIEWER', color: '#94a3b8', bg: 'rgba(148,163,184,0.10)'},
  }
  const c = cfg[role.toUpperCase()] ?? cfg['VIEWER']
  return (
    <span
      className="text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider"
      style={{ color: c.color, background: c.bg }}
    >
      {c.label}
    </span>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function WorkspacesPage() {
  const router = useRouter()
  const appCtx = useApp()
  const {
    workspaces,
    projects,
    loadingWorkspaces,
    loadingProjects,
    refreshWorkspaces,
    refreshProjects,
    createWorkspace,
    createProject,
    selectedWorkspace: ctxWorkspace,
    selectedProject: ctxProject,
    selectWorkspace,
    selectProject,
  } = appCtx

  // ── Local state ────────────────────────────────────────────────────────────
  const [selectedWs, setSelectedWs] = useState<WorkspaceInfo | null>(null)

  const [showCreateWs, setShowCreateWs] = useState(false)
  const [wsNameInput, setWsNameInput] = useState('')
  const [wsDescInput, setWsDescInput] = useState('')
  const [creatingWs, setCreatingWs] = useState(false)
  const [wsError, setWsError] = useState('')

  const [showCreateProj, setShowCreateProj] = useState(false)
  const [projNameInput, setProjNameInput] = useState('')
  const [projDescInput, setProjDescInput] = useState('')
  const [creatingProj, setCreatingProj] = useState(false)
  const [projError, setProjError] = useState('')

  // ── Mount: refresh workspaces ──────────────────────────────────────────────
  useEffect(() => {
    refreshWorkspaces()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Load projects when ws selection changes ────────────────────────────────
  useEffect(() => {
    if (selectedWs) {
      refreshProjects(selectedWs.workspace_id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedWs?.workspace_id])

  // ── Create workspace ───────────────────────────────────────────────────────
  async function handleCreateWs(e: React.FormEvent) {
    e.preventDefault()
    if (!wsNameInput.trim()) { setWsError('请输入工作空间名称'); return }
    setCreatingWs(true)
    setWsError('')
    try {
      const ws = await createWorkspace(wsNameInput.trim(), wsDescInput.trim() || undefined)
      setWsNameInput('')
      setWsDescInput('')
      setShowCreateWs(false)
      setSelectedWs(ws)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '创建失败，请重试'
      setWsError(`创建失败：${msg}`)
    } finally {
      setCreatingWs(false)
    }
  }

  // ── Create project ─────────────────────────────────────────────────────────
  async function handleCreateProj(e: React.FormEvent) {
    e.preventDefault()
    if (!selectedWs) return
    if (!projNameInput.trim()) { setProjError('请输入项目名称'); return }
    setCreatingProj(true)
    setProjError('')
    try {
      await createProject(
        selectedWs.workspace_id,
        projNameInput.trim(),
        projDescInput.trim() || undefined
      )
      setProjNameInput('')
      setProjDescInput('')
      setShowCreateProj(false)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '创建失败，请重试'
      setProjError(`创建失败：${msg}`)
    } finally {
      setCreatingProj(false)
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  function formatDate(ts?: number) {
    if (!ts) return '—'
    return new Date(ts * 1000).toLocaleDateString('zh-CN')
  }

  const isCurrentWs = (ws: WorkspaceInfo) =>
    ctxWorkspace?.workspace_id === ws.workspace_id

  const isCurrentProj = (proj: ProjectInfo) =>
    ctxProject?.project_id === proj.project_id

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      className="min-h-screen overflow-auto"
      style={{ background: 'var(--bg)', color: 'var(--text)' }}
    >
      {/* Header */}
      <div
        className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b"
        style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.back()}
            className="flex items-center gap-1.5 text-sm hover:text-purple-400 transition-colors"
            style={{ color: 'var(--muted)' }}
          >
            ← 返回
          </button>
          <span style={{ color: 'var(--border)' }}>|</span>
          <h1 className="text-base font-semibold flex items-center gap-2" style={{ color: '#f1f5f9' }}>
            🏢 工作空间管理
          </h1>
        </div>
        <button
          onClick={() => { setShowCreateWs(v => !v); setShowCreateProj(false) }}
          className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg font-medium transition-all hover:opacity-90"
          style={{ background: 'var(--purple)', color: '#fff' }}
        >
          + 新建工作空间
        </button>
      </div>

      {/* Body */}
      <div className="flex gap-4 p-6 max-w-6xl mx-auto">
        {/* ── Left panel: workspace list ──────────────────────────────────── */}
        <div className="w-64 flex-shrink-0 space-y-2">
          <div
            className="text-xs font-medium uppercase tracking-wider mb-2"
            style={{ color: 'var(--muted)' }}
          >
            工作空间列表
          </div>

          {loadingWorkspaces && (
            <div className="text-xs text-center py-6" style={{ color: 'var(--muted)' }}>
              加载中…
            </div>
          )}

          {!loadingWorkspaces && workspaces.length === 0 && !showCreateWs && (
            <button
              onClick={() => setShowCreateWs(true)}
              className="w-full rounded-xl border-2 border-dashed py-10 flex flex-col items-center gap-2 hover:border-purple-500/50 transition-colors"
              style={{ borderColor: 'var(--border)' }}
            >
              <span className="text-3xl">🏢</span>
              <span className="text-sm font-medium" style={{ color: '#e2e8f0' }}>
                创建第一个工作空间
              </span>
              <span className="text-xs" style={{ color: 'var(--muted)' }}>
                点击开始
              </span>
            </button>
          )}

          {workspaces.map(ws => (
            <div
              key={ws.workspace_id}
              onClick={() => setSelectedWs(ws)}
              className={cn(
                'rounded-xl p-3 cursor-pointer transition-all duration-150',
                'hover:ring-1 hover:ring-purple-500/30',
                selectedWs?.workspace_id === ws.workspace_id
                  ? 'ring-1 ring-purple-500/60'
                  : ''
              )}
              style={{
                background:
                  selectedWs?.workspace_id === ws.workspace_id
                    ? 'rgba(124,58,237,0.1)'
                    : 'var(--surface)',
                border:
                  selectedWs?.workspace_id === ws.workspace_id
                    ? '1px solid rgba(124,58,237,0.4)'
                    : '1px solid var(--border)',
              }}
            >
              <div className="flex items-start justify-between gap-1.5">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-base flex-shrink-0">🏢</span>
                  <span
                    className="text-sm font-medium truncate"
                    style={{
                      color:
                        selectedWs?.workspace_id === ws.workspace_id
                          ? '#c4b5fd'
                          : '#e2e8f0',
                    }}
                  >
                    {ws.name}
                  </span>
                </div>
                <RoleBadge role={ws.role} />
              </div>
              {ws.description && (
                <div
                  className="text-[11px] mt-1 line-clamp-2"
                  style={{ color: 'var(--muted)' }}
                >
                  {ws.description}
                </div>
              )}
              <div className="flex gap-3 mt-1.5 text-[10px]" style={{ color: 'var(--muted)' }}>
                {ws.member_count != null && (
                  <span>{ws.member_count} 成员</span>
                )}
                {ws.project_count != null && (
                  <span>{ws.project_count} 项目</span>
                )}
              </div>
            </div>
          ))}

          {/* Create workspace inline form */}
          {showCreateWs && (
            <form
              onSubmit={handleCreateWs}
              className="rounded-xl p-3 space-y-2"
              style={{
                background: 'var(--surface)',
                border: '1px solid rgba(124,58,237,0.4)',
              }}
            >
              <div
                className="text-xs font-semibold mb-1"
                style={{ color: '#a78bfa' }}
              >
                新建工作空间
              </div>
              <input
                autoFocus
                value={wsNameInput}
                onChange={e => setWsNameInput(e.target.value)}
                placeholder="工作空间名称 *"
                className="w-full text-xs px-2.5 py-1.5 rounded-lg outline-none"
                style={{
                  background: 'var(--elevated)',
                  border: '1px solid var(--border-strong)',
                  color: '#e2e8f0',
                }}
                maxLength={50}
              />
              <input
                value={wsDescInput}
                onChange={e => setWsDescInput(e.target.value)}
                placeholder="描述（可选）"
                className="w-full text-xs px-2.5 py-1.5 rounded-lg outline-none"
                style={{
                  background: 'var(--elevated)',
                  border: '1px solid var(--border-strong)',
                  color: '#e2e8f0',
                }}
                maxLength={120}
              />
              {wsError && (
                <div className="text-[11px]" style={{ color: '#f87171' }}>
                  {wsError}
                </div>
              )}
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={creatingWs}
                  className="flex-1 py-1.5 text-xs rounded-lg font-medium transition-opacity hover:opacity-90 disabled:opacity-50"
                  style={{ background: 'var(--purple)', color: '#fff' }}
                >
                  {creatingWs ? '创建中…' : '创建'}
                </button>
                <button
                  type="button"
                  onClick={() => { setShowCreateWs(false); setWsError('') }}
                  className="flex-1 py-1.5 text-xs rounded-lg transition-colors hover:bg-white/5"
                  style={{
                    color: 'var(--muted)',
                    border: '1px solid var(--border)',
                  }}
                >
                  取消
                </button>
              </div>
            </form>
          )}
        </div>

        {/* ── Right panel: workspace detail ───────────────────────────────── */}
        {selectedWs ? (
          <div className="flex-1 min-w-0 space-y-4">
            {/* Workspace header card */}
            <div
              className="rounded-xl p-4"
              style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
              }}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h2
                      className="text-lg font-semibold"
                      style={{ color: '#f1f5f9' }}
                    >
                      {selectedWs.name}
                    </h2>
                    <RoleBadge role={selectedWs.role} />
                    {isCurrentWs(selectedWs) && (
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
                        style={{ color: '#86efac', background: 'rgba(34,197,94,0.12)' }}
                      >
                        ✓ 当前
                      </span>
                    )}
                  </div>
                  {selectedWs.description && (
                    <p
                      className="text-sm mt-1"
                      style={{ color: 'var(--muted)' }}
                    >
                      {selectedWs.description}
                    </p>
                  )}
                  <div
                    className="flex gap-4 text-xs mt-2"
                    style={{ color: 'var(--muted)' }}
                  >
                    {selectedWs.member_count != null && (
                      <span>{selectedWs.member_count} 成员</span>
                    )}
                    {selectedWs.created_at != null && (
                      <span>创建于 {formatDate(selectedWs.created_at)}</span>
                    )}
                  </div>
                </div>
                {!isCurrentWs(selectedWs) && (
                  <button
                    onClick={() => selectWorkspace(selectedWs)}
                    className="flex-shrink-0 px-3 py-1.5 text-xs rounded-lg font-medium transition-all hover:opacity-90"
                    style={{ background: 'var(--purple)', color: '#fff' }}
                  >
                    设为当前工作空间
                  </button>
                )}
              </div>
            </div>

            {/* Projects section */}
            <div>
              <div className="flex items-center justify-between mb-2.5">
                <h3
                  className="text-sm font-semibold flex items-center gap-1.5"
                  style={{ color: '#e2e8f0' }}
                >
                  📁 项目列表
                </h3>
                <button
                  onClick={() => { setShowCreateProj(v => !v); setProjError('') }}
                  className="text-xs px-2.5 py-1 rounded-lg transition-all hover:opacity-90"
                  style={{
                    background: showCreateProj ? 'var(--elevated)' : 'rgba(124,58,237,0.15)',
                    color: '#a78bfa',
                    border: '1px solid rgba(124,58,237,0.3)',
                  }}
                >
                  + 新建项目
                </button>
              </div>

              {/* Create project inline form */}
              {showCreateProj && (
                <form
                  onSubmit={handleCreateProj}
                  className="rounded-xl p-3 mb-3 space-y-2"
                  style={{
                    background: 'var(--surface)',
                    border: '1px solid rgba(124,58,237,0.4)',
                  }}
                >
                  <div
                    className="text-xs font-semibold mb-1"
                    style={{ color: '#a78bfa' }}
                  >
                    新建项目
                  </div>
                  <input
                    autoFocus
                    value={projNameInput}
                    onChange={e => setProjNameInput(e.target.value)}
                    placeholder="项目名称 *"
                    className="w-full text-xs px-2.5 py-1.5 rounded-lg outline-none"
                    style={{
                      background: 'var(--elevated)',
                      border: '1px solid var(--border-strong)',
                      color: '#e2e8f0',
                    }}
                    maxLength={50}
                  />
                  <input
                    value={projDescInput}
                    onChange={e => setProjDescInput(e.target.value)}
                    placeholder="描述（可选）"
                    className="w-full text-xs px-2.5 py-1.5 rounded-lg outline-none"
                    style={{
                      background: 'var(--elevated)',
                      border: '1px solid var(--border-strong)',
                      color: '#e2e8f0',
                    }}
                    maxLength={120}
                  />
                  {projError && (
                    <div className="text-[11px]" style={{ color: '#f87171' }}>
                      {projError}
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button
                      type="submit"
                      disabled={creatingProj}
                      className="flex-1 py-1.5 text-xs rounded-lg font-medium transition-opacity hover:opacity-90 disabled:opacity-50"
                      style={{ background: 'var(--purple)', color: '#fff' }}
                    >
                      {creatingProj ? '创建中…' : '创建'}
                    </button>
                    <button
                      type="button"
                      onClick={() => { setShowCreateProj(false); setProjError('') }}
                      className="flex-1 py-1.5 text-xs rounded-lg transition-colors hover:bg-white/5"
                      style={{
                        color: 'var(--muted)',
                        border: '1px solid var(--border)',
                      }}
                    >
                      取消
                    </button>
                  </div>
                </form>
              )}

              {loadingProjects && (
                <div
                  className="text-xs text-center py-8"
                  style={{ color: 'var(--muted)' }}
                >
                  加载项目中…
                </div>
              )}

              {!loadingProjects && projects.length === 0 && !showCreateProj && (
                <div
                  className="rounded-xl border-2 border-dashed py-10 flex flex-col items-center gap-2"
                  style={{ borderColor: 'var(--border)' }}
                >
                  <span className="text-3xl">📁</span>
                  <span className="text-sm" style={{ color: '#e2e8f0' }}>
                    暂无项目
                  </span>
                  <button
                    onClick={() => setShowCreateProj(true)}
                    className="text-xs hover:text-purple-400 transition-colors"
                    style={{ color: 'var(--muted)' }}
                  >
                    + 点击创建第一个项目
                  </button>
                </div>
              )}

              {/* Project cards grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {projects.map(proj => (
                  <div
                    key={proj.project_id}
                    className={cn(
                      'rounded-xl p-3 transition-all duration-150',
                      'hover:ring-1 hover:ring-purple-500/30',
                      isCurrentProj(proj) ? 'ring-1 ring-purple-500/50' : ''
                    )}
                    style={{
                      background: isCurrentProj(proj)
                        ? 'rgba(124,58,237,0.08)'
                        : 'var(--elevated)',
                      border: isCurrentProj(proj)
                        ? '1px solid rgba(124,58,237,0.35)'
                        : '1px solid var(--border)',
                    }}
                  >
                    <div className="flex items-start justify-between gap-1 mb-1">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="text-sm">📁</span>
                        <span
                          className="text-sm font-medium truncate"
                          style={{
                            color: isCurrentProj(proj) ? '#c4b5fd' : '#e2e8f0',
                          }}
                        >
                          {proj.name}
                        </span>
                      </div>
                      {isCurrentProj(proj) && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded font-semibold flex-shrink-0"
                          style={{
                            color: '#86efac',
                            background: 'rgba(34,197,94,0.12)',
                          }}
                        >
                          ✓ 当前
                        </span>
                      )}
                    </div>

                    {proj.role && (
                      <div className="mb-1.5">
                        <RoleBadge role={proj.role} />
                      </div>
                    )}

                    {proj.description && (
                      <p
                        className="text-[11px] line-clamp-2 mb-1.5"
                        style={{ color: 'var(--muted)' }}
                      >
                        {proj.description}
                      </p>
                    )}

                    <div
                      className="flex gap-3 text-[10px] mb-2.5"
                      style={{ color: 'var(--muted)' }}
                    >
                      {proj.member_count != null && (
                        <span>{proj.member_count} 成员</span>
                      )}
                      {proj.created_at != null && (
                        <span>{formatDate(proj.created_at)}</span>
                      )}
                    </div>

                    {!isCurrentProj(proj) ? (
                      <button
                        onClick={() => {
                          selectWorkspace(selectedWs)
                          selectProject(proj)
                        }}
                        className="w-full py-1 text-[11px] rounded-lg font-medium transition-all hover:opacity-90"
                        style={{
                          background: 'rgba(124,58,237,0.18)',
                          color: '#a78bfa',
                          border: '1px solid rgba(124,58,237,0.25)',
                        }}
                      >
                        选择项目
                      </button>
                    ) : (
                      <button
                        onClick={() => selectProject(null)}
                        className="w-full py-1 text-[11px] rounded-lg font-medium transition-all hover:opacity-90"
                        style={{
                          background: 'rgba(34,197,94,0.1)',
                          color: '#86efac',
                          border: '1px solid rgba(34,197,94,0.2)',
                        }}
                      >
                        ✓ 已选择 · 取消
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Empty right panel */
          <div
            className="flex-1 flex items-center justify-center rounded-xl"
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
            }}
          >
            <div className="text-center space-y-2">
              <div className="text-4xl">👈</div>
              <div className="text-sm font-medium" style={{ color: '#e2e8f0' }}>
                选择一个工作空间
              </div>
              <div className="text-xs" style={{ color: 'var(--muted)' }}>
                在左侧点击工作空间查看详情
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
