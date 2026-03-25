'use client'
import type { SkillMode, SessionStats } from '@/types'
import { cn } from '@/lib/utils'
import { useEffect, useState, useRef, useCallback } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { fetchHealth } from '@/lib/api'
import { useApp } from '@/contexts/AppContext'
import type { WorkspaceInfo, ProjectInfo, ChatSession } from '@/contexts/AppContext'

// ─── Nav config ───────────────────────────────────────────────────────────────

interface NavItem {
  id: SkillMode | 'graph' | 'knowledge'
  icon: string
  label: string
  sub: string
  color: string
  href?: string
}

const NAV_ITEMS: NavItem[] = [
  { id: 'general',   icon: '✦',  label: '通用对话', sub: '全能助手',   color: '#a78bfa' },
  { id: 'weather',   icon: '🌤', label: '天气预报', sub: '实时气象查询', color: '#38bdf8' },
  { id: 'report',    icon: '📊', label: '智能报表', sub: '销售数据分析', color: '#34d399' },
  { id: 'marketing', icon: '📢', label: '营销方案', sub: '商家营销顾问', color: '#fb923c' },
  { id: 'graph',     icon: '🕸', label: '知识图谱', sub: '实体关系探索', color: '#06b6d4', href: '/graph' },
  { id: 'knowledge', icon: '📚', label: '知识库',   sub: '文档问答检索', color: '#8b5cf6', href: '/knowledge' },
]

const MODE_ICONS: Record<string, string> = {
  general: '✦',
  weather: '🌤',
  report: '📊',
  marketing: '📢',
  graph: '🕸',
  knowledge: '📚',
}

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  mode: SkillMode
  setMode: (m: SkillMode) => void
  stats: SessionStats
  sessionId: string
  open: boolean
  onClose: () => void
  onClear: () => void
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function RoleBadge({ role }: { role?: string }) {
  if (!role) return null
  const cfg: Record<string, { label: string; color: string; bg: string }> = {
    ADMIN:  { label: 'ADMIN',  color: '#86efac', bg: 'rgba(34,197,94,0.12)' },
    MEMBER: { label: 'MEMBER', color: '#93c5fd', bg: 'rgba(37,99,235,0.12)' },
    VIEWER: { label: 'VIEWER', color: '#94a3b8', bg: 'rgba(148,163,184,0.1)' },
  }
  const c = cfg[role] ?? cfg['VIEWER']
  return (
    <span
      className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
      style={{ color: c.color, background: c.bg }}
    >
      {c.label}
    </span>
  )
}

interface DropdownProps {
  label: string
  value: string | null
  items: Array<WorkspaceInfo | ProjectInfo>
  disabled?: boolean
  onSelect: (item: WorkspaceInfo | ProjectInfo | null) => void
  icon: string
}

function SelectorDropdown({ label, value, items, disabled, onSelect, icon }: DropdownProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const displayName = value ?? label

  return (
    <div className="relative" ref={ref}>
      <button
        disabled={disabled}
        onClick={() => !disabled && setOpen(o => !o)}
        className={cn(
          'w-full flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg text-xs transition-all',
          disabled
            ? 'opacity-40 cursor-not-allowed'
            : 'hover:bg-white/5 cursor-pointer',
        )}
        style={{
          background: open ? 'rgba(124,58,237,0.1)' : 'var(--elevated)',
          border: `1px solid ${open ? 'rgba(124,58,237,0.4)' : 'var(--border)'}`,
          color: value ? '#e2e8f0' : 'var(--muted)',
        }}
      >
        <span className="flex items-center gap-1.5 min-w-0">
          <span>{icon}</span>
          <span className="truncate">{displayName}</span>
        </span>
        <span style={{ color: 'var(--muted)', fontSize: 10 }}>▾</span>
      </button>

      {open && (
        <div
          className="absolute left-0 right-0 z-50 mt-1 rounded-lg overflow-hidden shadow-xl"
          style={{
            background: 'var(--elevated)',
            border: '1px solid var(--border-strong)',
            maxHeight: 200,
            overflowY: 'auto',
          }}
        >
          {/* Unselect option */}
          <button
            onClick={() => { onSelect(null); setOpen(false) }}
            className="w-full text-left px-3 py-2 text-xs hover:bg-white/5 transition-colors"
            style={{ color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}
          >
            未选择
          </button>
          {items.length === 0 && (
            <div className="px-3 py-3 text-xs text-center" style={{ color: 'var(--muted)' }}>
              暂无数据
            </div>
          )}
          {items.map(item => {
            const id = 'workspace_id' in item
              ? (item as WorkspaceInfo).workspace_id
              : (item as ProjectInfo).project_id
            const isSelected = value === item.name
            return (
              <button
                key={id}
                onClick={() => { onSelect(item); setOpen(false) }}
                className="w-full text-left px-3 py-2 text-xs hover:bg-white/5 transition-colors flex items-center justify-between gap-2"
                style={{
                  color: isSelected ? '#a78bfa' : '#cbd5e1',
                  background: isSelected ? 'rgba(124,58,237,0.08)' : undefined,
                }}
              >
                <span className="truncate">{item.name}</span>
                <RoleBadge role={item.role} />
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Sidebar({ mode, setMode, stats, sessionId, open, onClear }: Props) {
  const appCtx = useApp()
  const {
    currentUser,
    setUsername,
    selectedWorkspace,
    selectedProject,
    selectWorkspace,
    selectProject,
    workspaces,
    projects,
    sessions,
    currentSessionId,
    setCurrentSessionId,
    deleteSession,
    getProjectSessions,
  } = appCtx

  const [connected, setConnected] = useState<boolean | null>(null)
  const [editingName, setEditingName] = useState(false)
  const [nameInput, setNameInput] = useState('')
  const nameInputRef = useRef<HTMLInputElement>(null)

  const [showNewChatPopup, setShowNewChatPopup] = useState(false)
  const newChatPopupRef = useRef<HTMLDivElement>(null)

  const router = useRouter()
  const pathname = usePathname()

  // ── Health check ──────────────────────────────────────────────────────────
  useEffect(() => {
    fetchHealth()
      .then(() => setConnected(true))
      .catch(() => setConnected(false))
  }, [])

  // ── Close new-chat popup on outside click ─────────────────────────────────
  useEffect(() => {
    if (!showNewChatPopup) return
    function handler(e: MouseEvent) {
      if (newChatPopupRef.current && !newChatPopupRef.current.contains(e.target as Node)) {
        setShowNewChatPopup(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showNewChatPopup])

  // ── Start new chat ────────────────────────────────────────────────────────
  const doStartNewChat = useCallback(() => {
    setShowNewChatPopup(false)
    onClear()
    appCtx.setCurrentSessionId(null)
    if (pathname !== '/') router.push('/')
  }, [onClear, appCtx, pathname, router])

  // ── Nav handlers ──────────────────────────────────────────────────────────
  function handleNavClick(item: NavItem) {
    if (item.href) {
      router.push(item.href)
    } else {
      if (pathname !== '/') {
        router.push('/')
      }
      setMode(item.id as SkillMode)
    }
  }

  function isItemActive(item: NavItem): boolean {
    if (item.href) {
      return pathname === item.href || pathname.startsWith(item.href + '/')
    }
    return pathname === '/' && mode === item.id
  }

  // ── Username editing ──────────────────────────────────────────────────────
  function startEditName() {
    setNameInput(currentUser.username)
    setEditingName(true)
    setTimeout(() => nameInputRef.current?.focus(), 0)
  }

  function commitName() {
    if (nameInput.trim()) setUsername(nameInput.trim())
    setEditingName(false)
  }

  // ── Session list ──────────────────────────────────────────────────────────
  const filteredSessions = getProjectSessions(
    selectedWorkspace?.workspace_id,
    selectedProject?.project_id
  )
    .slice()
    .sort((a, b) => b.updated_at - a.updated_at)
    .slice(0, 8)

  // ── Workspace/project selector handlers ───────────────────────────────────
  function handleSelectWorkspace(item: WorkspaceInfo | ProjectInfo | null) {
    selectWorkspace(item as WorkspaceInfo | null)
  }

  function handleSelectProject(item: WorkspaceInfo | ProjectInfo | null) {
    selectProject(item as ProjectInfo | null)
  }

  return (
    <aside
      className={cn(
        'flex flex-col flex-shrink-0 transition-all duration-300 overflow-hidden',
        'border-r',
        open ? 'w-60' : 'w-0',
      )}
      style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
    >
      {/* ── User section ───────────────────────────────────────────────────── */}
      <div
        className="px-3 py-3 border-b"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-2 min-w-0">
          {/* Avatar */}
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 select-none"
            style={{
              background: currentUser.avatar_color,
              color: '#fff',
              textShadow: '0 1px 2px rgba(0,0,0,0.4)',
            }}
          >
            {currentUser.username.charAt(0).toUpperCase()}
          </div>

          {/* Name / edit input */}
          <div className="flex-1 min-w-0">
            {editingName ? (
              <input
                ref={nameInputRef}
                value={nameInput}
                onChange={e => setNameInput(e.target.value)}
                onBlur={commitName}
                onKeyDown={e => {
                  if (e.key === 'Enter') commitName()
                  if (e.key === 'Escape') setEditingName(false)
                }}
                className="w-full bg-transparent border-b text-sm font-medium outline-none"
                style={{
                  color: '#f1f5f9',
                  borderColor: 'rgba(124,58,237,0.6)',
                  caretColor: '#a78bfa',
                }}
                maxLength={24}
              />
            ) : (
              <div
                className="text-sm font-medium truncate"
                style={{ color: '#f1f5f9' }}
              >
                {currentUser.username}
              </div>
            )}
            <div
              className="text-[10px] font-mono truncate mt-0.5"
              style={{ color: 'var(--muted)' }}
            >
              {currentUser.user_id}
            </div>
          </div>

          {/* Edit pencil */}
          {!editingName && (
            <button
              onClick={startEditName}
              className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded hover:bg-white/10 transition-colors text-sm"
              style={{ color: 'var(--muted)' }}
              title="编辑用户名"
            >
              ✏️
            </button>
          )}
        </div>
      </div>

      {/* ── Workspace / Project section ────────────────────────────────────── */}
      <div
        className="px-3 py-2.5 border-b space-y-2"
        style={{ borderColor: 'var(--border)' }}
      >
        <div>
          <div
            className="text-[10px] font-medium uppercase tracking-wider mb-1.5 flex items-center gap-1"
            style={{ color: 'var(--muted)' }}
          >
            🏢 工作空间
          </div>
          <SelectorDropdown
            label="未选择"
            value={selectedWorkspace?.name ?? null}
            items={workspaces}
            onSelect={handleSelectWorkspace}
            icon="🏢"
          />
        </div>

        <div>
          <div
            className="text-[10px] font-medium uppercase tracking-wider mb-1.5 flex items-center gap-1"
            style={{ color: 'var(--muted)' }}
          >
            📁 项目
          </div>
          <SelectorDropdown
            label="未选择"
            value={selectedProject?.name ?? null}
            items={projects}
            disabled={!selectedWorkspace}
            onSelect={handleSelectProject}
            icon="📁"
          />
        </div>

        <button
          onClick={() => router.push('/workspaces')}
          className="w-full text-left text-xs px-1 py-0.5 transition-colors hover:text-purple-400"
          style={{ color: 'var(--muted)' }}
        >
          + 管理工作空间
        </button>
      </div>

      {/* ── Nav ────────────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto py-2 px-2">
        <div
          className="text-[10px] px-2 mb-1.5 font-medium uppercase tracking-wider"
          style={{ color: 'var(--muted)' }}
        >
          导航
        </div>
        <div className="space-y-0.5">
          {NAV_ITEMS.map(item => {
            const active = isItemActive(item)
            return (
              <button
                key={item.id}
                onClick={() => handleNavClick(item)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all duration-150',
                  active ? 'text-white' : 'hover:bg-white/5'
                )}
                style={
                  active
                    ? {
                        background: `${item.color}22`,
                        border: `1px solid ${item.color}44`,
                      }
                    : { border: '1px solid transparent' }
                }
              >
                <span className="text-base leading-none">{item.icon}</span>
                <div className="min-w-0">
                  <div
                    className="text-xs font-medium truncate"
                    style={{ color: active ? item.color : '#cbd5e1' }}
                  >
                    {item.label}
                  </div>
                  <div className="text-[10px] truncate" style={{ color: 'var(--muted)' }}>
                    {item.sub}
                  </div>
                </div>
              </button>
            )
          })}
        </div>

        {/* ── Recent sessions ──────────────────────────────────────────────── */}
        <div className="mt-3">
          <div
            className="text-[10px] px-2 mb-1.5 font-medium uppercase tracking-wider flex items-center justify-between"
            style={{ color: 'var(--muted)' }}
          >
            <span>最近对话</span>
            {filteredSessions.length > 0 && (
              <button
                onClick={() => {
                  filteredSessions.forEach(s =>
                    appCtx.deleteSession(s.session_id)
                  )
                  setCurrentSessionId(null)
                }}
                className="text-[10px] hover:text-red-400 transition-colors"
                style={{ color: 'var(--muted)' }}
                title="清空"
              >
                清空
              </button>
            )}
          </div>

          <div className="space-y-0.5">
            {filteredSessions.map(session => {
              const isActive = session.session_id === currentSessionId
              return (
                <div
                  key={session.session_id}
                  className={cn(
                    'group flex items-center gap-1.5 px-2 py-1.5 rounded-lg cursor-pointer transition-all duration-150',
                    isActive ? 'bg-purple-900/30' : 'hover:bg-white/5'
                  )}
                  style={{
                    border: isActive
                      ? '1px solid rgba(124,58,237,0.35)'
                      : '1px solid transparent',
                  }}
                  onClick={() => setCurrentSessionId(session.session_id)}
                >
                  <span className="text-xs flex-shrink-0" title={session.mode}>
                    {MODE_ICONS[session.mode] ?? '✦'}
                  </span>
                  <span
                    className="flex-1 text-xs truncate"
                    style={{
                      color: isActive ? '#c4b5fd' : '#94a3b8',
                      maxWidth: 140,
                    }}
                    title={session.title}
                  >
                    {session.title.slice(0, 25)}
                  </span>
                  <button
                    onClick={e => {
                      e.stopPropagation()
                      deleteSession(session.session_id)
                    }}
                    className="flex-shrink-0 w-5 h-5 flex items-center justify-center rounded opacity-0 group-hover:opacity-100 hover:bg-red-900/40 transition-all text-[11px]"
                    style={{ color: '#f87171' }}
                    title="删除"
                  >
                    🗑
                  </button>
                </div>
              )
            })}

            {filteredSessions.length === 0 && (
              <div
                className="px-2 py-2 text-[11px] text-center"
                style={{ color: 'var(--muted)' }}
              >
                暂无对话记录
              </div>
            )}
          </div>

          {/* New chat button + popup */}
          <div className="relative mt-1.5" ref={newChatPopupRef}>
            <button
              onClick={() => setShowNewChatPopup(v => !v)}
              className="w-full text-xs py-1.5 rounded-lg transition-all duration-150 hover:bg-white/5 flex items-center justify-center gap-1.5"
              style={{
                color: '#a78bfa',
                border: showNewChatPopup
                  ? '1px solid rgba(124,58,237,0.5)'
                  : '1px dashed rgba(124,58,237,0.3)',
                background: showNewChatPopup ? 'rgba(124,58,237,0.08)' : undefined,
              }}
            >
              <span>✦</span>
              <span>新建对话</span>
            </button>

            {showNewChatPopup && (
              <div
                className="absolute bottom-full left-0 right-0 mb-1.5 z-50 rounded-xl p-3 space-y-2.5 shadow-2xl"
                style={{
                  background: 'var(--elevated)',
                  border: '1px solid var(--border-strong)',
                }}
              >
                <div className="text-xs font-semibold" style={{ color: '#a78bfa' }}>
                  ✦ 新建对话
                </div>
                <div className="text-[11px]" style={{ color: 'var(--muted)' }}>
                  选择项目（可选）
                </div>
                <SelectorDropdown
                  label="未选择工作空间"
                  value={selectedWorkspace?.name ?? null}
                  items={workspaces}
                  onSelect={handleSelectWorkspace}
                  icon="🏢"
                />
                <SelectorDropdown
                  label="未选择项目"
                  value={selectedProject?.name ?? null}
                  items={projects}
                  disabled={!selectedWorkspace}
                  onSelect={handleSelectProject}
                  icon="📁"
                />
                <div className="flex gap-2 pt-0.5">
                  <button
                    onClick={doStartNewChat}
                    className="flex-1 py-1.5 text-xs rounded-lg font-medium transition-opacity hover:opacity-90"
                    style={{ background: 'var(--purple)', color: '#fff' }}
                  >
                    开始对话
                  </button>
                  <button
                    onClick={() => setShowNewChatPopup(false)}
                    className="flex-1 py-1.5 text-xs rounded-lg transition-colors hover:bg-white/5"
                    style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}
                  >
                    取消
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Session stats ─────────────────────────────────────────────────── */}
        <div className="mt-3 px-1">
          <div
            className="text-[10px] px-1 mb-1.5 font-medium uppercase tracking-wider"
            style={{ color: 'var(--muted)' }}
          >
            会话统计
          </div>
          <div
            className="rounded-lg p-2.5 space-y-1.5"
            style={{
              background: 'var(--elevated)',
              border: '1px solid var(--border)',
            }}
          >
            {[
              { label: '对话次数', value: stats.calls },
              { label: '总 Tokens', value: stats.totalTokens.toLocaleString() },
              { label: '工具调用', value: stats.toolCalls },
            ].map(s => (
              <div
                key={s.label}
                className="flex justify-between items-center text-[11px]"
              >
                <span style={{ color: 'var(--muted)' }}>{s.label}</span>
                <span
                  className="font-mono font-medium"
                  style={{ color: '#e2e8f0' }}
                >
                  {s.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Footer / Status ────────────────────────────────────────────────── */}
      <div
        className="px-3 py-2.5 border-t"
        style={{ borderColor: 'var(--border)' }}
      >
        <div
          className="flex items-center gap-2 text-[11px]"
          style={{ color: 'var(--muted)' }}
        >
          <div
            className={cn(
              'w-1.5 h-1.5 rounded-full flex-shrink-0',
              connected === true
                ? 'bg-green-500'
                : connected === false
                ? 'bg-red-500'
                : 'bg-amber-500'
            )}
          />
          <span>
            {connected === true
              ? '已连接'
              : connected === false
              ? '未连接'
              : '检测中…'}
          </span>
          <span className="mx-0.5">|</span>
          <span>{stats.calls}次</span>
          <span>{stats.totalTokens.toLocaleString()}tok</span>
        </div>
        <div
          className="text-[10px] mt-0.5 font-mono truncate"
          style={{ color: 'var(--muted)', opacity: 0.5 }}
        >
          {sessionId.slice(-8)}
        </div>
      </div>
    </aside>
  )
}
