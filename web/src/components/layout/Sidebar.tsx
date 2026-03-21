'use client'
import type { SkillMode, SessionStats } from '@/types'
import { cn } from '@/lib/utils'
import { useEffect, useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { fetchHealth } from '@/lib/api'

interface NavItem {
  id: SkillMode | 'graph'
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
]

interface Props {
  mode: SkillMode
  setMode: (m: SkillMode) => void
  stats: SessionStats
  sessionId: string
  open: boolean
  onClose: () => void
  onClear: () => void
}

export default function Sidebar({ mode, setMode, stats, sessionId, open, onClear }: Props) {
  const [connected, setConnected] = useState<boolean | null>(null)
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    fetchHealth()
      .then(() => setConnected(true))
      .catch(() => setConnected(false))
  }, [])

  function handleNavClick(item: NavItem) {
    if (item.href) {
      router.push(item.href)
    } else {
      // If we're on a sub-route, navigate back to home first
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

  return (
    <aside className={cn(
      'flex flex-col flex-shrink-0 transition-all duration-300 overflow-hidden',
      'border-r',
      open ? 'w-60' : 'w-0',
    )}
    style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="w-7 h-7 rounded-lg flex items-center justify-center text-sm"
          style={{ background: 'var(--purple-dim)', border: '1px solid rgba(124,58,237,0.4)' }}>
          ✦
        </div>
        <div>
          <div className="font-semibold text-sm" style={{ color: '#f1f5f9' }}>AI Agent</div>
          <div className="text-xs" style={{ color: 'var(--muted)' }}>智能助手</div>
        </div>
      </div>

      {/* Nav */}
      <div className="flex-1 overflow-y-auto py-3 px-2">
        <div className="text-xs px-2 mb-2 font-medium uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
          对话模式
        </div>
        <div className="space-y-0.5">
          {NAV_ITEMS.map(item => {
            const active = isItemActive(item)
            return (
              <button
                key={item.id}
                onClick={() => handleNavClick(item)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all duration-150',
                  active ? 'text-white' : 'hover:bg-white/5'
                )}
                style={active ? {
                  background: `${item.color}22`,
                  border: `1px solid ${item.color}44`,
                } : { border: '1px solid transparent' }}
              >
                <span className="text-base leading-none">{item.icon}</span>
                <div className="min-w-0">
                  <div className="text-sm font-medium truncate"
                    style={{ color: active ? item.color : '#cbd5e1' }}>
                    {item.label}
                  </div>
                  <div className="text-xs truncate" style={{ color: 'var(--muted)' }}>{item.sub}</div>
                </div>
              </button>
            )
          })}
        </div>

        {/* Stats */}
        <div className="mt-4 px-2">
          <div className="text-xs mb-2 font-medium uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
            会话统计
          </div>
          <div className="rounded-lg p-3 space-y-2" style={{ background: 'var(--elevated)', border: '1px solid var(--border)' }}>
            {[
              { label: '对话次数', value: stats.calls },
              { label: '总 Tokens', value: stats.totalTokens.toLocaleString() },
              { label: '工具调用', value: stats.toolCalls },
            ].map(s => (
              <div key={s.label} className="flex justify-between items-center text-xs">
                <span style={{ color: 'var(--muted)' }}>{s.label}</span>
                <span className="font-mono font-medium" style={{ color: '#e2e8f0' }}>{s.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="mt-3 px-2">
          <button
            onClick={onClear}
            className="w-full text-xs py-2 rounded-lg transition-all duration-150 hover:bg-white/5 text-left px-3"
            style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}
          >
            🗑 清空会话
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="px-3 py-3 border-t" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--muted)' }}>
          <div className={cn('w-2 h-2 rounded-full', connected === true ? 'bg-green-500' : connected === false ? 'bg-red-500' : 'bg-amber-500')} />
          <span>{connected === true ? 'API 已连接' : connected === false ? 'API 未连接' : '检测中…'}</span>
        </div>
        <div className="text-xs mt-1 truncate font-mono" style={{ color: 'var(--muted)', opacity: 0.6 }}>
          {sessionId.slice(-8)}
        </div>
      </div>
    </aside>
  )
}
