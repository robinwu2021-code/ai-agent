'use client'
import type { SkillMode } from '@/types'
import { cn } from '@/lib/utils'

const MODE_LABEL: Record<SkillMode, string> = {
  general: '通用对话',
  weather: '天气预报',
  report: '智能报表',
  marketing: '营销方案',
  graph: '知识图谱',
}

const MODE_COLOR: Record<SkillMode, string> = {
  general: '#a78bfa',
  weather: '#38bdf8',
  report: '#34d399',
  marketing: '#fb923c',
  graph: '#f472b6',
}

interface Props {
  mode: SkillMode
  busy: boolean
  onToggleSidebar: () => void
}

export default function Header({ mode, busy, onToggleSidebar }: Props) {
  return (
    <div className="flex items-center gap-3 px-4 h-12 flex-shrink-0 border-b"
      style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
      <button
        onClick={onToggleSidebar}
        className="w-7 h-7 rounded-md flex items-center justify-center transition-colors hover:bg-white/5"
        style={{ color: 'var(--muted)' }}
        title="切换侧边栏"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="6" x2="21" y2="6"/>
          <line x1="3" y1="12" x2="21" y2="12"/>
          <line x1="3" y1="18" x2="21" y2="18"/>
        </svg>
      </button>

      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full" style={{ background: MODE_COLOR[mode] }} />
        <span className="text-sm font-medium" style={{ color: '#f1f5f9' }}>{MODE_LABEL[mode]}</span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        {busy && (
          <div className="flex items-center gap-1.5 text-xs" style={{ color: '#f59e0b' }}>
            <div className="w-3 h-3 rounded-full border-2 border-amber-500/30 border-t-amber-500 animate-spin" />
            <span>运行中</span>
          </div>
        )}
        <div className={cn('text-xs px-2.5 py-1 rounded-full', busy ? 'opacity-0' : '')}
          style={{ background: 'rgba(34,197,94,0.1)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.25)' }}>
          就绪
        </div>
      </div>
    </div>
  )
}
