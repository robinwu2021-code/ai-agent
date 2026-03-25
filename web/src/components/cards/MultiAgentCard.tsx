'use client'
import type { SubTaskInfo } from '@/types'

interface Props {
  subTasks: SubTaskInfo[]
}

const STATUS_ICON: Record<SubTaskInfo['status'], string> = {
  pending: '○',
  running: '◌',
  done:    '✓',
  error:   '✗',
}

const STATUS_COLOR: Record<SubTaskInfo['status'], string> = {
  pending: 'text-slate-400',
  running: 'text-violet-400',
  done:    'text-emerald-400',
  error:   'text-red-400',
}

const STATUS_BG: Record<SubTaskInfo['status'], string> = {
  pending: 'bg-slate-800 border-slate-700',
  running: 'bg-violet-950 border-violet-700',
  done:    'bg-emerald-950 border-emerald-800',
  error:   'bg-red-950 border-red-800',
}

/** Color tag per agent name (cycles through a palette) */
const AGENT_COLORS = [
  'bg-blue-800 text-blue-200',
  'bg-violet-800 text-violet-200',
  'bg-amber-800 text-amber-200',
  'bg-teal-800 text-teal-200',
  'bg-rose-800 text-rose-200',
]

function agentColor(name: string, allAgents: string[]) {
  const idx = allAgents.indexOf(name)
  return AGENT_COLORS[idx % AGENT_COLORS.length]
}

export default function MultiAgentCard({ subTasks }: Props) {
  const agents = Array.from(new Set(subTasks.map(s => s.agent)))
  const total  = subTasks.length
  const done   = subTasks.filter(s => s.status === 'done').length
  const pct    = total > 0 ? Math.round((done / total) * 100) : 0
  const totalTokens = subTasks.reduce((acc, s) => acc + (s.tokens ?? 0), 0)

  return (
    <div className="mt-2 rounded-xl border border-slate-700 bg-slate-900/80 p-4 text-sm backdrop-blur">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base">🤖</span>
          <span className="font-semibold text-slate-200">多 Agent 协作</span>
          <span className="rounded-full bg-slate-700 px-2 py-0.5 text-xs text-slate-400">
            {done}/{total} 子任务
          </span>
        </div>
        {totalTokens > 0 && (
          <span className="text-xs text-slate-500">{totalTokens.toLocaleString()} tokens</span>
        )}
      </div>

      {/* Agent legend */}
      <div className="mb-3 flex flex-wrap gap-1.5">
        {agents.map(a => (
          <span key={a} className={`rounded-full px-2 py-0.5 text-xs font-medium ${agentColor(a, agents)}`}>
            {a}
          </span>
        ))}
      </div>

      {/* Progress bar */}
      <div className="mb-4 h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-violet-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Sub-tasks */}
      <div className="flex flex-col gap-2">
        {subTasks.map(st => (
          <div
            key={st.id}
            className={`flex items-start gap-2.5 rounded-lg border p-2.5 ${STATUS_BG[st.status]}`}
          >
            <span className={`mt-0.5 text-base font-bold leading-none ${
              st.status === 'running' ? STATUS_COLOR[st.status] + ' animate-pulse' : STATUS_COLOR[st.status]
            }`}>
              {STATUS_ICON[st.status]}
            </span>
            <div className="min-w-0 flex-1">
              <div className="mb-0.5 flex items-center gap-1.5">
                <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${agentColor(st.agent, agents)}`}>
                  {st.agent}
                </span>
                {st.depends.length > 0 && (
                  <span className="text-xs text-slate-500">← {st.depends.join(', ')}</span>
                )}
                {st.tokens && (
                  <span className="ml-auto text-xs text-slate-500">{st.tokens} tk</span>
                )}
              </div>
              <p className="text-slate-200">{st.goal}</p>
              {st.status === 'running' && (
                <p className="mt-0.5 text-xs text-violet-400 animate-pulse">Agent 处理中…</p>
              )}
              {st.error && (
                <p className="mt-0.5 text-xs text-red-400">{st.error}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
