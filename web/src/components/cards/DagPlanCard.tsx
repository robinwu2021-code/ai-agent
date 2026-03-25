'use client'
import type { DagStep } from '@/types'

interface Props {
  steps: DagStep[]
}

const STATUS_ICON: Record<DagStep['status'], string> = {
  pending: '○',
  running: '◌',
  done: '✓',
  error: '✗',
}

const STATUS_COLOR: Record<DagStep['status'], string> = {
  pending:  'text-slate-400',
  running:  'text-blue-400 animate-pulse',
  done:     'text-emerald-400',
  error:    'text-red-400',
}

const STATUS_BG: Record<DagStep['status'], string> = {
  pending:  'bg-slate-800 border-slate-700',
  running:  'bg-blue-950 border-blue-700',
  done:     'bg-emerald-950 border-emerald-800',
  error:    'bg-red-950 border-red-800',
}

/** Groups steps by their parallelGroup (or solo) for visual grouping */
function groupSteps(steps: DagStep[]): DagStep[][] {
  const seen = new Set<string>()
  const groups: DagStep[][] = []
  for (const s of steps) {
    if (seen.has(s.id)) continue
    if (s.parallelGroup && s.parallelGroup.length > 1) {
      const group = steps.filter(x => s.parallelGroup!.includes(x.id))
      group.forEach(x => seen.add(x.id))
      groups.push(group)
    } else {
      seen.add(s.id)
      groups.push([s])
    }
  }
  return groups
}

export default function DagPlanCard({ steps }: Props) {
  const groups = groupSteps(steps)
  const total = steps.length
  const done  = steps.filter(s => s.status === 'done').length
  const pct   = total > 0 ? Math.round((done / total) * 100) : 0

  return (
    <div className="mt-2 rounded-xl border border-slate-700 bg-slate-900/80 p-4 text-sm backdrop-blur">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base">🗂</span>
          <span className="font-semibold text-slate-200">任务执行计划</span>
          <span className="rounded-full bg-slate-700 px-2 py-0.5 text-xs text-slate-400">
            {done}/{total} 步骤
          </span>
        </div>
        <span className="text-xs text-slate-500">{pct}%</span>
      </div>

      {/* Progress bar */}
      <div className="mb-4 h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-emerald-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Step groups */}
      <div className="flex flex-col gap-2">
        {groups.map((group, gi) => (
          <div key={gi}>
            {group.length > 1 && (
              <div className="mb-1 flex items-center gap-1 text-xs text-slate-500">
                <span className="text-blue-400">⚡</span>
                <span>并行执行 ({group.length} 步)</span>
              </div>
            )}
            <div className={`flex ${group.length > 1 ? 'gap-2' : 'flex-col gap-1'}`}>
              {group.map(step => (
                <div
                  key={step.id}
                  className={`flex flex-1 items-start gap-2 rounded-lg border p-2.5 ${STATUS_BG[step.status]}`}
                >
                  <span className={`mt-0.5 text-base font-bold leading-none ${STATUS_COLOR[step.status]}`}>
                    {STATUS_ICON[step.status]}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-slate-200">{step.goal}</p>
                    {step.status === 'running' && (
                      <p className="mt-0.5 text-xs text-blue-400 animate-pulse">执行中…</p>
                    )}
                    {step.error && (
                      <p className="mt-0.5 truncate text-xs text-red-400">{step.error}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
