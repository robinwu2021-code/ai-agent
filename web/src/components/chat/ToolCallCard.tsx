import type { ToolCall } from '@/types'
import { cn } from '@/lib/utils'

const TOOL_ICONS: Record<string, string> = {
  weather_current: '🌤',
  agent_bi: '📊',
  marketing_advisor: '📢',
  datetime: '🕐',
  currency_exchange: '💱',
  unit_converter: '📐',
  text_translate: '🌐',
  stock_price: '📈',
}

interface Props {
  toolCalls: ToolCall[]
}

export default function ToolCallCards({ toolCalls }: Props) {
  if (!toolCalls.length) return null
  return (
    <div className="space-y-1.5 mb-2">
      {toolCalls.map(tc => (
        <ToolCallItem key={tc.id} tc={tc} />
      ))}
    </div>
  )
}

function ToolCallItem({ tc }: { tc: ToolCall }) {
  const icon = TOOL_ICONS[tc.name] ?? '⚙'
  const isRunning = tc.status === 'running'
  const isDone = tc.status === 'done'
  const isError = tc.status === 'error'

  return (
    <div className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs animate-slide-up"
      style={{
        background: 'var(--elevated)',
        border: `1px solid ${isDone ? 'rgba(34,197,94,0.2)' : isError ? 'rgba(239,68,68,0.2)' : 'rgba(245,158,11,0.2)'}`,
      }}>
      <span className="text-sm">{icon}</span>
      <span className="font-mono" style={{ color: '#e2e8f0' }}>{tc.name}</span>
      {isRunning && (
        <div className="w-3 h-3 ml-auto rounded-full border-2 border-amber-500/30 border-t-amber-500 animate-spin" />
      )}
      {isDone && (
        <span className="ml-auto text-xs px-1.5 py-0.5 rounded" style={{ background: 'rgba(34,197,94,0.1)', color: '#22c55e' }}>
          完成
        </span>
      )}
      {isError && (
        <span className="ml-auto text-xs px-1.5 py-0.5 rounded" style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444' }}>
          {tc.error ?? '失败'}
        </span>
      )}
    </div>
  )
}
