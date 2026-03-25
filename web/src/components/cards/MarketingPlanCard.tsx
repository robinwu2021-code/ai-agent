import type { MarketingData } from '@/types'

const CHANNEL_ICONS: Record<string, string> = {
  微信: '💬', 抖音: '🎵', 小红书: '📕', 外卖: '🛵', 微博: '🔴',
  线下: '🏪', 私域: '👥', 社群: '👥', 朋友圈: '💬',
}

function getChannelIcon(channel: string): string {
  for (const [k, v] of Object.entries(CHANNEL_ICONS)) {
    if (channel.includes(k)) return v
  }
  return '📣'
}

interface Props { data: MarketingData }

export default function MarketingPlanCard({ data }: Props) {
  return (
    <div className="rounded-2xl overflow-hidden"
      style={{ background: 'var(--elevated)', border: '1px solid rgba(251,146,60,0.25)' }}>
      {/* Header */}
      <div className="px-5 py-4"
        style={{ background: 'linear-gradient(135deg, rgba(251,146,60,0.12) 0%, rgba(239,68,68,0.08) 100%)', borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">📢</span>
          <span className="font-semibold text-sm" style={{ color: '#f1f5f9' }}>
            {data.plan_name ?? '营销方案'}
          </span>
        </div>
        {data.merchant && (
          <div className="text-xs" style={{ color: 'rgba(251,146,60,0.8)' }}>
            🏪 {data.merchant}
          </div>
        )}
        {data.goal && (
          <div className="mt-2 text-sm" style={{ color: '#cbd5e1' }}>{data.goal}</div>
        )}
      </div>

      {/* Activities */}
      {data.activities && data.activities.length > 0 && (
        <div className="px-5 py-4">
          <div className="text-xs font-medium mb-3 uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
            活动方案 ({data.activities.length})
          </div>
          <div className="space-y-3">
            {data.activities.map((act, i) => (
              <div key={i} className="rounded-xl p-4"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="font-medium text-sm" style={{ color: '#f1f5f9' }}>
                    <span className="text-xs mr-2 px-1.5 py-0.5 rounded"
                      style={{ background: 'rgba(251,146,60,0.15)', color: '#fb923c' }}>
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    {act.name}
                  </div>
                  {act.type && (
                    <span className="text-xs px-2 py-0.5 rounded-full flex-shrink-0"
                      style={{ background: 'rgba(251,146,60,0.1)', color: '#fb923c', border: '1px solid rgba(251,146,60,0.2)' }}>
                      {act.type}
                    </span>
                  )}
                </div>
                {act.description && (
                  <div className="text-xs leading-relaxed mb-2" style={{ color: '#94a3b8' }}>{act.description}</div>
                )}
                <div className="flex flex-wrap gap-3 text-xs" style={{ color: 'var(--muted)' }}>
                  {act.timeline && (
                    <span>🗓 {act.timeline}</span>
                  )}
                  {act.budget && (
                    <span>💰 {act.budget}</span>
                  )}
                  {act.expected_outcome && (
                    <span>🎯 {act.expected_outcome}</span>
                  )}
                </div>
                {act.channels && act.channels.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {act.channels.map((c, ci) => (
                      <span key={ci} className="text-xs px-2 py-0.5 rounded-full"
                        style={{ background: 'rgba(255,255,255,0.06)', color: '#cbd5e1' }}>
                        {getChannelIcon(c)} {c}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary + next steps */}
      {(data.summary || (data.next_steps && data.next_steps.length > 0)) && (
        <div className="px-5 pb-5">
          {data.summary && (
            <div className="text-xs leading-relaxed mb-3" style={{ color: '#94a3b8' }}>{data.summary}</div>
          )}
          {data.next_steps && data.next_steps.length > 0 && (
            <>
              <div className="text-xs font-medium mb-2 uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
                下一步行动
              </div>
              <div className="space-y-1.5">
                {data.next_steps.map((step, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs" style={{ color: '#cbd5e1' }}>
                    <span className="mt-0.5 w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0 text-xs"
                      style={{ background: 'rgba(251,146,60,0.2)', color: '#fb923c' }}>
                      {i + 1}
                    </span>
                    {step}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
