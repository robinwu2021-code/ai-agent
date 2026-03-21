'use client'
import { useState } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ChainStep {
  src: string
  relation: string
  dst: string
  evidence?: string
}

interface Props {
  chain: ChainStep[]
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ReasoningTraceCard({ chain }: Props) {
  const [collapsed, setCollapsed] = useState(false)
  const [expandedStep, setExpandedStep] = useState<number | null>(null)

  if (!chain || chain.length === 0) return null

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        background: '#0f172a',
        border: '1px solid #1e293b',
      }}
    >
      {/* Header */}
      <button
        className="w-full flex items-center justify-between px-4 py-3 text-left transition-colors"
        style={{ borderBottom: collapsed ? 'none' : '1px solid #1e293b' }}
        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.03)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
        onClick={() => setCollapsed(c => !c)}
      >
        <div className="flex items-center gap-2">
          <span className="text-sm" style={{ color: '#7c3aed' }}>⛓</span>
          <span className="text-sm font-medium" style={{ color: '#e2e8f0' }}>
            推理链
          </span>
          <span
            className="text-xs px-1.5 py-0.5 rounded-full font-mono"
            style={{ background: '#1e293b', color: '#64748b' }}
          >
            {chain.length} 步
          </span>
        </div>
        <span
          className="text-xs transition-transform duration-200"
          style={{
            color: '#475569',
            display: 'inline-block',
            transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
          }}
        >
          ▾
        </span>
      </button>

      {/* Chain content */}
      {!collapsed && (
        <div className="px-4 py-4">
          {/* Horizontal flow */}
          <div className="flex items-start flex-wrap gap-0">
            {chain.map((step, i) => (
              <div key={i} className="flex items-center flex-shrink-0">
                {/* Source node pill (only for first step) */}
                {i === 0 && (
                  <NodePill name={step.src} />
                )}

                {/* Relation arrow */}
                <RelationArrow
                  relation={step.relation}
                  hasEvidence={!!step.evidence}
                  expanded={expandedStep === i}
                  onClick={() => setExpandedStep(expandedStep === i ? null : i)}
                />

                {/* Destination node pill */}
                <NodePill name={step.dst} isLast={i === chain.length - 1} />
              </div>
            ))}
          </div>

          {/* Evidence expanded view */}
          {expandedStep !== null && chain[expandedStep]?.evidence && (
            <div
              className="mt-3 rounded-lg p-3"
              style={{
                background: '#1e293b',
                border: '1px solid #334155',
              }}
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-medium uppercase tracking-wider" style={{ color: '#475569' }}>
                  证据来源
                </span>
                <span
                  className="text-xs px-1.5 py-0.5 rounded"
                  style={{ background: 'rgba(124,58,237,0.15)', color: '#a78bfa' }}
                >
                  步骤 {expandedStep + 1}：{chain[expandedStep].src} → {chain[expandedStep].dst}
                </span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: '#94a3b8', fontStyle: 'italic' }}>
                "{chain[expandedStep].evidence}"
              </p>
            </div>
          )}

          {/* Step-by-step list (fallback for long chains) */}
          {chain.length > 4 && (
            <div className="mt-4 space-y-2">
              <div className="text-xs font-medium uppercase tracking-wider" style={{ color: '#475569' }}>
                步骤详情
              </div>
              {chain.map((step, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-xs"
                >
                  <span
                    className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-mono font-bold"
                    style={{ background: '#1e293b', color: '#7c3aed', marginTop: '1px' }}
                  >
                    {i + 1}
                  </span>
                  <div className="flex items-center gap-1.5 flex-wrap" style={{ color: '#94a3b8' }}>
                    <span className="font-medium" style={{ color: '#e2e8f0' }}>{step.src}</span>
                    <span
                      className="px-1.5 py-0.5 rounded text-xs"
                      style={{ background: 'rgba(124,58,237,0.15)', color: '#a78bfa', fontWeight: 600 }}
                    >
                      {step.relation}
                    </span>
                    <span style={{ color: '#475569' }}>→</span>
                    <span className="font-medium" style={{ color: '#e2e8f0' }}>{step.dst}</span>
                    {step.evidence && (
                      <button
                        onClick={() => setExpandedStep(expandedStep === i ? null : i)}
                        className="text-xs underline ml-1"
                        style={{ color: '#475569' }}
                      >
                        {expandedStep === i ? '收起' : '查看证据'}
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function NodePill({ name, isLast = false }: { name: string; isLast?: boolean }) {
  return (
    <div
      className="px-3 py-1.5 rounded-full text-xs font-medium flex-shrink-0"
      style={{
        background: isLast ? 'rgba(124,58,237,0.2)' : '#1e293b',
        border: `1px solid ${isLast ? 'rgba(124,58,237,0.4)' : '#334155'}`,
        color: isLast ? '#a78bfa' : '#e2e8f0',
        maxWidth: '120px',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}
      title={name}
    >
      {name}
    </div>
  )
}

function RelationArrow({
  relation,
  hasEvidence,
  expanded,
  onClick,
}: {
  relation: string
  hasEvidence: boolean
  expanded: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="flex flex-col items-center mx-1 group flex-shrink-0"
      style={{ minWidth: '60px', cursor: hasEvidence ? 'pointer' : 'default' }}
      title={hasEvidence ? '点击查看证据' : undefined}
    >
      {/* Relation label */}
      <div
        className="text-xs font-medium px-1.5 py-0.5 rounded transition-colors"
        style={{
          background: expanded ? 'rgba(124,58,237,0.25)' : 'rgba(124,58,237,0.1)',
          color: '#a78bfa',
          border: `1px solid ${expanded ? 'rgba(124,58,237,0.5)' : 'rgba(124,58,237,0.2)'}`,
          maxWidth: '100px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          marginBottom: '2px',
        }}
        title={relation}
      >
        {relation}
      </div>
      {/* Arrow line */}
      <div className="flex items-center w-full">
        <div className="flex-1 h-px" style={{ background: '#334155' }} />
        <span style={{ color: '#475569', fontSize: '10px' }}>▶</span>
      </div>
      {/* Evidence indicator */}
      {hasEvidence && (
        <div
          className="text-xs mt-0.5 opacity-60"
          style={{ color: '#475569', fontSize: '9px' }}
        >
          {expanded ? '▴' : '▾'}
        </div>
      )}
    </button>
  )
}
