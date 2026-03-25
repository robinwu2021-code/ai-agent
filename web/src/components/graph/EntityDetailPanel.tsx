'use client'
import { useEffect, useRef } from 'react'
import type { GraphNode, GraphEdge } from './ForceGraphPanel'

// ── Type badge colors ─────────────────────────────────────────────────────────

const TYPE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  ORG:      { bg: 'rgba(59,130,246,0.15)',  text: '#93c5fd', border: 'rgba(59,130,246,0.35)' },
  PERSON:   { bg: 'rgba(249,115,22,0.15)',  text: '#fdba74', border: 'rgba(249,115,22,0.35)' },
  PRODUCT:  { bg: 'rgba(168,85,247,0.15)',  text: '#d8b4fe', border: 'rgba(168,85,247,0.35)' },
  CONCEPT:  { bg: 'rgba(34,197,94,0.15)',   text: '#86efac', border: 'rgba(34,197,94,0.35)' },
  EVENT:    { bg: 'rgba(239,68,68,0.15)',   text: '#fca5a5', border: 'rgba(239,68,68,0.35)' },
  LOCATION: { bg: 'rgba(6,182,212,0.15)',   text: '#67e8f9', border: 'rgba(6,182,212,0.35)' },
  OTHER:    { bg: 'rgba(148,163,184,0.15)', text: '#cbd5e1', border: 'rgba(148,163,184,0.35)' },
}

function typeBadgeStyle(type: string) {
  return TYPE_COLORS[type?.toUpperCase()] ?? TYPE_COLORS.OTHER
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  node: GraphNode | null
  edges: GraphEdge[]
  nodes: GraphNode[]
  onClose: () => void
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function EntityDetailPanel({ node, edges, nodes, onClose }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)

  // Focus panel when opened for accessibility
  useEffect(() => {
    if (node && panelRef.current) {
      panelRef.current.focus()
    }
  }, [node])

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  // Build node lookup map
  const nodeMap = new Map<string, GraphNode>(nodes.map(n => [n.id, n]))

  // Outgoing edges: node is src
  const outgoing = node
    ? edges.filter(e => e.src_id === node.id)
    : []

  // Incoming edges: node is dst
  const incoming = node
    ? edges.filter(e => e.dst_id === node.id)
    : []

  const badgeStyle = node ? typeBadgeStyle(node.type) : typeBadgeStyle('OTHER')

  return (
    <div
      ref={panelRef}
      tabIndex={-1}
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        height: '100%',
        width: node ? '300px' : '0px',
        overflow: 'hidden',
        transition: 'width 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        background: '#0f172a',
        borderLeft: node ? '1px solid #1e293b' : 'none',
        zIndex: 10,
        outline: 'none',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {node && (
        <>
          {/* Header */}
          <div
            className="flex items-start justify-between gap-2 px-4 py-4"
            style={{ borderBottom: '1px solid #1e293b', flexShrink: 0 }}
          >
            <div className="min-w-0 flex-1">
              <h2
                className="font-semibold text-base leading-snug truncate"
                style={{ color: '#f1f5f9' }}
                title={node.name}
              >
                {node.name}
              </h2>
              {/* Type badge */}
              <span
                className="inline-block mt-1.5 text-xs font-medium px-2 py-0.5 rounded-full"
                style={{
                  background: badgeStyle.bg,
                  color: badgeStyle.text,
                  border: `1px solid ${badgeStyle.border}`,
                }}
              >
                {node.type?.toUpperCase() || 'OTHER'}
              </span>
            </div>
            {/* Close button */}
            <button
              onClick={onClose}
              className="flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-md transition-colors"
              style={{ color: '#64748b', background: 'transparent' }}
              onMouseEnter={e => (e.currentTarget.style.background = '#1e293b')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
              aria-label="关闭"
            >
              ✕
            </button>
          </div>

          {/* Scrollable body */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">

            {/* Description */}
            {node.description && (
              <section>
                <div className="text-xs font-medium uppercase tracking-wider mb-1.5" style={{ color: '#475569' }}>
                  描述
                </div>
                <p className="text-sm leading-relaxed" style={{ color: '#94a3b8' }}>
                  {node.description}
                </p>
              </section>
            )}

            {/* Degree */}
            <section>
              <div className="text-xs font-medium uppercase tracking-wider mb-1.5" style={{ color: '#475569' }}>
                连接度
              </div>
              <div className="text-sm font-mono" style={{ color: '#e2e8f0' }}>
                {node.degree ?? (outgoing.length + incoming.length)} 条关系
              </div>
            </section>

            {/* Outgoing relations */}
            {outgoing.length > 0 && (
              <section>
                <div className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color: '#475569' }}>
                  出向关系
                </div>
                <div className="space-y-1.5">
                  {outgoing.map(edge => {
                    const target = nodeMap.get(edge.dst_id)
                    const targetName = target?.name ?? edge.dst_id
                    const targetBadge = typeBadgeStyle(target?.type ?? 'OTHER')
                    return (
                      <div
                        key={edge.id}
                        className="flex items-center gap-1.5 text-xs flex-wrap"
                        style={{ color: '#64748b' }}
                      >
                        {/* Relation arrow */}
                        <span
                          className="px-1.5 py-0.5 rounded text-xs"
                          style={{ background: '#1e293b', color: '#7c3aed', fontWeight: 600 }}
                        >
                          {edge.relation}
                        </span>
                        <span style={{ color: '#475569' }}>→</span>
                        <span
                          className="px-1.5 py-0.5 rounded-full text-xs font-medium"
                          style={{
                            background: targetBadge.bg,
                            color: targetBadge.text,
                            border: `1px solid ${targetBadge.border}`,
                            maxWidth: '120px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={targetName}
                        >
                          {targetName}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </section>
            )}

            {/* Incoming relations */}
            {incoming.length > 0 && (
              <section>
                <div className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color: '#475569' }}>
                  入向关系
                </div>
                <div className="space-y-1.5">
                  {incoming.map(edge => {
                    const source = nodeMap.get(edge.src_id)
                    const sourceName = source?.name ?? edge.src_id
                    const sourceBadge = typeBadgeStyle(source?.type ?? 'OTHER')
                    return (
                      <div
                        key={edge.id}
                        className="flex items-center gap-1.5 text-xs flex-wrap"
                        style={{ color: '#64748b' }}
                      >
                        <span
                          className="px-1.5 py-0.5 rounded-full text-xs font-medium"
                          style={{
                            background: sourceBadge.bg,
                            color: sourceBadge.text,
                            border: `1px solid ${sourceBadge.border}`,
                            maxWidth: '120px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={sourceName}
                        >
                          {sourceName}
                        </span>
                        <span style={{ color: '#475569' }}>→</span>
                        <span
                          className="px-1.5 py-0.5 rounded text-xs"
                          style={{ background: '#1e293b', color: '#7c3aed', fontWeight: 600 }}
                        >
                          {edge.relation}
                        </span>
                        <span style={{ color: '#475569' }}>→</span>
                        <span style={{ color: '#94a3b8', fontWeight: 500 }}>（本节点）</span>
                      </div>
                    )
                  })}
                </div>
              </section>
            )}

            {/* No relations */}
            {outgoing.length === 0 && incoming.length === 0 && (
              <div className="text-xs text-center py-4" style={{ color: '#334155' }}>
                该实体暂无已加载的关系
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
