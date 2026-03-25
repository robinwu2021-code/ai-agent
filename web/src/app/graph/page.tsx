'use client'
import { useState, useEffect, useCallback, useMemo } from 'react'
import ForceGraphPanel, { type GraphNode, type GraphEdge } from '@/components/graph/ForceGraphPanel'
import EntityDetailPanel from '@/components/graph/EntityDetailPanel'

// ── Types ─────────────────────────────────────────────────────────────────────

interface GraphStats {
  total_nodes: number
  total_edges: number
  kb_id: string
  node_types: Record<string, number>
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

type NodeTypeFilter = 'ALL' | 'ORG' | 'PERSON' | 'PRODUCT' | 'CONCEPT' | 'EVENT' | 'LOCATION'

const TYPE_FILTER_LABELS: NodeTypeFilter[] = [
  'ALL', 'ORG', 'PERSON', 'PRODUCT', 'CONCEPT', 'EVENT', 'LOCATION',
]

const TYPE_COLORS: Record<string, string> = {
  ORG:      '#3b82f6',
  PERSON:   '#f97316',
  PRODUCT:  '#a855f7',
  CONCEPT:  '#22c55e',
  EVENT:    '#ef4444',
  LOCATION: '#06b6d4',
  ALL:      '#7c3aed',
}

// ── API helpers ───────────────────────────────────────────────────────────────

async function fetchGraphData(kbId: string): Promise<GraphData> {
  const res = await fetch(`/api/agent/kg/graph?kb_id=${encodeURIComponent(kbId)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
  return res.json()
}

async function fetchGraphStats(kbId: string): Promise<GraphStats> {
  const res = await fetch(`/api/agent/kg/stats?kb_id=${encodeURIComponent(kbId)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
  return res.json()
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function GraphPage() {
  const [kbId] = useState('global')
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<NodeTypeFilter>('ALL')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // ── Data fetching ────────────────────────────────────────────────────────

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [data, statsData] = await Promise.all([
        fetchGraphData(kbId),
        fetchGraphStats(kbId),
      ])
      setGraphData(data)
      setStats(statsData)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [kbId])

  useEffect(() => {
    loadData()
  }, [loadData])

  // ── Filtering ────────────────────────────────────────────────────────────

  const filteredNodes = useMemo(() => {
    let result = graphData.nodes
    if (typeFilter !== 'ALL') {
      result = result.filter(n => n.type?.toUpperCase() === typeFilter)
    }
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase()
      result = result.filter(
        n =>
          n.name.toLowerCase().includes(q) ||
          n.description?.toLowerCase().includes(q) ||
          n.type?.toLowerCase().includes(q)
      )
    }
    return result
  }, [graphData.nodes, typeFilter, searchQuery])

  const filteredNodeIds = useMemo(
    () => new Set(filteredNodes.map(n => n.id)),
    [filteredNodes]
  )

  const filteredEdges = useMemo(
    () => graphData.edges.filter(
      e => filteredNodeIds.has(e.src_id) && filteredNodeIds.has(e.dst_id)
    ),
    [graphData.edges, filteredNodeIds]
  )

  // Highlight IDs: nodes matching search
  const highlightIds = useMemo(() => {
    if (!searchQuery.trim()) return []
    const q = searchQuery.trim().toLowerCase()
    return graphData.nodes
      .filter(n => n.name.toLowerCase().includes(q))
      .map(n => n.id)
  }, [graphData.nodes, searchQuery])

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(prev => prev?.id === node.id ? null : node)
  }, [])

  const handleCloseDetail = useCallback(() => {
    setSelectedNode(null)
  }, [])

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div
      className="flex flex-col h-screen overflow-hidden"
      style={{ background: '#0a0d14', color: '#e2e8f0' }}
    >
      {/* Top bar */}
      <header
        className="flex items-center gap-4 px-5 py-3 flex-shrink-0"
        style={{ background: '#0f1117', borderBottom: '1px solid #1e293b' }}
      >
        {/* Title */}
        <div className="flex items-center gap-2.5 mr-2">
          <span className="text-xl">🕸</span>
          <div>
            <h1 className="text-sm font-semibold leading-none" style={{ color: '#f1f5f9' }}>
              知识图谱
            </h1>
            <div className="text-xs mt-0.5" style={{ color: '#475569' }}>
              Knowledge Graph Explorer
            </div>
          </div>
        </div>

        {/* Stats badges */}
        {stats && (
          <div className="flex items-center gap-3">
            <StatBadge label="节点" value={stats.total_nodes} color="#3b82f6" />
            <StatBadge label="关系" value={stats.total_edges} color="#7c3aed" />
            <StatBadge label="显示" value={filteredNodes.length} color="#22c55e" />
          </div>
        )}

        <div className="flex-1" />

        {/* Sidebar toggle */}
        <button
          onClick={() => setSidebarOpen(o => !o)}
          className="text-xs px-2.5 py-1.5 rounded-md transition-colors"
          style={{
            background: sidebarOpen ? 'rgba(124,58,237,0.15)' : '#1e293b',
            border: `1px solid ${sidebarOpen ? 'rgba(124,58,237,0.35)' : '#334155'}`,
            color: sidebarOpen ? '#a78bfa' : '#64748b',
          }}
        >
          {sidebarOpen ? '◀ 收起' : '▶ 展开'}
        </button>

        {/* Reload button */}
        <button
          onClick={loadData}
          disabled={loading}
          className="text-xs px-2.5 py-1.5 rounded-md transition-colors flex items-center gap-1.5"
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            color: loading ? '#334155' : '#64748b',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          <span className={loading ? 'animate-spin inline-block' : ''}>↻</span>
          刷新
        </button>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left sidebar */}
        <aside
          className="flex flex-col flex-shrink-0 overflow-hidden transition-all duration-300"
          style={{
            width: sidebarOpen ? '240px' : '0px',
            borderRight: sidebarOpen ? '1px solid #1e293b' : 'none',
            background: '#0f1117',
          }}
        >
          {sidebarOpen && (
            <>
              {/* Search */}
              <div className="px-3 pt-3 pb-2" style={{ borderBottom: '1px solid #1e293b' }}>
                <div className="relative">
                  <span
                    className="absolute left-2.5 top-1/2 -translate-y-1/2 text-xs"
                    style={{ color: '#475569', pointerEvents: 'none' }}
                  >
                    🔍
                  </span>
                  <input
                    type="text"
                    placeholder="搜索节点..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    className="w-full pl-7 pr-3 py-2 text-xs rounded-lg outline-none"
                    style={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      color: '#e2e8f0',
                    }}
                    onFocus={e => (e.currentTarget.style.borderColor = '#7c3aed')}
                    onBlur={e => (e.currentTarget.style.borderColor = '#334155')}
                  />
                </div>
              </div>

              {/* Type filter chips */}
              <div className="px-3 py-2 flex flex-wrap gap-1.5" style={{ borderBottom: '1px solid #1e293b' }}>
                {TYPE_FILTER_LABELS.map(t => {
                  const color = TYPE_COLORS[t] ?? '#7c3aed'
                  const active = typeFilter === t
                  const count = t === 'ALL'
                    ? graphData.nodes.length
                    : graphData.nodes.filter(n => n.type?.toUpperCase() === t).length
                  return (
                    <button
                      key={t}
                      onClick={() => setTypeFilter(t)}
                      className="text-xs px-2 py-0.5 rounded-full transition-all"
                      style={{
                        background: active ? `${color}22` : 'transparent',
                        border: `1px solid ${active ? color : '#1e293b'}`,
                        color: active ? color : '#475569',
                      }}
                    >
                      {t} {count > 0 && <span className="opacity-60">({count})</span>}
                    </button>
                  )
                })}
              </div>

              {/* Node list */}
              <div className="flex-1 overflow-y-auto py-2">
                {loading ? (
                  <div className="text-xs text-center py-8" style={{ color: '#334155' }}>
                    加载中…
                  </div>
                ) : filteredNodes.length === 0 ? (
                  <div className="text-xs text-center py-8" style={{ color: '#334155' }}>
                    {searchQuery ? '无匹配节点' : '暂无数据'}
                  </div>
                ) : (
                  filteredNodes.slice(0, 200).map(node => {
                    const color = TYPE_COLORS[node.type?.toUpperCase()] ?? '#94a3b8'
                    const isSelected = selectedNode?.id === node.id
                    return (
                      <button
                        key={node.id}
                        onClick={() => handleNodeClick(node)}
                        className="w-full text-left px-3 py-2 transition-colors flex items-start gap-2"
                        style={{
                          background: isSelected ? 'rgba(124,58,237,0.1)' : 'transparent',
                          borderLeft: isSelected ? '2px solid #7c3aed' : '2px solid transparent',
                        }}
                        onMouseEnter={e => {
                          if (!isSelected) e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
                        }}
                        onMouseLeave={e => {
                          if (!isSelected) e.currentTarget.style.background = 'transparent'
                        }}
                      >
                        <div
                          className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5"
                          style={{ background: `${color}22`, border: `1px solid ${color}44`, color }}
                        >
                          {node.type?.[0]?.toUpperCase() ?? '?'}
                        </div>
                        <div className="min-w-0">
                          <div className="text-xs font-medium truncate" style={{ color: isSelected ? '#a78bfa' : '#cbd5e1' }}>
                            {node.name}
                          </div>
                          <div className="text-xs truncate" style={{ color: '#475569' }}>
                            {node.type} · {node.degree ?? 0} 关系
                          </div>
                        </div>
                      </button>
                    )
                  })
                )}
                {filteredNodes.length > 200 && (
                  <div className="text-xs text-center py-2" style={{ color: '#334155' }}>
                    显示前 200 个节点（共 {filteredNodes.length} 个）
                  </div>
                )}
              </div>
            </>
          )}
        </aside>

        {/* Main graph area */}
        <main className="flex-1 relative overflow-hidden">
          {/* Error overlay */}
          {error && (
            <div
              className="absolute inset-x-4 top-4 z-20 rounded-xl p-4 text-sm"
              style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}
            >
              <div className="font-medium mb-1">加载失败</div>
              <div className="text-xs opacity-80">{error}</div>
              <button
                onClick={loadData}
                className="mt-2 text-xs underline"
                style={{ color: '#f87171' }}
              >
                重试
              </button>
            </div>
          )}

          {/* Loading overlay */}
          {loading && (
            <div
              className="absolute inset-0 z-10 flex items-center justify-center"
              style={{ background: 'rgba(15,17,23,0.7)', backdropFilter: 'blur(2px)' }}
            >
              <div className="text-center">
                <div className="text-2xl mb-2 animate-spin inline-block">↻</div>
                <div className="text-sm" style={{ color: '#94a3b8' }}>加载知识图谱…</div>
              </div>
            </div>
          )}

          {/* Force graph */}
          <ForceGraphPanel
            nodes={filteredNodes}
            edges={filteredEdges}
            highlightIds={highlightIds}
            onNodeClick={handleNodeClick}
          />

          {/* Entity detail panel (slides in from right) */}
          <EntityDetailPanel
            node={selectedNode}
            edges={graphData.edges}
            nodes={graphData.nodes}
            onClose={handleCloseDetail}
          />
        </main>
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatBadge({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div
      className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md"
      style={{ background: `${color}15`, border: `1px solid ${color}30` }}
    >
      <span style={{ color }}>{label}</span>
      <span className="font-mono font-bold" style={{ color }}>
        {value.toLocaleString()}
      </span>
    </div>
  )
}
