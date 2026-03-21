'use client'
import { useRef, useEffect, useState, useCallback } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface GraphNode {
  id: string
  name: string
  type: string
  description?: string
  degree?: number
}

export interface GraphEdge {
  id: string
  src_id: string
  dst_id: string
  relation: string
  weight?: number
}

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  highlightIds?: string[]
  onNodeClick?: (node: GraphNode) => void
}

// ── Constants ─────────────────────────────────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  ORG:      '#3b82f6',
  PERSON:   '#f97316',
  PRODUCT:  '#a855f7',
  CONCEPT:  '#22c55e',
  EVENT:    '#ef4444',
  LOCATION: '#06b6d4',
  OTHER:    '#94a3b8',
}

function nodeColor(type: string): string {
  return TYPE_COLORS[type?.toUpperCase()] ?? TYPE_COLORS.OTHER
}

function nodeRadius(degree: number | undefined): number {
  return Math.min(8 + (degree ?? 1) * 2, 24)
}

// ── Simulation state (mutable, not React state) ───────────────────────────────

interface SimNode {
  id: string
  x: number
  y: number
  vx: number
  vy: number
  radius: number
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ForceGraphPanel({ nodes, edges, highlightIds = [], onNodeClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const simNodesRef = useRef<SimNode[]>([])
  const rafRef = useRef<number>(0)
  const lastRenderRef = useRef<number>(0)
  const transformRef = useRef({ x: 0, y: 0, scale: 1 })
  const isDraggingCanvasRef = useRef(false)
  const dragStartRef = useRef({ mx: 0, my: 0, tx: 0, ty: 0 })
  const hoveredEdgeRef = useRef<string | null>(null)
  const [renderTick, setRenderTick] = useState(0)
  const [size, setSize] = useState({ w: 800, h: 600 })
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null)

  // Resize observer
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setSize({ w: Math.max(width, 300), h: Math.max(height, 300) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Initialize / re-initialize sim nodes when nodes prop changes
  useEffect(() => {
    const cx = size.w / 2
    const cy = size.h / 2
    const existing = new Map<string, SimNode>(simNodesRef.current.map(n => [n.id, n]))

    simNodesRef.current = nodes.map(n => {
      if (existing.has(n.id)) {
        const prev = existing.get(n.id)!
        return { ...prev, radius: nodeRadius(n.degree) }
      }
      const angle = Math.random() * 2 * Math.PI
      const dist = 60 + Math.random() * 120
      return {
        id: n.id,
        x: cx + Math.cos(angle) * dist,
        y: cy + Math.sin(angle) * dist,
        vx: 0,
        vy: 0,
        radius: nodeRadius(n.degree),
      }
    })
  }, [nodes, size])

  // Force simulation loop
  useEffect(() => {
    if (nodes.length === 0) return

    const REPULSION = 3500
    const REPULSION_DIST = 180
    const ATTRACTION = 0.04
    const GRAVITY = 0.005
    const DAMPING = 0.9
    const TARGET_FPS = 30
    const FRAME_MS = 1000 / TARGET_FPS

    // Build adjacency for fast edge lookup
    const adjMap = new Map<string, string[]>()
    for (const e of edges) {
      if (!adjMap.has(e.src_id)) adjMap.set(e.src_id, [])
      if (!adjMap.has(e.dst_id)) adjMap.set(e.dst_id, [])
      adjMap.get(e.src_id)!.push(e.dst_id)
      adjMap.get(e.dst_id)!.push(e.src_id)
    }

    function tick() {
      const sn = simNodesRef.current
      if (sn.length === 0) {
        rafRef.current = requestAnimationFrame(tick)
        return
      }
      const cx = size.w / 2
      const cy = size.h / 2

      // 1. Repulsion between all pairs
      for (let i = 0; i < sn.length; i++) {
        for (let j = i + 1; j < sn.length; j++) {
          const a = sn[i], b = sn[j]
          const dx = b.x - a.x
          const dy = b.y - a.y
          const dist2 = dx * dx + dy * dy
          const dist = Math.sqrt(dist2) || 0.01
          if (dist < REPULSION_DIST) {
            const force = REPULSION / (dist2 + 1)
            const fx = (dx / dist) * force
            const fy = (dy / dist) * force
            a.vx -= fx
            a.vy -= fy
            b.vx += fx
            b.vy += fy
          }
        }
      }

      // 2. Edge attraction
      const idxMap = new Map<string, number>(sn.map((n, i) => [n.id, i]))
      for (const e of edges) {
        const ai = idxMap.get(e.src_id)
        const bi = idxMap.get(e.dst_id)
        if (ai === undefined || bi === undefined) continue
        const a = sn[ai], b = sn[bi]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01
        const idealDist = 100 + (a.radius + b.radius)
        const force = (dist - idealDist) * ATTRACTION
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        a.vx += fx
        a.vy += fy
        b.vx -= fx
        b.vy -= fy
      }

      // 3. Center gravity
      for (const n of sn) {
        n.vx += (cx - n.x) * GRAVITY
        n.vy += (cy - n.y) * GRAVITY
      }

      // 4. Damping + integrate + clamp
      const pad = 40
      for (const n of sn) {
        n.vx *= DAMPING
        n.vy *= DAMPING
        n.x += n.vx
        n.y += n.vy
        n.x = Math.max(pad + n.radius, Math.min(size.w - pad - n.radius, n.x))
        n.y = Math.max(pad + n.radius, Math.min(size.h - pad - n.radius, n.y))
      }

      // 5. Throttle re-render to 30fps
      const now = performance.now()
      if (now - lastRenderRef.current >= FRAME_MS) {
        lastRenderRef.current = now
        setRenderTick(t => t + 1)
      }

      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [nodes, edges, size])

  // Build quick lookup maps for rendering
  const nodeMap = new Map<string, GraphNode>(nodes.map(n => [n.id, n]))
  const simMap = new Map<string, SimNode>(simNodesRef.current.map(n => [n.id, n]))
  const highlightSet = new Set(highlightIds)

  // Pan handlers
  const onMouseDownCanvas = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if ((e.target as SVGElement).closest('.kg-node')) return
    isDraggingCanvasRef.current = true
    dragStartRef.current = {
      mx: e.clientX,
      my: e.clientY,
      tx: transformRef.current.x,
      ty: transformRef.current.y,
    }
  }, [])

  const onMouseMoveCanvas = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDraggingCanvasRef.current) return
    transformRef.current = {
      ...transformRef.current,
      x: dragStartRef.current.tx + (e.clientX - dragStartRef.current.mx),
      y: dragStartRef.current.ty + (e.clientY - dragStartRef.current.my),
    }
    setRenderTick(t => t + 1)
  }, [])

  const onMouseUpCanvas = useCallback(() => {
    isDraggingCanvasRef.current = false
  }, [])

  // Zoom handler
  const onWheel = useCallback((e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault()
    const factor = e.deltaY > 0 ? 0.9 : 1.1
    const newScale = Math.max(0.2, Math.min(4, transformRef.current.scale * factor))
    transformRef.current = { ...transformRef.current, scale: newScale }
    setRenderTick(t => t + 1)
  }, [])

  // Node drag
  const draggingNodeRef = useRef<string | null>(null)

  const onNodeMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation()
    draggingNodeRef.current = nodeId
  }, [])

  const onNodeMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!draggingNodeRef.current) return
    const sn = simNodesRef.current.find(n => n.id === draggingNodeRef.current)
    if (!sn) return
    const svgRect = svgRef.current!.getBoundingClientRect()
    const t = transformRef.current
    sn.x = (e.clientX - svgRect.left - t.x) / t.scale
    sn.y = (e.clientY - svgRect.top - t.y) / t.scale
    sn.vx = 0
    sn.vy = 0
    setRenderTick(tick => tick + 1)
  }, [])

  const onNodeMouseUp = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation()
    if (draggingNodeRef.current === nodeId) {
      draggingNodeRef.current = null
      const gn = nodeMap.get(nodeId)
      if (gn && onNodeClick) onNodeClick(gn)
    }
  }, [nodeMap, onNodeClick])

  const onSvgMouseUp = useCallback(() => {
    draggingNodeRef.current = null
    isDraggingCanvasRef.current = false
  }, [])

  const { x: tx, y: ty, scale } = transformRef.current

  if (nodes.length === 0) {
    return (
      <div
        ref={containerRef}
        className="w-full h-full flex items-center justify-center"
        style={{ background: '#0f1117' }}
      >
        <div className="text-center" style={{ color: '#475569' }}>
          <div className="text-4xl mb-3">🕸</div>
          <div className="text-sm">暂无图谱数据</div>
          <div className="text-xs mt-1">请先构建知识图谱或选择知识库</div>
        </div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden" style={{ background: '#0f1117' }}>
      <svg
        ref={svgRef}
        width={size.w}
        height={size.h}
        style={{ cursor: isDraggingCanvasRef.current ? 'grabbing' : 'grab', userSelect: 'none' }}
        onMouseDown={onMouseDownCanvas}
        onMouseMove={e => { onMouseMoveCanvas(e); onNodeMouseMove(e) }}
        onMouseUp={e => { onMouseUpCanvas(); onSvgMouseUp() }}
        onMouseLeave={onSvgMouseUp}
        onWheel={onWheel}
      >
        {/* Arrow marker defs */}
        <defs>
          <marker
            id="kg-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />
          </marker>
          <marker
            id="kg-arrow-highlight"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#7c3aed" />
          </marker>
          {/* Glow filter */}
          <filter id="kg-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <g transform={`translate(${tx},${ty}) scale(${scale})`}>
          {/* Edges */}
          {edges.map(edge => {
            const src = simMap.get(edge.src_id)
            const dst = simMap.get(edge.dst_id)
            if (!src || !dst) return null

            const dx = dst.x - src.x
            const dy = dst.y - src.y
            const dist = Math.sqrt(dx * dx + dy * dy) || 1
            const endX = dst.x - (dx / dist) * (dst.radius + 8)
            const endY = dst.y - (dy / dist) * (dst.radius + 8)
            const startX = src.x + (dx / dist) * (src.radius + 4)
            const startY = src.y + (dy / dist) * (src.radius + 4)

            const isHighlightEdge =
              highlightSet.has(edge.src_id) && highlightSet.has(edge.dst_id)
            const isHovered = hoveredEdge === edge.id

            // Midpoint for label
            const mx = (startX + endX) / 2
            const my = (startY + endY) / 2

            return (
              <g key={edge.id}>
                <line
                  x1={startX} y1={startY}
                  x2={endX} y2={endY}
                  stroke={isHighlightEdge ? '#7c3aed' : '#334155'}
                  strokeWidth={isHovered ? 2 : 1}
                  strokeOpacity={isHovered ? 1 : 0.6}
                  markerEnd={`url(#${isHighlightEdge ? 'kg-arrow-highlight' : 'kg-arrow'})`}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => { hoveredEdgeRef.current = edge.id; setHoveredEdge(edge.id) }}
                  onMouseLeave={() => { hoveredEdgeRef.current = null; setHoveredEdge(null) }}
                />
                {/* Invisible wider hit area */}
                <line
                  x1={startX} y1={startY}
                  x2={endX} y2={endY}
                  stroke="transparent"
                  strokeWidth={12}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => { hoveredEdgeRef.current = edge.id; setHoveredEdge(edge.id) }}
                  onMouseLeave={() => { hoveredEdgeRef.current = null; setHoveredEdge(null) }}
                />
                {/* Relation label on hover */}
                {isHovered && (
                  <g>
                    <rect
                      x={mx - edge.relation.length * 3.5 - 4}
                      y={my - 10}
                      width={edge.relation.length * 7 + 8}
                      height={18}
                      rx={4}
                      fill="#1e293b"
                      stroke="#475569"
                      strokeWidth={1}
                    />
                    <text
                      x={mx}
                      y={my + 4}
                      textAnchor="middle"
                      fontSize={11}
                      fill="#94a3b8"
                    >
                      {edge.relation}
                    </text>
                  </g>
                )}
              </g>
            )
          })}

          {/* Nodes */}
          {simNodesRef.current.map(sn => {
            const gn = nodeMap.get(sn.id)
            if (!gn) return null
            const color = nodeColor(gn.type)
            const isHighlighted = highlightSet.has(sn.id)
            const showLabel = (gn.degree ?? 0) >= 3 || isHighlighted

            return (
              <g
                key={sn.id}
                className="kg-node"
                transform={`translate(${sn.x},${sn.y})`}
                style={{ cursor: 'pointer' }}
                onMouseDown={e => onNodeMouseDown(e, sn.id)}
                onMouseUp={e => onNodeMouseUp(e, sn.id)}
              >
                {/* Glow ring for highlighted nodes */}
                {isHighlighted && (
                  <circle
                    r={sn.radius + 8}
                    fill="none"
                    stroke={color}
                    strokeWidth={2}
                    strokeOpacity={0.5}
                    filter="url(#kg-glow)"
                  />
                )}
                {/* Node circle */}
                <circle
                  r={sn.radius}
                  fill={color}
                  fillOpacity={0.85}
                  stroke={isHighlighted ? '#fff' : color}
                  strokeWidth={isHighlighted ? 2 : 1}
                />
                {/* Type letter */}
                <text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={Math.max(8, sn.radius * 0.7)}
                  fill="white"
                  fontWeight="600"
                  style={{ pointerEvents: 'none' }}
                >
                  {(gn.type?.[0] ?? '?').toUpperCase()}
                </text>
                {/* Label */}
                {showLabel && (
                  <text
                    y={sn.radius + 12}
                    textAnchor="middle"
                    fontSize={11}
                    fill="#e2e8f0"
                    stroke="#0f1117"
                    strokeWidth={3}
                    paintOrder="stroke"
                    style={{ pointerEvents: 'none' }}
                  >
                    {gn.name.length > 14 ? gn.name.slice(0, 13) + '…' : gn.name}
                  </text>
                )}
              </g>
            )
          })}
        </g>
      </svg>

      {/* Legend */}
      <div
        className="absolute bottom-3 left-3 text-xs rounded-lg p-2 space-y-1"
        style={{ background: 'rgba(15,17,23,0.85)', border: '1px solid #1e293b' }}
      >
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
            <span style={{ color: '#94a3b8' }}>{type}</span>
          </div>
        ))}
      </div>

      {/* Zoom hint */}
      <div
        className="absolute bottom-3 right-3 text-xs"
        style={{ color: '#334155' }}
      >
        滚轮缩放 · 拖拽平移 · 点击节点
      </div>
    </div>
  )
}
