'use client'

import { useState, useEffect, useRef, useCallback, DragEvent, useMemo } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface KBDoc {
  doc_id: string
  filename: string
  source: string
  doc_type: string
  char_count: number
  chunk_count: number
  status: 'pending' | 'indexing' | 'ready' | 'error'
  error_msg?: string
  created_at: number
  kb_id: string
}

interface Citation {
  index: number
  source: string
  filename: string
  text_preview: string
  score?: number
}

interface ProcessStep {
  id: string
  label: string
  state: 'waiting' | 'running' | 'done' | 'error'
  detail?: string
}

interface KbMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  timestamp: Date
  streaming?: boolean
  thinkingElapsed?: number
  processSteps?: ProcessStep[]
  scopeLabel?: string  // e.g. "全部文档" | "3 份文档"
}

interface KBStats {
  doc_count: number
  chunk_count: number
  total_chars: number
}

interface UploadProgressEvent {
  type: 'progress' | 'done' | 'error' | 'heartbeat'
  step?: 'parsing' | 'chunking' | 'embedding'
  chunk_idx?: number
  total?: number
  progress?: number
  message?: string
  doc_id?: string
  chunk_count?: number
  status?: string
  filename?: string
}

interface UploadTask {
  id: string
  filename: string
  filesize: number
  progress: number
  currentStep: 'parsing' | 'chunking' | 'embedding' | 'done' | 'error'
  embeddingProgress: { current: number; total: number } | null
  done: boolean
  error: string | null
}

interface GraphNode {
  id: string
  name: string
  type: string
  description: string
  degree: number
  x?: number
  y?: number
  vx?: number
  vy?: number
}

interface GraphEdge {
  id: string
  src_id: string
  dst_id: string
  relation: string
  weight: number
}

interface KGStats {
  node_count?: number
  edge_count?: number
  entity_count?: number
  relation_count?: number
}

// ─── Constants ────────────────────────────────────────────────────────────────

const KB_ID = 'global'
const DEMO_PROMPTS = [
  '智星平台有哪些功能？',
  '公司创始人是谁？',
  '产品定价是多少？',
  '星辰科技的竞争对手有哪些？',
  '如何绑定微信公众号？',
]
const DEMO_DOCS = [
  {
    title: 'company_overview.md',
    content: `# 星辰科技有限公司 企业概览\n\n## 公司简介\n\n星辰科技有限公司（Starlight Technology Co., Ltd）成立于2018年，总部位于上海市浦东新区张江高科技园区。公司专注于人工智能驱动的企业数字化营销与数据智能领域，致力于帮助中国企业实现精准营销、智能运营与数据驱动决策。经过六年深耕，星辰科技已成长为国内AI营销领域的领军企业之一，旗下产品覆盖营销自动化、数据分析与智能客服三大核心场景。\n\n## 核心管理团队\n\n公司创始人兼首席执行官**张明远**，毕业于复旦大学计算机科学专业，拥有15年互联网与AI从业经验，曾先后任职于百度数据智能部门及微软亚洲研究院。首席技术官**李晓红**，毕业于清华大学人工智能研究院，深耕自然语言处理与机器学习方向，拥有20余项AI相关专利，主导构建了星辰科技的核心算法引擎。\n\n## 核心产品线\n\n**智星 AI营销平台**是公司旗舰产品，基于大语言模型与多模态AI技术，提供多渠道营销内容自动生成、用户画像分析、智能投放策略及ROI追踪等功能。智星平台已集成微信公众号、抖音、微博、淘宝等主流流量渠道，目前服务超过3000家企业客户。\n\n**数睿数据分析工具**专为中型以上企业设计，提供实时数据大屏、多维度漏斗分析、用户行为追踪及预测模型构建能力。\n\n**星客智能客服系统**融合对话式AI与知识图谱技术，支持全渠道接入、意图识别、情绪分析及人机协同等功能，平均响应时延低于200毫秒，客户满意度达92%。\n\n## 竞争格局\n\n主要竞争对手包括**明略科技**和**神策数据**。星辰科技以全栈AI能力与一体化产品矩阵形成差异化竞争优势。`,
  },
  {
    title: 'product_manual.md',
    content: `# 智星 AI营销平台 产品使用手册\n\n## 一、产品概述\n\n智星AI营销平台是由星辰科技有限公司研发的一站式智能营销解决方案。\n\n## 二、核心功能模块\n\n### 2.1 多渠道营销管理\n\n支持统一工作台管理微信公众号、抖音企业号、微博蓝V、淘宝店铺等渠道。\n\n操作步骤：\n1. 进入"渠道管理"菜单，点击"绑定渠道"\n2. 选择目标平台，按提示完成OAuth授权\n3. 授权成功后，渠道状态显示为"已连接"\n\n### 2.2 AI内容生成\n\n基于GPT级大语言模型，输入产品关键词、目标人群与营销目的，平台可在10秒内自动生成标题、正文、标签及配图建议。\n\n## 三、定价方案\n\n| 版本 | 月费 | 适用规模 |\n|------|------|----------|\n| 基础版 | ¥1,999/月 | 1-5人团队 |\n| 专业版 | ¥4,999/月 | 5-50人团队 |\n| 企业版 | 定制报价 | 50人以上 |\n\n## 四、技术支持\n\n- 邮件支持：support@starlight-ai.com\n- 电话支持：400-888-9999（工作日 9:00-18:00）`,
  },
  {
    title: 'faq.md',
    content: `# 星辰科技 · 智星AI营销平台 综合问答手册\n\nQ1：智星平台是哪家公司的产品？\nA：智星AI营销平台由**星辰科技有限公司**自主研发，成立于2018年，总部位于上海张江高科技园区。\n\nQ2：如何注册并开通智星平台账号？\nA：访问 https://app.starlight-ai.com，点击"免费试用"，填写企业信息完成手机验证码校验后激活14天免费试用。\n\nQ3：智星平台支持哪些内容发布渠道？\nA：支持微信公众号、抖音企业号、微博蓝V、淘宝店铺。\n\nQ4：智星平台的定价体系是怎样的？\nA：基础版¥1,999/月、专业版¥4,999/月、企业版定制报价。年付享受8.5折优惠。\n\nQ5：星辰科技获得了哪些融资？\nA：2021年完成B轮融资5000万美元，由红杉资本中国领投，腾讯投资和云九资本跟投。\n\nQ8：智星平台的竞争对手是哪些？\nA：主要竞争对手为明略科技和神策数据。星辰科技优势在于一体化平台与低上手门槛。`,
  },
]

const NODE_TYPE_COLOR: Record<string, string> = {
  PERSON: '#a78bfa', ORGANIZATION: '#60a5fa', LOCATION: '#34d399',
  EVENT: '#fb923c',  CONCEPT: '#f472b6',      PRODUCT: '#67e8f9',
  DEFAULT: '#94a3b8',
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDate(ts: number): string {
  const d = new Date(ts * 1000)
  const diffMin = Math.floor((Date.now() - d.getTime()) / 60000)
  if (diffMin < 1) return '刚刚'
  if (diffMin < 60) return `${diffMin}m 前`
  if (diffMin < 1440) return `${Math.floor(diffMin / 60)}h 前`
  return d.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}
function fileIcon(name: string): string {
  const e = name.split('.').pop()?.toLowerCase()
  return e === 'pdf' ? '📄' : e === 'md' ? '📝' : e === 'txt' ? '📃' : e === 'docx' || e === 'doc' ? '📋' : e === 'html' || e === 'htm' ? '🌐' : '📁'
}
function fileTypeColor(name: string): string {
  const e = name.split('.').pop()?.toLowerCase()
  return e === 'pdf' ? '#f87171' : e === 'md' ? '#a78bfa' : e === 'txt' ? '#94a3b8' : e === 'docx' || e === 'doc' ? '#60a5fa' : e === 'html' || e === 'htm' ? '#fb923c' : '#64748b'
}
function formatBytes(n: number): string {
  if (n < 1000) return `${n}字`; if (n < 100000) return `${(n / 1000).toFixed(1)}K字`; return `${Math.round(n / 1000)}K字`
}
function formatFileSize(b: number): string {
  if (b < 1024) return `${b}B`; if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`; return `${(b / (1024 * 1024)).toFixed(1)}MB`
}

// ─── Upload Step Pipeline ─────────────────────────────────────────────────────

const UPLOAD_STEPS: { key: UploadTask['currentStep']; label: string }[] = [
  { key: 'parsing', label: '解析' }, { key: 'chunking', label: '分块' },
  { key: 'embedding', label: '向量化' }, { key: 'done', label: '完成' },
]

function UploadStepPipeline({ task }: { task: UploadTask }) {
  const stepOrder = ['parsing', 'chunking', 'embedding', 'done']
  const ci = stepOrder.indexOf(task.currentStep)
  return (
    <div className="rounded-xl px-3 py-2.5" style={{
      background: task.error ? 'rgba(239,68,68,0.06)' : task.done ? 'rgba(52,211,153,0.06)' : 'rgba(124,58,237,0.08)',
      border: `1px solid ${task.error ? 'rgba(239,68,68,0.2)' : task.done ? 'rgba(52,211,153,0.2)' : 'rgba(124,58,237,0.2)'}`,
    }}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-sm flex-shrink-0">{fileIcon(task.filename)}</span>
          <span className="text-xs font-medium truncate" style={{ color: task.error ? '#f87171' : task.done ? '#6ee7b7' : '#e2e8f0' }}>{task.filename}</span>
        </div>
        <span className="text-xs flex-shrink-0" style={{ color: '#475569' }}>{formatFileSize(task.filesize)}</span>
      </div>
      {task.error ? <p className="text-xs" style={{ color: '#f87171' }}>✕ {task.error}</p> : (
        <>
          <div className="flex items-center gap-0 mb-2">
            {UPLOAD_STEPS.map((step, idx) => {
              const isDone = task.done || ci > idx, isActive = !task.done && ci === idx
              return (
                <div key={step.key} className="flex items-center flex-1">
                  <div className="flex flex-col items-center" style={{ minWidth: 0 }}>
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs transition-all${isActive ? ' animate-pulse' : ''}`}
                      style={{ background: isDone ? 'rgba(52,211,153,0.2)' : isActive ? 'rgba(124,58,237,0.35)' : 'rgba(255,255,255,0.05)', border: `1px solid ${isDone ? 'rgba(52,211,153,0.5)' : isActive ? 'rgba(124,58,237,0.6)' : 'rgba(255,255,255,0.1)'}`, color: isDone ? '#34d399' : isActive ? '#a78bfa' : '#475569' }}>
                      {isDone ? '✓' : isActive ? '…' : idx + 1}
                    </div>
                    <span style={{ fontSize: 10, color: isDone ? '#34d399' : isActive ? '#a78bfa' : '#334155' }}>{step.label}</span>
                  </div>
                  {idx < UPLOAD_STEPS.length - 1 && <div className="flex-1 h-px mx-1" style={{ background: ci > idx ? 'rgba(52,211,153,0.4)' : 'rgba(255,255,255,0.06)' }} />}
                </div>
              )
            })}
          </div>
          {!task.done && (
            <>
              <div className="h-1 rounded-full overflow-hidden mb-1" style={{ background: 'rgba(255,255,255,0.05)' }}>
                <div className="h-full rounded-full transition-all duration-300" style={{ width: `${task.progress}%`, background: 'linear-gradient(90deg,#6d28d9,#a78bfa)' }} />
              </div>
              {task.embeddingProgress && <p className="text-xs" style={{ color: '#64748b' }}>向量化 {task.embeddingProgress.current}/{task.embeddingProgress.total} 片段</p>}
            </>
          )}
          {task.done && <p className="text-xs" style={{ color: '#34d399' }}>✓ 索引完成</p>}
        </>
      )}
    </div>
  )
}

// ─── Process Timeline ─────────────────────────────────────────────────────────

function ProcessTimeline({ steps, collapsed }: { steps: ProcessStep[]; collapsed: boolean }) {
  if (!steps.length || collapsed) return null
  return (
    <div className="flex flex-col gap-1 mb-2 px-1">
      {steps.map(s => (
        <div key={s.id} className="flex items-start gap-2">
          <span className={`text-xs w-3.5 flex-shrink-0 mt-0.5${s.state === 'running' ? ' animate-pulse' : ''}`}
            style={{ color: s.state === 'done' ? '#34d399' : s.state === 'running' ? '#a78bfa' : s.state === 'error' ? '#f87171' : '#334155' }}>
            {s.state === 'done' ? '✓' : s.state === 'running' ? '◎' : s.state === 'error' ? '✕' : '○'}
          </span>
          <span className="text-xs" style={{ color: s.state === 'done' ? '#64748b' : s.state === 'running' ? '#94a3b8' : '#334155' }}>
            {s.label}
            {s.detail && s.state !== 'waiting' && <span className="ml-1.5" style={{ color: s.state === 'done' ? '#4ade80' : '#64748b' }}>{s.detail}</span>}
          </span>
        </div>
      ))}
    </div>
  )
}

// ─── Status Dot ───────────────────────────────────────────────────────────────

function StatusDot({ status }: { status: KBDoc['status'] }) {
  const c = { ready: { c: '#4ade80', b: '#052e16', bd: '#166534', l: '就绪' }, indexing: { c: '#60a5fa', b: '#0c1a3d', bd: '#1e40af', l: '索引中' }, error: { c: '#f87171', b: '#2d0a0a', bd: '#991b1b', l: '错误' }, pending: { c: '#94a3b8', b: '#1c1c2e', bd: '#334155', l: '等待' } }[status]
  return <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium${status === 'indexing' ? ' animate-pulse' : ''}`} style={{ background: c.b, color: c.c, border: `1px solid ${c.bd}` }}><span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: c.c }} />{c.l}</span>
}

// ─── Toast ────────────────────────────────────────────────────────────────────

function Toast({ message, type = 'info', onDone }: { message: string; type?: 'info' | 'success' | 'error'; onDone: () => void }) {
  useEffect(() => { const t = setTimeout(onDone, 3500); return () => clearTimeout(t) }, [onDone])
  return (
    <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 rounded-xl text-sm font-medium shadow-2xl" style={{ background: type === 'error' ? '#991b1b' : type === 'success' ? '#065f46' : '#6d28d9', color: '#fff', border: `1px solid ${type === 'error' ? 'rgba(239,68,68,0.4)' : type === 'success' ? 'rgba(52,211,153,0.4)' : 'rgba(139,92,246,0.5)'}`, maxWidth: 320 }}>
      {type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'} {message}
    </div>
  )
}

// ─── Force Graph ──────────────────────────────────────────────────────────────

function ForceGraph({ nodes: rawNodes, edges, selectedNode, onSelectNode }: {
  nodes: GraphNode[]
  edges: GraphEdge[]
  selectedNode: GraphNode | null
  onSelectNode: (n: GraphNode | null) => void
}) {
  const svgRef   = useRef<SVGSVGElement>(null)
  const nodesRef = useRef<GraphNode[]>([])
  const rafRef   = useRef<number>(0)
  const [tick, setTick] = useState(0)

  // Init positions
  useEffect(() => {
    const W = 800, H = 500
    nodesRef.current = rawNodes.map((n, i) => ({
      ...n,
      x: W / 2 + Math.cos(i * 2 * Math.PI / rawNodes.length) * 180,
      y: H / 2 + Math.sin(i * 2 * Math.PI / rawNodes.length) * 180,
      vx: 0, vy: 0,
    }))
  }, [rawNodes])

  // Force simulation
  useEffect(() => {
    const W = 800, H = 500
    let iter = 0
    const edgeMap = new Map<string, GraphNode>()

    function step() {
      if (iter++ > 200) return
      const ns = nodesRef.current
      const nodeById = new Map(ns.map(n => [n.id, n]))

      // Repulsion
      for (let i = 0; i < ns.length; i++) {
        for (let j = i + 1; j < ns.length; j++) {
          const a = ns[i], b = ns[j]
          const dx = (b.x ?? 0) - (a.x ?? 0), dy = (b.y ?? 0) - (a.y ?? 0)
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
          const force = 3000 / (dist * dist)
          a.vx = (a.vx ?? 0) - dx / dist * force; a.vy = (a.vy ?? 0) - dy / dist * force
          b.vx = (b.vx ?? 0) + dx / dist * force; b.vy = (b.vy ?? 0) + dy / dist * force
        }
      }
      // Spring attraction
      for (const e of edges) {
        const a = nodeById.get(e.src_id), b = nodeById.get(e.dst_id)
        if (!a || !b) continue
        const dx = (b.x ?? 0) - (a.x ?? 0), dy = (b.y ?? 0) - (a.y ?? 0)
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
        const target = 120, force = (dist - target) * 0.05
        a.vx = (a.vx ?? 0) + dx / dist * force; a.vy = (a.vy ?? 0) + dy / dist * force
        b.vx = (b.vx ?? 0) - dx / dist * force; b.vy = (b.vy ?? 0) - dy / dist * force
      }
      // Center gravity
      for (const n of ns) {
        n.vx = ((n.vx ?? 0) + (W / 2 - (n.x ?? 0)) * 0.005) * 0.85
        n.vy = ((n.vy ?? 0) + (H / 2 - (n.y ?? 0)) * 0.005) * 0.85
        n.x = Math.max(30, Math.min(W - 30, (n.x ?? 0) + (n.vx ?? 0)))
        n.y = Math.max(30, Math.min(H - 30, (n.y ?? 0) + (n.vy ?? 0)))
      }
      edgeMap.clear()
      setTick(t => t + 1)
      if (iter < 200) rafRef.current = requestAnimationFrame(step)
    }
    rafRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(rafRef.current)
  }, [rawNodes, edges])

  const nodeById = useMemo(() => new Map(nodesRef.current.map(n => [n.id, n])), [tick]) // eslint-disable-line react-hooks/exhaustive-deps

  // Drag
  const draggingRef = useRef<{ id: string; ox: number; oy: number } | null>(null)

  function onMouseDown(e: React.MouseEvent, n: GraphNode) {
    e.stopPropagation()
    const svg = svgRef.current!
    const rect = svg.getBoundingClientRect()
    const scaleX = 800 / rect.width, scaleY = 500 / rect.height
    draggingRef.current = { id: n.id, ox: e.clientX * scaleX - (n.x ?? 0), oy: e.clientY * scaleY - (n.y ?? 0) }
  }

  useEffect(() => {
    function onMove(e: MouseEvent) {
      const d = draggingRef.current; if (!d) return
      const svg = svgRef.current; if (!svg) return
      const rect = svg.getBoundingClientRect()
      const scaleX = 800 / rect.width, scaleY = 500 / rect.height
      const n = nodesRef.current.find(x => x.id === d.id)
      if (n) { n.x = e.clientX * scaleX - d.ox; n.y = e.clientY * scaleY - d.oy; n.vx = 0; n.vy = 0; setTick(t => t + 1) }
    }
    function onUp() { draggingRef.current = null }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  const ns = nodesRef.current
  const radius = (n: GraphNode) => Math.min(4 + Math.sqrt(n.degree ?? 1) * 3, 20)

  return (
    <svg ref={svgRef} viewBox="0 0 800 500" className="w-full h-full" style={{ cursor: 'default' }}
      onClick={() => onSelectNode(null)}>
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="rgba(100,116,139,0.5)" />
        </marker>
      </defs>
      {/* Edges */}
      {edges.map(e => {
        const a = nodeById.get(e.src_id), b = nodeById.get(e.dst_id)
        if (!a || !b) return null
        const mx = ((a.x ?? 0) + (b.x ?? 0)) / 2
        const my = ((a.y ?? 0) + (b.y ?? 0)) / 2
        const isHighlighted = selectedNode && (selectedNode.id === e.src_id || selectedNode.id === e.dst_id)
        return (
          <g key={e.id}>
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              stroke={isHighlighted ? 'rgba(167,139,250,0.6)' : 'rgba(51,65,85,0.7)'}
              strokeWidth={isHighlighted ? 1.5 : 1}
              markerEnd="url(#arrow)" />
            {isHighlighted && (
              <text x={mx} y={my} textAnchor="middle" fill="#64748b" fontSize={9} dy={-3}>
                {e.relation}
              </text>
            )}
          </g>
        )
      })}
      {/* Nodes */}
      {ns.map(n => {
        const color = NODE_TYPE_COLOR[n.type?.toUpperCase()] ?? NODE_TYPE_COLOR.DEFAULT
        const r = radius(n)
        const isSelected = selectedNode?.id === n.id
        return (
          <g key={n.id} style={{ cursor: 'grab' }}
            onClick={ev => { ev.stopPropagation(); onSelectNode(n) }}
            onMouseDown={ev => onMouseDown(ev, n)}>
            {isSelected && <circle cx={n.x} cy={n.y} r={r + 5} fill="none" stroke={color} strokeWidth={1.5} opacity={0.4} />}
            <circle cx={n.x} cy={n.y} r={r}
              fill={isSelected ? color : color + '55'}
              stroke={color}
              strokeWidth={isSelected ? 2 : 1} />
            <text x={n.x} y={(n.y ?? 0) + r + 10} textAnchor="middle" fill="#94a3b8" fontSize={10}>
              {n.name.length > 8 ? n.name.slice(0, 8) + '…' : n.name}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function KnowledgePage() {
  // State
  const [documents, setDocuments] = useState<KBDoc[]>([])
  const [messages,  setMessages]  = useState<KbMessage[]>([])
  const [stats,     setStats]     = useState<KBStats | null>(null)
  const [uploading, setUploading] = useState(false)
  const [asking,    setAsking]    = useState(false)
  const [dragOver,  setDragOver]  = useState(false)
  const [activeTab, setActiveTab] = useState<'qa' | 'citations' | 'graph'>('qa')
  const [inputText, setInputText] = useState('')
  const [toast, setToast]         = useState<{ msg: string; type: 'info' | 'success' | 'error' } | null>(null)
  const [loadingDemo, setLoadingDemo] = useState(false)
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set())
  const [collapsedLogs,     setCollapsedLogs]     = useState<Set<string>>(new Set())
  const [reindexing,        setReindexing]         = useState<Set<string>>(new Set())
  const [graphBuilding,     setGraphBuilding]      = useState<Set<string>>(new Set())
  const [uploadTasks,       setUploadTasks]        = useState<UploadTask[]>([])
  // Doc scope selector
  const [scopeMode,    setScopeMode]   = useState<'all' | 'select'>('all')
  const [scopeDocIds,  setScopeDocIds] = useState<Set<string>>(new Set())
  // Graph tab
  const [graphNodes,    setGraphNodes]    = useState<GraphNode[]>([])
  const [graphEdges,    setGraphEdges]    = useState<GraphEdge[]>([])
  const [kgStats,       setKgStats]       = useState<KGStats | null>(null)
  const [graphLoading,  setGraphLoading]  = useState(false)
  const [selectedNode,  setSelectedNode]  = useState<GraphNode | null>(null)
  const [nodeFilter,    setNodeFilter]    = useState('')

  const fileInputRef  = useRef<HTMLInputElement>(null)
  const chatBottomRef = useRef<HTMLDivElement>(null)
  const pollingRef    = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Data loading ───────────────────────────────────────────────────────────

  const loadDocuments = useCallback(async () => {
    try {
      const res = await fetch(`/api/agent/kb/documents?kb_id=${KB_ID}`)
      if (!res.ok) return
      const data = await res.json()
      setDocuments((data.documents ?? data ?? []).sort((a: KBDoc, b: KBDoc) => b.created_at - a.created_at))
    } catch { /* silent */ }
  }, [])

  const loadStats = useCallback(async () => {
    try {
      const res = await fetch(`/api/agent/kb/stats?kb_id=${KB_ID}`)
      if (!res.ok) return
      setStats(await res.json())
    } catch { /* silent */ }
  }, [])

  const loadGraph = useCallback(async () => {
    setGraphLoading(true)
    try {
      const [gRes, sRes] = await Promise.all([
        fetch(`/api/agent/kg/graph?kb_id=${KB_ID}&limit=300`),
        fetch(`/api/agent/kg/stats?kb_id=${KB_ID}`),
      ])
      if (gRes.ok) {
        const gData = await gRes.json()
        setGraphNodes(gData.nodes ?? [])
        setGraphEdges(gData.edges ?? [])
      }
      if (sRes.ok) setKgStats(await sRes.json())
    } catch { /* silent */ }
    setGraphLoading(false)
  }, [])

  useEffect(() => { loadDocuments(); loadStats() }, [loadDocuments, loadStats])

  useEffect(() => {
    if (activeTab === 'graph') loadGraph()
  }, [activeTab, loadGraph])

  useEffect(() => {
    const hasIndexing = documents.some(d => d.status === 'indexing' || d.status === 'pending')
    if (hasIndexing) {
      if (!pollingRef.current) pollingRef.current = setInterval(() => { loadDocuments(); loadStats() }, 3000)
    } else {
      if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null }
    }
    return () => { if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null } }
  }, [documents, loadDocuments, loadStats])

  useEffect(() => { chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  // ── Upload ─────────────────────────────────────────────────────────────────

  async function uploadFileStream(file: File) {
    const taskId = `${file.name}-${Date.now()}`
    setUploadTasks(prev => [...prev, { id: taskId, filename: file.name, filesize: file.size, progress: 0, currentStep: 'parsing', embeddingProgress: null, done: false, error: null }])
    const form = new FormData(); form.append('file', file); form.append('kb_id', KB_ID)
    try {
      const res = await fetch('/api/agent/kb/documents/upload/stream', { method: 'POST', body: form })
      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({}))
        setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, done: true, error: err.detail ?? res.statusText } : t)); return
      }
      const reader = res.body.getReader(); const decoder = new TextDecoder(); let buffer = ''
      while (true) {
        const { done, value } = await reader.read(); if (done) break
        buffer += decoder.decode(value, { stream: true }); const lines = buffer.split('\n'); buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const ev = JSON.parse(line.slice(6).trim()) as UploadProgressEvent
            if (ev.type === 'heartbeat') continue
            if (ev.type === 'progress') {
              setUploadTasks(prev => prev.map(t => { if (t.id !== taskId) return t; const up: Partial<UploadTask> = { progress: ev.progress ?? t.progress, currentStep: (ev.step as UploadTask['currentStep']) ?? t.currentStep }; if (ev.step === 'embedding' && ev.chunk_idx != null && ev.total != null) up.embeddingProgress = { current: ev.chunk_idx, total: ev.total }; return { ...t, ...up } }))
            } else if (ev.type === 'done') {
              setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, done: true, progress: 100, currentStep: 'done', embeddingProgress: null } : t))
              await loadDocuments(); await loadStats()
            } else if (ev.type === 'error') {
              setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, done: true, error: ev.message ?? '未知错误' } : t))
            }
          } catch { /* bad json */ }
        }
      }
    } catch (e: unknown) { setUploadTasks(prev => prev.map(t => t.id === taskId ? { ...t, done: true, error: e instanceof Error ? e.message : String(e) } : t)) }
  }

  async function uploadFiles(files: FileList | File[]) {
    const arr = Array.from(files); if (!arr.length) return
    setUploading(true); await Promise.all(arr.map(f => uploadFileStream(f))); setUploading(false)
    setTimeout(() => setUploadTasks(prev => prev.filter(t => !t.done || t.error !== null)), 8000)
  }

  // ── Document actions ───────────────────────────────────────────────────────

  async function deleteDoc(docId: string) {
    const res = await fetch(`/api/agent/kb/documents/${docId}?kb_id=${KB_ID}`, { method: 'DELETE' }).catch(() => null)
    if (res?.ok) { setDocuments(prev => prev.filter(d => d.doc_id !== docId)); setScopeDocIds(prev => { const s = new Set(prev); s.delete(docId); return s }); await loadStats(); showToast('文档已删除', 'success') }
    else showToast('删除失败', 'error')
  }

  async function reindexDoc(docId: string) {
    setReindexing(prev => new Set(prev).add(docId))
    try {
      const res = await fetch(`/api/agent/kb/documents/${docId}/reindex`, { method: 'POST' })
      if (res.ok) { showToast('重新索引成功', 'success'); await loadDocuments(); await loadStats() }
      else { const err = await res.json().catch(() => ({})); showToast(`重新索引失败：${err.detail ?? res.statusText}`, 'error') }
    } catch (e: unknown) { showToast(`出错：${e instanceof Error ? e.message : String(e)}`, 'error') }
    finally { setReindexing(prev => { const s = new Set(prev); s.delete(docId); return s }) }
  }

  async function buildDocGraph(docId: string) {
    setGraphBuilding(prev => new Set(prev).add(docId))
    showToast('图谱生成中…', 'info')
    try {
      const res = await fetch(`/api/agent/kb/build-graph/${docId}?kb_id=${KB_ID}`, { method: 'POST' })
      if (res.ok) { showToast('图谱任务已启动', 'success'); if (activeTab === 'graph') setTimeout(loadGraph, 2000) }
      else { const err = await res.json().catch(() => ({})); showToast(`图谱生成失败：${err.detail ?? '未知错误'}`, 'error') }
    } catch { showToast('图谱生成出错', 'error') }
    finally { setGraphBuilding(prev => { const s = new Set(prev); s.delete(docId); return s }) }
  }

  // ── Scope selector ─────────────────────────────────────────────────────────

  function toggleScopeDoc(docId: string) {
    setScopeDocIds(prev => { const s = new Set(prev); s.has(docId) ? s.delete(docId) : s.add(docId); return s })
  }

  const activeScopeDocIds = scopeMode === 'all' ? [] : Array.from(scopeDocIds)
  const scopeLabel = scopeMode === 'all' ? '全部文档' : scopeDocIds.size === 0 ? '请选择文档' : scopeDocIds.size === 1 ? documents.find(d => scopeDocIds.has(d.doc_id))?.filename ?? '1 份文档' : `${scopeDocIds.size} 份文档`

  // ── Ask ────────────────────────────────────────────────────────────────────

  async function askQuestion(query: string) {
    if (!query.trim() || asking) return
    if (scopeMode === 'select' && scopeDocIds.size === 0) { showToast('请先选择要查询的文档', 'error'); return }

    const history = messages.filter(m => !m.streaming).map(m => ({ role: m.role, content: m.content }))
    const assistantId = `a-${Date.now()}`
    const initSteps: ProcessStep[] = [
      { id: 'retrieve', label: '检索知识库', state: 'running', detail: scopeMode === 'select' ? `范围：${scopeLabel}` : undefined },
      { id: 'generate', label: '生成回答', state: 'waiting' },
    ]

    setMessages(prev => [
      ...prev,
      { id: `u-${Date.now()}`, role: 'user', content: query.trim(), timestamp: new Date() },
      { id: assistantId, role: 'assistant', content: '', citations: [], timestamp: new Date(), streaming: true, processSteps: initSteps, scopeLabel },
    ])
    setInputText(''); setAsking(true)

    try {
      const res = await fetch('/api/agent/kb/ask/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), kb_id: KB_ID, top_k: 5, with_graph: false, history, doc_ids: activeScopeDocIds }),
      })
      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({}))
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: `请求失败：${err.detail ?? res.statusText}`, streaming: false, processSteps: [] } : m)); return
      }
      const reader = res.body.getReader(); const decoder = new TextDecoder(); let buffer = ''
      while (true) {
        const { done, value } = await reader.read(); if (done) break
        buffer += decoder.decode(value, { stream: true }); const lines = buffer.split('\n'); buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim(); if (raw === '[DONE]') break
          try {
            const ev = JSON.parse(raw)
            if (ev.type === 'context') {
              const count = (ev.citations ?? []).length
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : { ...m, citations: ev.citations ?? [], processSteps: (m.processSteps ?? []).map(s => s.id === 'retrieve' ? { ...s, state: 'done' as const, detail: `召回 ${count} 个片段` } : s.id === 'generate' ? { ...s, state: 'running' as const } : s) }))
            } else if (ev.type === 'thinking') {
              const elapsed = ev.elapsed ?? 0
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : { ...m, thinkingElapsed: elapsed, processSteps: (m.processSteps ?? []).map(s => s.id === 'generate' && s.state === 'running' ? { ...s, detail: elapsed > 0 ? `${elapsed}s…` : undefined } : s) }))
            } else if (ev.type === 'delta' && ev.text) {
              setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: m.content + ev.text, thinkingElapsed: undefined } : m))
            } else if (ev.type === 'done') {
              setMessages(prev => prev.map(m => m.id !== assistantId ? m : { ...m, streaming: false, thinkingElapsed: undefined, processSteps: (m.processSteps ?? []).map(s => s.state !== 'done' ? { ...s, state: 'done' as const, detail: undefined } : s) }))
            }
          } catch { /* bad json */ }
        }
      }
      setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, streaming: false } : m))
    } catch (e: unknown) {
      setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: `网络错误：${e instanceof Error ? e.message : String(e)}`, streaming: false, processSteps: [] } : m))
    } finally { setAsking(false) }
  }

  // ── Demo docs ──────────────────────────────────────────────────────────────

  async function loadDemoDocuments() {
    setLoadingDemo(true); let ok = 0
    for (const doc of DEMO_DOCS) {
      try {
        const res = await fetch('/api/agent/kb/documents/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: doc.content, filename: doc.title, kb_id: KB_ID }) })
        if (res.ok) ok++
      } catch { /* skip */ }
    }
    showToast(ok > 0 ? `已加载 ${ok} 个示例文档` : '加载失败，请检查 API 连接', ok > 0 ? 'success' : 'error')
    await loadDocuments(); await loadStats(); setLoadingDemo(false)
  }

  // ── Misc ───────────────────────────────────────────────────────────────────

  function showToast(msg: string, type: 'info' | 'success' | 'error' = 'info') { setToast({ msg, type }) }
  function toggleCitation(k: string) { setExpandedCitations(prev => { const s = new Set(prev); s.has(k) ? s.delete(k) : s.add(k); return s }) }
  function toggleLog(id: string) { setCollapsedLogs(prev => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s }) }

  const hasDocuments = documents.length > 0
  const allCitations = messages.filter(m => m.role === 'assistant' && m.citations?.length).flatMap(m => m.citations!)
  const filteredNodes = graphNodes.filter(n => !nodeFilter || n.name.toLowerCase().includes(nodeFilter.toLowerCase()) || n.type.toLowerCase().includes(nodeFilter.toLowerCase()))

  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen" style={{ background: '#020617', color: '#e2e8f0' }}>

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-3 border-b flex-shrink-0" style={{ background: '#0f172a', borderColor: '#1e293b' }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(124,58,237,0.2)', border: '1px solid rgba(124,58,237,0.4)' }}>📚</div>
          <div>
            <h1 className="font-semibold text-sm" style={{ color: '#f1f5f9' }}>知识库问答</h1>
            <p className="text-xs" style={{ color: '#475569' }}>Knowledge Base · RAG</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {stats && (
            <div className="flex items-center gap-1.5">
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium" style={{ background: 'rgba(124,58,237,0.12)', color: '#a78bfa', border: '1px solid rgba(124,58,237,0.2)' }}>{stats.doc_count} 文档</span>
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium" style={{ background: 'rgba(6,182,212,0.08)', color: '#67e8f9', border: '1px solid rgba(6,182,212,0.2)' }}>{stats.chunk_count} 片段</span>
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium" style={{ background: 'rgba(52,211,153,0.08)', color: '#6ee7b7', border: '1px solid rgba(52,211,153,0.2)' }}>{formatBytes(stats.total_chars)}</span>
            </div>
          )}
          <div className="w-px h-4 mx-1" style={{ background: '#1e293b' }} />
          <button onClick={() => fileInputRef.current?.click()} disabled={uploading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
            style={{ background: uploading ? 'rgba(124,58,237,0.25)' : '#7c3aed', color: '#fff', border: '1px solid rgba(139,92,246,0.5)', cursor: uploading ? 'not-allowed' : 'pointer' }}>
            {uploading ? <><svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg>上传中</> : <><svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" /></svg>上传文档</>}
          </button>
          <input ref={fileInputRef} type="file" className="hidden" accept=".pdf,.txt,.md,.docx,.html" multiple onChange={e => { if (e.target.files?.length) uploadFiles(e.target.files); e.target.value = '' }} />
        </div>
      </header>

      <div className="flex flex-1 min-h-0">

        {/* ── Left: Documents ── */}
        <aside className="w-72 flex-shrink-0 flex flex-col border-r" style={{ background: '#0a0f1e', borderColor: '#1e293b' }}>
          {/* Drop zone */}
          <div onDragOver={e => { e.preventDefault(); setDragOver(true) }} onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files) }}
            onClick={() => fileInputRef.current?.click()}
            className="mx-3 mt-3 rounded-xl flex flex-col items-center justify-center gap-1.5 cursor-pointer transition-all duration-200 py-3"
            style={{ border: `2px dashed ${dragOver ? '#7c3aed' : '#1e293b'}`, background: dragOver ? 'rgba(124,58,237,0.08)' : 'transparent', minHeight: 72 }}>
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: dragOver ? '#a78bfa' : '#334155' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.338-2.32 5.75 5.75 0 011.048 11.09H6.75z" />
            </svg>
            <p className="text-xs text-center" style={{ color: dragOver ? '#a78bfa' : '#475569' }}>拖拽或点击上传<br /><span style={{ color: '#334155' }}>PDF · TXT · MD · DOCX · HTML</span></p>
          </div>

          {/* Upload tasks */}
          {uploadTasks.length > 0 && (
            <div className="mx-3 mt-2 space-y-1.5">
              <div className="flex items-center justify-between px-1">
                <span className="text-xs uppercase font-medium tracking-wider" style={{ color: '#475569' }}>上传进度</span>
                <button onClick={() => setUploadTasks(prev => prev.filter(t => !t.done))} className="text-xs" style={{ color: '#334155' }}>清除</button>
              </div>
              {uploadTasks.map(t => (
                <div key={t.id} className="relative">
                  <UploadStepPipeline task={t} />
                  {t.done && <button onClick={() => setUploadTasks(prev => prev.filter(x => x.id !== t.id))} className="absolute top-2 right-2 text-xs" style={{ color: '#334155' }}>✕</button>}
                </div>
              ))}
            </div>
          )}

          {/* Document list header */}
          <div className="flex items-center justify-between px-4 pt-3 pb-1.5">
            <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: '#475569' }}>文档列表</span>
            <span className="text-xs px-1.5 py-0.5 rounded font-mono" style={{ background: '#1e293b', color: '#64748b' }}>{documents.length}</span>
          </div>

          {/* Documents */}
          <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1.5">
            {documents.length === 0 ? (
              <div className="py-8 text-center"><p className="text-xs" style={{ color: '#334155' }}>暂无文档</p></div>
            ) : (
              documents.map(doc => {
                const color    = fileTypeColor(doc.filename)
                const inScope  = scopeMode === 'select' && scopeDocIds.has(doc.doc_id)
                return (
                  <div key={doc.doc_id} className="rounded-xl cursor-pointer transition-all duration-150 group overflow-hidden"
                    onClick={() => scopeMode === 'select' && toggleScopeDoc(doc.doc_id)}
                    style={{ background: inScope ? 'rgba(124,58,237,0.12)' : '#111827', border: `1px solid ${inScope ? 'rgba(124,58,237,0.4)' : '#1e293b'}`, borderLeft: `3px solid ${doc.status === 'ready' ? color : doc.status === 'indexing' ? '#3b82f6' : doc.status === 'error' ? '#ef4444' : '#334155'}` }}>
                    <div className="flex items-start gap-2.5 px-3 pt-2.5 pb-2">
                      {scopeMode === 'select' && (
                        <div className="w-4 h-4 rounded flex items-center justify-center flex-shrink-0 mt-0.5" style={{ background: inScope ? '#7c3aed' : 'transparent', border: `1px solid ${inScope ? '#7c3aed' : '#334155'}` }}>
                          {inScope && <svg className="w-2.5 h-2.5" viewBox="0 0 20 20" fill="white"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>}
                        </div>
                      )}
                      <span className="text-lg flex-shrink-0 leading-none mt-0.5">{fileIcon(doc.filename)}</span>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium truncate mb-1.5" style={{ color: '#e2e8f0' }} title={doc.filename}>{doc.filename}</p>
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <StatusDot status={doc.status} />
                          {doc.chunk_count > 0 && <span className="text-xs px-1.5 py-0.5 rounded font-medium" style={{ background: 'rgba(6,182,212,0.08)', color: '#67e8f9', border: '1px solid rgba(6,182,212,0.15)' }}>{doc.chunk_count} 块</span>}
                          {doc.char_count > 0 && <span className="text-xs" style={{ color: '#475569' }}>{formatBytes(doc.char_count)}</span>}
                        </div>
                        {doc.error_msg && <p className="text-xs mt-1 truncate" style={{ color: '#f87171' }} title={doc.error_msg}>{doc.error_msg}</p>}
                      </div>
                    </div>
                    <div className="flex items-center justify-between px-3 pb-2" style={{ borderTop: '1px solid rgba(255,255,255,0.03)' }}>
                      <span className="text-xs" style={{ color: '#334155' }}>{formatDate(doc.created_at)}</span>
                      <span className="text-xs uppercase font-mono" style={{ color: color, opacity: 0.7 }}>{doc.filename.split('.').pop()?.toUpperCase()}</span>
                    </div>
                    {/* Hover actions */}
                    <div className="flex items-center gap-1 px-2 pb-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      {doc.status === 'ready' && (
                        <button onClick={e => { e.stopPropagation(); buildDocGraph(doc.doc_id) }} disabled={graphBuilding.has(doc.doc_id)}
                          className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium"
                          style={{ background: 'rgba(244,114,182,0.1)', color: '#f472b6', border: '1px solid rgba(244,114,182,0.2)' }}>
                          {graphBuilding.has(doc.doc_id) ? <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg> : '⟡'} 生成图谱
                        </button>
                      )}
                      {doc.status === 'error' && (
                        <button onClick={e => { e.stopPropagation(); reindexDoc(doc.doc_id) }} disabled={reindexing.has(doc.doc_id)}
                          className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium"
                          style={{ background: 'rgba(251,146,60,0.1)', color: '#fb923c', border: '1px solid rgba(251,146,60,0.2)' }}>
                          ↺ 重新索引
                        </button>
                      )}
                      <button onClick={e => { e.stopPropagation(); deleteDoc(doc.doc_id) }}
                        className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium ml-auto"
                        style={{ background: 'rgba(239,68,68,0.08)', color: '#f87171', border: '1px solid rgba(239,68,68,0.2)' }}>
                        <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
                        删除
                      </button>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="flex-1 flex flex-col min-w-0" style={{ background: '#020617' }}>
          {!hasDocuments ? (
            /* ── Empty State ── */
            <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8">
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl" style={{ background: 'rgba(124,58,237,0.12)', border: '1px solid rgba(124,58,237,0.25)' }}>📚</div>
              <div className="text-center max-w-xs">
                <h2 className="text-lg font-semibold mb-2" style={{ color: '#f1f5f9' }}>知识库尚无文档</h2>
                <p className="text-sm leading-relaxed" style={{ color: '#475569' }}>上传文档后即可通过自然语言提问，系统将基于文档内容精准回答。</p>
              </div>
              <div className="flex gap-3 items-center">
                <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium" style={{ background: '#7c3aed', color: '#fff', border: '1px solid rgba(139,92,246,0.4)' }}>
                  <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" /></svg>
                  上传文档
                </button>
                <button onClick={loadDemoDocuments} disabled={loadingDemo} className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium" style={{ background: 'rgba(6,182,212,0.1)', color: '#67e8f9', border: '1px solid rgba(6,182,212,0.25)', cursor: loadingDemo ? 'not-allowed' : 'pointer' }}>
                  {loadingDemo ? <><svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg>加载中…</> : <>📖 加载示例文档</>}
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* ── Tab bar + scope selector ── */}
              <div className="flex items-center justify-between px-4 pt-2 pb-0 border-b flex-shrink-0" style={{ borderColor: '#1e293b' }}>
                <div className="flex items-center gap-0.5">
                  {([['qa', '问答'], ['citations', '引用'], ['graph', '知识图谱']] as const).map(([tab, label]) => (
                    <button key={tab} onClick={() => setActiveTab(tab)}
                      className="px-4 py-2 text-xs font-medium rounded-t-lg transition-all border-b-2"
                      style={{ color: activeTab === tab ? '#a78bfa' : '#475569', borderBottomColor: activeTab === tab ? '#7c3aed' : 'transparent', background: activeTab === tab ? 'rgba(124,58,237,0.06)' : 'transparent' }}>
                      {label}
                      {tab === 'citations' && allCitations.length > 0 && <span className="ml-1.5 px-1.5 py-0.5 rounded-full text-xs" style={{ background: 'rgba(124,58,237,0.2)', color: '#a78bfa' }}>{allCitations.length}</span>}
                      {tab === 'graph' && graphNodes.length > 0 && <span className="ml-1.5 px-1.5 py-0.5 rounded-full text-xs" style={{ background: 'rgba(244,114,182,0.15)', color: '#f472b6' }}>{graphNodes.length}</span>}
                    </button>
                  ))}
                </div>
                {/* Scope selector pills */}
                {activeTab === 'qa' && (
                  <div className="flex items-center gap-1.5 pb-1">
                    <span className="text-xs" style={{ color: '#475569' }}>范围：</span>
                    <button onClick={() => { setScopeMode('all'); setScopeDocIds(new Set()) }}
                      className="px-2.5 py-1 rounded-lg text-xs font-medium transition-all"
                      style={{ background: scopeMode === 'all' ? 'rgba(124,58,237,0.2)' : 'transparent', color: scopeMode === 'all' ? '#a78bfa' : '#475569', border: `1px solid ${scopeMode === 'all' ? 'rgba(124,58,237,0.4)' : '#1e293b'}` }}>
                      全部文档
                    </button>
                    <button onClick={() => setScopeMode(scopeMode === 'select' ? 'all' : 'select')}
                      className="px-2.5 py-1 rounded-lg text-xs font-medium transition-all"
                      style={{ background: scopeMode === 'select' ? 'rgba(124,58,237,0.2)' : 'transparent', color: scopeMode === 'select' ? '#a78bfa' : '#475569', border: `1px solid ${scopeMode === 'select' ? 'rgba(124,58,237,0.4)' : '#1e293b'}` }}>
                      {scopeMode === 'select' && scopeDocIds.size > 0 ? scopeLabel : '选择文档'}
                    </button>
                  </div>
                )}
              </div>

              {/* ── Scope hint when in select mode ── */}
              {activeTab === 'qa' && scopeMode === 'select' && (
                <div className="px-4 py-2 flex-shrink-0 border-b" style={{ borderColor: '#0f172a', background: 'rgba(124,58,237,0.04)' }}>
                  <p className="text-xs" style={{ color: '#64748b' }}>
                    {scopeDocIds.size === 0
                      ? '← 点击左侧文档勾选，可对单个或多个文档提问'
                      : `已选：${Array.from(scopeDocIds).map(id => documents.find(d => d.doc_id === id)?.filename ?? id).join('、')}`}
                  </p>
                </div>
              )}

              {activeTab === 'qa' && (
                <>
                  <div className="flex-1 overflow-y-auto px-4 py-5 space-y-5">
                    {messages.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full gap-5 pb-8">
                        <div className="text-center">
                          <div className="w-10 h-10 rounded-xl mx-auto mb-3 flex items-center justify-center text-xl" style={{ background: 'rgba(124,58,237,0.12)', border: '1px solid rgba(124,58,237,0.2)' }}>💬</div>
                          <h3 className="font-medium text-sm mb-1" style={{ color: '#e2e8f0' }}>开始提问</h3>
                          <p className="text-xs" style={{ color: '#475569' }}>基于 {documents.length} 份文档回答问题 · {scopeMode === 'select' && scopeDocIds.size > 0 ? `已选 ${scopeDocIds.size} 份` : '全库检索'}</p>
                        </div>
                        <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                          {DEMO_PROMPTS.map(p => <button key={p} onClick={() => askQuestion(p)} className="px-3.5 py-1.5 rounded-xl text-xs transition-all" style={{ background: '#111827', color: '#94a3b8', border: '1px solid #1e293b' }}>{p}</button>)}
                        </div>
                      </div>
                    ) : (
                      messages.map(msg => (
                        <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                          <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs flex-shrink-0 mt-0.5 font-medium"
                            style={{ background: msg.role === 'user' ? 'rgba(124,58,237,0.3)' : 'rgba(6,182,212,0.15)', border: `1px solid ${msg.role === 'user' ? 'rgba(124,58,237,0.5)' : 'rgba(6,182,212,0.25)'}`, color: msg.role === 'user' ? '#c4b5fd' : '#67e8f9' }}>
                            {msg.role === 'user' ? '你' : '✦'}
                          </div>
                          <div className={`max-w-[75%] flex flex-col gap-1.5 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                            {/* Scope tag */}
                            {msg.role === 'user' && msg.scopeLabel && msg.scopeLabel !== '全部文档' && (
                              <span className="text-xs px-2 py-0.5 rounded-full mb-0.5" style={{ background: 'rgba(124,58,237,0.15)', color: '#a78bfa', border: '1px solid rgba(124,58,237,0.2)' }}>
                                📂 {msg.scopeLabel}
                              </span>
                            )}
                            {/* Process log */}
                            {msg.role === 'assistant' && msg.processSteps && msg.processSteps.length > 0 && (
                              <div className="w-full">
                                <button onClick={() => toggleLog(msg.id)} className="flex items-center gap-1.5 mb-1 text-xs" style={{ color: '#334155' }}>
                                  <svg className={`w-3 h-3 transition-transform ${collapsedLogs.has(msg.id) ? '' : 'rotate-90'}`} viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" /></svg>
                                  执行过程
                                </button>
                                <ProcessTimeline steps={msg.processSteps} collapsed={collapsedLogs.has(msg.id)} />
                              </div>
                            )}
                            {/* Bubble */}
                            <div className="px-4 py-3 text-sm leading-relaxed"
                              style={{ background: msg.role === 'user' ? 'rgba(109,40,217,0.85)' : '#111827', color: msg.role === 'user' ? '#ede9fe' : '#e2e8f0', borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px', border: msg.role === 'user' ? '1px solid rgba(139,92,246,0.3)' : '1px solid #1e293b' }}>
                              {msg.role === 'assistant' && msg.streaming && msg.content === '' ? (
                                <div className="flex items-center gap-2">
                                  <div className="flex items-center gap-1">{[0,1,2].map(i => <div key={i} className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: '#334155', animationDelay: `${i*150}ms` }} />)}</div>
                                  {msg.thinkingElapsed != null && msg.thinkingElapsed > 0 && <span className="text-xs" style={{ color: '#334155' }}>{msg.thinkingElapsed}s</span>}
                                </div>
                              ) : (
                                <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}{msg.streaming && msg.content && <span className="animate-pulse ml-0.5" style={{ color: '#7c3aed' }}>▍</span>}</span>
                              )}
                            </div>
                            {/* Citations */}
                            {msg.citations && msg.citations.length > 0 && (
                              <div className="flex flex-wrap gap-1.5">
                                {msg.citations.map((cit, i) => {
                                  const k = `${msg.id}-${i}`, exp = expandedCitations.has(k)
                                  return (
                                    <div key={k} className="flex flex-col gap-1">
                                      <button onClick={() => toggleCitation(k)} className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs" style={{ background: 'rgba(124,58,237,0.1)', color: '#a78bfa', border: '1px solid rgba(124,58,237,0.2)' }}>
                                        <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor"><path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" /></svg>
                                        来源{cit.index ?? i + 1}{cit.filename && <span style={{ opacity: 0.6 }}>·{cit.filename}</span>}
                                        {cit.score != null && cit.score > 0 && <span className="px-1 rounded" style={{ background: 'rgba(52,211,153,0.12)', color: '#6ee7b7' }}>{(cit.score*100).toFixed(0)}%</span>}
                                        <svg className={`w-3 h-3 transition-transform ${exp ? 'rotate-180' : ''}`} viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
                                      </button>
                                      {exp && <div className="px-3 py-2 rounded-lg text-xs leading-relaxed max-w-sm" style={{ background: '#0a0f1e', color: '#64748b', border: '1px solid #1e293b' }}>{cit.text_preview}</div>}
                                    </div>
                                  )
                                })}
                              </div>
                            )}
                            <span className="text-xs px-1" style={{ color: '#1e293b' }}>{msg.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>
                          </div>
                        </div>
                      ))
                    )}
                    <div ref={chatBottomRef} />
                  </div>

                  {/* Input */}
                  <div className="px-4 pb-4 pt-2 flex-shrink-0 border-t" style={{ borderColor: '#0f172a' }}>
                    <div className="flex items-end gap-2 rounded-2xl px-4 py-3" style={{ background: '#0a0f1e', border: `1px solid ${scopeMode === 'select' && scopeDocIds.size > 0 ? 'rgba(124,58,237,0.35)' : '#1e293b'}` }}>
                      {scopeMode === 'select' && scopeDocIds.size > 0 && (
                        <span className="flex-shrink-0 text-xs px-2 py-1 rounded-lg self-center" style={{ background: 'rgba(124,58,237,0.15)', color: '#a78bfa', border: '1px solid rgba(124,58,237,0.25)', whiteSpace: 'nowrap' }}>
                          📂 {scopeLabel}
                        </span>
                      )}
                      <textarea className="flex-1 bg-transparent text-sm resize-none outline-none leading-relaxed" style={{ color: '#e2e8f0', minHeight: '24px', maxHeight: '120px' }}
                        placeholder={scopeMode === 'select' && scopeDocIds.size === 0 ? '请先在左侧勾选文档…' : '输入问题，按 Enter 发送…'}
                        rows={1} value={inputText}
                        onChange={e => { setInputText(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px' }}
                        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(inputText) } }} />
                      <button onClick={() => askQuestion(inputText)} disabled={asking || !inputText.trim() || (scopeMode === 'select' && scopeDocIds.size === 0)}
                        className="flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all"
                        style={{ background: asking || !inputText.trim() ? 'rgba(124,58,237,0.15)' : '#7c3aed', color: '#fff', cursor: asking || !inputText.trim() ? 'not-allowed' : 'pointer' }}>
                        {asking ? <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg> : <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" /></svg>}
                      </button>
                    </div>
                    <p className="text-xs mt-1.5 text-center" style={{ color: '#1e293b' }}>Enter 发送 · Shift+Enter 换行</p>
                  </div>
                </>
              )}

              {activeTab === 'citations' && (
                <div className="flex-1 overflow-y-auto px-4 py-4">
                  {allCitations.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: '#334155' }}>
                      <svg className="w-10 h-10 opacity-30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" /></svg>
                      <p className="text-sm">提问后将在此显示引用来源</p>
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: '#475569' }}>全部引用来源 ({allCitations.length})</p>
                      {allCitations.map((cit, i) => (
                        <div key={i} className="rounded-xl p-3.5" style={{ background: '#0a0f1e', border: '1px solid #1e293b' }}>
                          <div className="flex items-center gap-2 mb-2">
                            <span className="w-5 h-5 rounded flex items-center justify-center text-xs font-bold flex-shrink-0" style={{ background: 'rgba(124,58,237,0.2)', color: '#a78bfa' }}>{cit.index ?? i + 1}</span>
                            <span className="text-xs font-medium truncate flex-1" style={{ color: '#e2e8f0' }}>{cit.filename || cit.source}</span>
                            {cit.score != null && cit.score > 0 && <span className="text-xs px-2 py-0.5 rounded-full flex-shrink-0" style={{ background: 'rgba(52,211,153,0.08)', color: '#6ee7b7', border: '1px solid rgba(52,211,153,0.15)' }}>{(cit.score*100).toFixed(0)}% 相关</span>}
                          </div>
                          <p className="text-xs leading-relaxed" style={{ color: '#475569' }}>{cit.text_preview}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'graph' && (
                <div className="flex-1 flex min-h-0">
                  {/* Graph canvas */}
                  <div className="flex-1 flex flex-col min-w-0 relative">
                    {graphLoading ? (
                      <div className="flex-1 flex items-center justify-center" style={{ color: '#475569' }}>
                        <svg className="w-6 h-6 animate-spin mr-2" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg>
                        加载图谱中…
                      </div>
                    ) : graphNodes.length === 0 ? (
                      <div className="flex-1 flex flex-col items-center justify-center gap-4" style={{ color: '#334155' }}>
                        <span className="text-4xl opacity-30">⟡</span>
                        <p className="text-sm">暂无图谱数据</p>
                        <p className="text-xs" style={{ color: '#1e293b' }}>在左侧文档列表点击「生成图谱」按钮来提取知识图谱</p>
                        <button onClick={loadGraph} className="text-xs px-3 py-1.5 rounded-lg" style={{ background: 'rgba(244,114,182,0.1)', color: '#f472b6', border: '1px solid rgba(244,114,182,0.2)' }}>刷新</button>
                      </div>
                    ) : (
                      <>
                        {/* Stats bar */}
                        <div className="flex items-center gap-3 px-4 py-2 flex-shrink-0 border-b" style={{ borderColor: '#0f172a' }}>
                          <span className="text-xs" style={{ color: '#475569' }}>知识图谱</span>
                          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(244,114,182,0.1)', color: '#f472b6', border: '1px solid rgba(244,114,182,0.2)' }}>
                            {graphNodes.length} 实体
                          </span>
                          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(167,139,250,0.1)', color: '#a78bfa', border: '1px solid rgba(167,139,250,0.2)' }}>
                            {graphEdges.length} 关系
                          </span>
                          <button onClick={loadGraph} className="ml-auto text-xs px-2.5 py-1 rounded-lg" style={{ background: 'rgba(255,255,255,0.04)', color: '#475569', border: '1px solid #1e293b' }}>刷新</button>
                        </div>

                        {/* Legend */}
                        <div className="flex items-center gap-3 px-4 py-1.5 flex-shrink-0 border-b flex-wrap" style={{ borderColor: '#0f172a' }}>
                          {Object.entries(NODE_TYPE_COLOR).filter(([k]) => k !== 'DEFAULT').map(([type, color]) => (
                            <div key={type} className="flex items-center gap-1">
                              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                              <span className="text-xs" style={{ color: '#475569' }}>{type}</span>
                            </div>
                          ))}
                          <span className="text-xs ml-auto" style={{ color: '#334155' }}>拖动节点 · 点击查看详情</span>
                        </div>

                        {/* SVG graph */}
                        <div className="flex-1 overflow-hidden" style={{ background: '#020617' }}>
                          <ForceGraph
                            nodes={graphNodes}
                            edges={graphEdges}
                            selectedNode={selectedNode}
                            onSelectNode={setSelectedNode}
                          />
                        </div>
                      </>
                    )}

                    {/* Selected node detail popup */}
                    {selectedNode && (
                      <div className="absolute bottom-4 left-4 right-4 rounded-xl p-4 shadow-2xl" style={{ background: '#0f172a', border: '1px solid #1e293b', maxWidth: 380 }}>
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: (NODE_TYPE_COLOR[selectedNode.type?.toUpperCase()] ?? NODE_TYPE_COLOR.DEFAULT) + '22', border: `1px solid ${NODE_TYPE_COLOR[selectedNode.type?.toUpperCase()] ?? NODE_TYPE_COLOR.DEFAULT}55` }}>
                            <div className="w-3 h-3 rounded-full" style={{ background: NODE_TYPE_COLOR[selectedNode.type?.toUpperCase()] ?? NODE_TYPE_COLOR.DEFAULT }} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <p className="font-semibold text-sm" style={{ color: '#f1f5f9' }}>{selectedNode.name}</p>
                              <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'rgba(255,255,255,0.05)', color: '#64748b' }}>{selectedNode.type}</span>
                              <span className="text-xs" style={{ color: '#334155' }}>度:{selectedNode.degree}</span>
                            </div>
                            {selectedNode.description && <p className="text-xs leading-relaxed" style={{ color: '#64748b' }}>{selectedNode.description}</p>}
                            <div className="mt-2">
                              <p className="text-xs mb-1" style={{ color: '#475569' }}>相关关系：</p>
                              <div className="flex flex-wrap gap-1">
                                {graphEdges.filter(e => e.src_id === selectedNode.id || e.dst_id === selectedNode.id).slice(0, 8).map(e => {
                                  const other = graphNodes.find(n => n.id === (e.src_id === selectedNode.id ? e.dst_id : e.src_id))
                                  return other ? (
                                    <span key={e.id} className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(167,139,250,0.08)', color: '#a78bfa', border: '1px solid rgba(167,139,250,0.15)' }}>
                                      {e.src_id === selectedNode.id ? `→ ${other.name}` : `← ${other.name}`} <span style={{ opacity: 0.6 }}>({e.relation})</span>
                                    </span>
                                  ) : null
                                })}
                              </div>
                            </div>
                          </div>
                          <button onClick={() => setSelectedNode(null)} className="text-sm flex-shrink-0" style={{ color: '#334155' }}>✕</button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Right: Entity list */}
                  <div className="w-60 flex-shrink-0 flex flex-col border-l" style={{ borderColor: '#1e293b', background: '#0a0f1e' }}>
                    <div className="px-3 pt-3 pb-2">
                      <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: '#475569' }}>实体列表</p>
                      <input value={nodeFilter} onChange={e => setNodeFilter(e.target.value)} placeholder="搜索实体…"
                        className="w-full bg-transparent text-xs outline-none px-2.5 py-1.5 rounded-lg"
                        style={{ background: '#111827', border: '1px solid #1e293b', color: '#94a3b8' }} />
                    </div>
                    <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1">
                      {filteredNodes.slice(0, 100).map(n => {
                        const color = NODE_TYPE_COLOR[n.type?.toUpperCase()] ?? NODE_TYPE_COLOR.DEFAULT
                        const isSel = selectedNode?.id === n.id
                        return (
                          <button key={n.id} onClick={() => setSelectedNode(isSel ? null : n)}
                            className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-left transition-all"
                            style={{ background: isSel ? 'rgba(124,58,237,0.12)' : 'transparent', border: `1px solid ${isSel ? 'rgba(124,58,237,0.3)' : 'transparent'}` }}>
                            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                            <span className="text-xs truncate flex-1" style={{ color: '#e2e8f0' }}>{n.name}</span>
                            <span className="text-xs flex-shrink-0" style={{ color: '#334155' }}>{n.degree}</span>
                          </button>
                        )
                      })}
                      {filteredNodes.length > 100 && <p className="text-xs text-center py-2" style={{ color: '#334155' }}>+{filteredNodes.length - 100} 更多</p>}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {toast && <Toast message={toast.msg} type={toast.type} onDone={() => setToast(null)} />}
    </div>
  )
}
