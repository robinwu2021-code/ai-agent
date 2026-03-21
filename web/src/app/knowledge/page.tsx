'use client'

import { useState, useEffect, useRef, useCallback, DragEvent } from 'react'

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

interface KbMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  timestamp: Date
}

interface KBStats {
  doc_count: number
  chunk_count: number
  total_chars: number
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
    content: `# 星辰科技有限公司 企业概览

## 公司简介

星辰科技有限公司（Starlight Technology Co., Ltd）成立于2018年，总部位于上海市浦东新区张江高科技园区。公司专注于人工智能驱动的企业数字化营销与数据智能领域，致力于帮助中国企业实现精准营销、智能运营与数据驱动决策。经过六年深耕，星辰科技已成长为国内AI营销领域的领军企业之一，旗下产品覆盖营销自动化、数据分析与智能客服三大核心场景。

## 核心管理团队

公司创始人兼首席执行官**张明远**，毕业于复旦大学计算机科学专业，拥有15年互联网与AI从业经验，曾先后任职于百度数据智能部门及微软亚洲研究院。首席技术官**李晓红**，毕业于清华大学人工智能研究院，深耕自然语言处理与机器学习方向，拥有20余项AI相关专利，主导构建了星辰科技的核心算法引擎。

## 核心产品线

**智星 AI营销平台**是公司旗舰产品，基于大语言模型与多模态AI技术，提供多渠道营销内容自动生成、用户画像分析、智能投放策略及ROI追踪等功能。智星平台已集成微信公众号、抖音、微博、淘宝等主流流量渠道，目前服务超过3000家企业客户。

**数睿数据分析工具**专为中型以上企业设计，提供实时数据大屏、多维度漏斗分析、用户行为追踪及预测模型构建能力。

**星客智能客服系统**融合对话式AI与知识图谱技术，支持全渠道接入、意图识别、情绪分析及人机协同等功能，平均响应时延低于200毫秒，客户满意度达92%。

## 办公布局

公司在全国设有四大办公中心：**上海总部**、**北京分公司**、**深圳分公司**、**成都研发中心**。截至2024年底，公司全球员工总数已超过1200人，其中研发人员占比约60%。

## 融资历程

2021年，星辰科技完成**B轮融资5000万美元**，由**红杉资本中国**领投，腾讯投资、云九资本跟投。

## 战略合作

公司与**阿里云**建立深度合作，依托阿里云的弹性计算与PAI机器学习平台构建产品底座；同时与**腾讯云**在私有化部署及金融行业解决方案方面保持紧密合作。

## 竞争格局

主要竞争对手包括**明略科技**和**神策数据**。星辰科技以全栈AI能力与一体化产品矩阵形成差异化竞争优势。`
  },
  {
    title: 'product_manual.md',
    content: `# 智星 AI营销平台 产品使用手册

## 一、产品概述

智星AI营销平台是由星辰科技有限公司研发的一站式智能营销解决方案。平台依托大语言模型与多模态AI技术，帮助企业在微信公众号、抖音、微博、淘宝等主流渠道实现营销内容的自动化生成、精准投放与效果追踪。

## 二、核心功能模块

### 2.1 多渠道营销管理

支持统一工作台管理微信公众号、抖音企业号、微博蓝V、淘宝店铺等渠道。

操作步骤：
1. 进入"渠道管理"菜单，点击"绑定渠道"
2. 选择目标平台，按提示完成OAuth授权
3. 授权成功后，渠道状态显示为"已连接"
4. 在"发布中心"新建内容，勾选目标渠道后点击"一键发布"

### 2.2 AI内容生成

基于GPT级大语言模型，输入产品关键词、目标人群与营销目的，平台可在10秒内自动生成标题、正文、标签及配图建议。

### 2.3 用户画像分析

构建360°用户画像，支持按消费偏好、活跃时段、地域分布、互动行为等多维度进行用户分群。

### 2.4 ROI分析与报表

实时追踪每次营销活动的曝光量、点击率、转化率及获客成本（CAC）。

## 三、定价方案

| 版本 | 月费 | 适用规模 |
|------|------|---------|
| 基础版 | ¥1,999/月 | 1-5人团队 |
| 专业版 | ¥4,999/月 | 5-50人团队 |
| 企业版 | 定制报价 | 50人以上 |

## 四、技术要求

- 浏览器：Chrome 90及以上版本
- 内存：最低4GB RAM
- 网络：稳定宽带连接，上传速度≥5Mbps

## 五、技术支持

- 邮件支持：support@starlight-ai.com
- 电话支持：400-888-9999（工作日 9:00-18:00）`
  },
  {
    title: 'faq.md',
    content: `# 星辰科技 · 智星AI营销平台 综合问答手册

Q1：智星平台是哪家公司的产品？
A：智星AI营销平台由**星辰科技有限公司**自主研发，成立于2018年，总部位于上海张江高科技园区。

Q2：如何注册并开通智星平台账号？
A：访问 https://app.starlight-ai.com，点击"免费试用"，填写企业信息完成手机验证码校验后激活14天免费试用。

Q3：智星平台支持哪些内容发布渠道？
A：支持微信公众号、抖音企业号、微博蓝V、淘宝店铺。

Q4：智星平台的定价体系是怎样的？
A：基础版¥1,999/月、专业版¥4,999/月、企业版定制报价。年付享受8.5折优惠。

Q5：星辰科技获得了哪些融资？
A：2021年完成B轮融资5000万美元，由红杉资本中国领投，腾讯投资和云九资本跟投。CEO张明远表示资金主要用于产品研发与销售网络扩张。

Q6：智星、数睿、星客三款产品有什么区别？
A：智星专注营销内容自动化，数睿专注数据分析可视化，星客专注智能客服对话。三款产品可独立使用也可打通协同。

Q7：技术问题如何联系支持？
A：邮件：support@starlight-ai.com；电话：400-888-9999；企业版客户可通过星客智能客服系统获得7×24小时AI辅助支持。

Q8：智星平台的竞争对手是哪些？
A：主要竞争对手为明略科技和神策数据。星辰科技优势在于一体化平台与低上手门槛。`
  }
]

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString('zh-CN', {
    month: 'numeric', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function fileIcon(docType: string, filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase()
  if (ext === 'pdf') return '📄'
  if (ext === 'md') return '📝'
  if (ext === 'txt') return '📃'
  if (ext === 'docx' || ext === 'doc') return '📋'
  if (ext === 'html' || ext === 'htm') return '🌐'
  if (docType === 'text') return '📃'
  return '📁'
}

function formatBytes(n: number): string {
  if (n < 1000) return `${n}字`
  if (n < 10000) return `${(n / 1000).toFixed(1)}K字`
  return `${Math.round(n / 1000)}K字`
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: KBDoc['status'] }) {
  if (status === 'ready') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
        style={{ background: '#052e16', color: '#4ade80', border: '1px solid #166534' }}>
        <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />
        就绪
      </span>
    )
  }
  if (status === 'indexing') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium animate-pulse"
        style={{ background: '#0c1a3d', color: '#60a5fa', border: '1px solid #1e40af' }}>
        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 inline-block" />
        索引中
      </span>
    )
  }
  if (status === 'error') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
        style={{ background: '#2d0a0a', color: '#f87171', border: '1px solid #991b1b' }}>
        <span className="w-1.5 h-1.5 rounded-full bg-red-400 inline-block" />
        错误
      </span>
    )
  }
  // pending
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
      style={{ background: '#1c1c2e', color: '#94a3b8', border: '1px solid #334155' }}>
      <span className="w-1.5 h-1.5 rounded-full bg-slate-400 inline-block" />
      等待中
    </span>
  )
}

function Toast({ message, onDone }: { message: string; onDone: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDone, 3000)
    return () => clearTimeout(t)
  }, [onDone])

  return (
    <div className="fixed bottom-6 right-6 z-50 px-4 py-3 rounded-xl text-sm font-medium shadow-2xl animate-in fade-in slide-in-from-bottom-2"
      style={{ background: '#7c3aed', color: '#fff', border: '1px solid rgba(139,92,246,0.5)' }}>
      {message}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function KnowledgePage() {
  const [documents, setDocuments] = useState<KBDoc[]>([])
  const [messages, setMessages] = useState<KbMessage[]>([])
  const [stats, setStats] = useState<KBStats | null>(null)
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'qa' | 'citations'>('qa')
  const [inputText, setInputText] = useState('')
  const [toast, setToast] = useState<string | null>(null)
  const [loadingDemo, setLoadingDemo] = useState(false)
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set())
  const [graphBuilding, setGraphBuilding] = useState<Set<string>>(new Set())

  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatBottomRef = useRef<HTMLDivElement>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── API calls ──────────────────────────────────────────────────────────────

  const loadDocuments = useCallback(async () => {
    try {
      const res = await fetch(`/api/agent/kb/documents?kb_id=${KB_ID}`)
      if (!res.ok) return
      const data = await res.json()
      setDocuments(data.documents ?? data ?? [])
    } catch { /* silent */ }
  }, [])

  const loadStats = useCallback(async () => {
    try {
      const res = await fetch(`/api/agent/kb/stats?kb_id=${KB_ID}`)
      if (!res.ok) return
      const data = await res.json()
      setStats(data)
    } catch { /* silent */ }
  }, [])

  useEffect(() => {
    loadDocuments()
    loadStats()
  }, [loadDocuments, loadStats])

  // Auto-refresh while any doc is indexing
  useEffect(() => {
    const hasIndexing = documents.some(d => d.status === 'indexing' || d.status === 'pending')
    if (hasIndexing) {
      if (!pollingRef.current) {
        pollingRef.current = setInterval(() => {
          loadDocuments()
          loadStats()
        }, 3000)
      }
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
  }, [documents, loadDocuments, loadStats])

  // Auto-scroll chat
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Upload ─────────────────────────────────────────────────────────────────

  async function uploadFile(file: File) {
    setUploading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('kb_id', KB_ID)
      const res = await fetch('/api/agent/kb/documents/upload', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        setToast(`上传失败：${err.detail ?? res.statusText}`)
      } else {
        setToast(`"${file.name}" 上传成功，正在建立索引…`)
        await loadDocuments()
        await loadStats()
      }
    } catch (e: unknown) {
      setToast(`上传出错：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setUploading(false)
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) uploadFile(file)
    e.target.value = ''
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) uploadFile(file)
  }

  // ── Delete ─────────────────────────────────────────────────────────────────

  async function deleteDoc(docId: string) {
    try {
      const res = await fetch(`/api/agent/kb/documents/${docId}?kb_id=${KB_ID}`, { method: 'DELETE' })
      if (res.ok) {
        setDocuments(prev => prev.filter(d => d.doc_id !== docId))
        if (selectedDocId === docId) setSelectedDocId(null)
        await loadStats()
        setToast('文档已删除')
      } else {
        setToast('删除失败')
      }
    } catch {
      setToast('删除出错')
    }
  }

  // ── Build Graph ────────────────────────────────────────────────────────────

  async function buildGraph(docId: string) {
    setGraphBuilding(prev => new Set(prev).add(docId))
    setToast('图谱生成中…')
    try {
      await fetch(`/api/agent/kb/build-graph/${docId}?kb_id=${KB_ID}`, { method: 'POST' })
    } catch { /* silent */ } finally {
      setGraphBuilding(prev => { const s = new Set(prev); s.delete(docId); return s })
    }
  }

  // ── Ask ────────────────────────────────────────────────────────────────────

  async function askQuestion(query: string) {
    if (!query.trim() || asking) return
    const userMsg: KbMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: query.trim(),
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])
    setInputText('')
    setAsking(true)
    try {
      const res = await fetch('/api/agent/kb/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), kb_id: KB_ID, top_k: 5, with_graph: false }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        const errMsg: KbMessage = {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: `请求失败：${err.detail ?? res.statusText}`,
          timestamp: new Date(),
        }
        setMessages(prev => [...prev, errMsg])
      } else {
        const data = await res.json()
        const assistantMsg: KbMessage = {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: data.answer ?? data.content ?? JSON.stringify(data),
          citations: data.citations ?? data.sources ?? [],
          timestamp: new Date(),
        }
        setMessages(prev => [...prev, assistantMsg])
      }
    } catch (e: unknown) {
      const errMsg: KbMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: `网络错误：${e instanceof Error ? e.message : String(e)}`,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setAsking(false)
    }
  }

  // ── Load demo documents ────────────────────────────────────────────────────

  async function loadDemoDocuments() {
    setLoadingDemo(true)
    let successCount = 0
    for (const doc of DEMO_DOCS) {
      try {
        const res = await fetch('/api/agent/kb/documents/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: doc.title, content: doc.content, kb_id: KB_ID }),
        })
        if (res.ok) successCount++
      } catch { /* skip */ }
    }
    setToast(successCount > 0 ? `已加载 ${successCount} 个示例文档` : '加载示例文档失败，请检查 API 连接')
    await loadDocuments()
    await loadStats()
    setLoadingDemo(false)
  }

  // ── Citation toggle ────────────────────────────────────────────────────────

  function toggleCitation(key: string) {
    setExpandedCitations(prev => {
      const s = new Set(prev)
      s.has(key) ? s.delete(key) : s.add(key)
      return s
    })
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  const hasDocuments = documents.length > 0
  const allCitations = messages
    .filter(m => m.role === 'assistant' && m.citations && m.citations.length > 0)
    .flatMap(m => m.citations!)

  return (
    <div className="flex flex-col h-screen" style={{ background: '#020617', color: '#e2e8f0' }}>

      {/* ── Top Bar ── */}
      <header className="flex items-center justify-between px-6 py-3 border-b flex-shrink-0"
        style={{ background: '#0f172a', borderColor: '#1e293b' }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-base"
            style={{ background: 'rgba(124,58,237,0.2)', border: '1px solid rgba(124,58,237,0.4)' }}>
            📚
          </div>
          <div>
            <h1 className="font-semibold text-base" style={{ color: '#f1f5f9' }}>知识库问答</h1>
            <p className="text-xs" style={{ color: '#64748b' }}>Knowledge Base Q&A</p>
          </div>
        </div>

        {/* Stats chips */}
        <div className="flex items-center gap-2">
          {stats && (
            <>
              <span className="px-3 py-1 rounded-full text-xs font-medium"
                style={{ background: 'rgba(124,58,237,0.15)', color: '#a78bfa', border: '1px solid rgba(124,58,237,0.3)' }}>
                {stats.doc_count} 文档
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium"
                style={{ background: 'rgba(6,182,212,0.1)', color: '#67e8f9', border: '1px solid rgba(6,182,212,0.25)' }}>
                {stats.chunk_count} 片段
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium"
                style={{ background: 'rgba(52,211,153,0.1)', color: '#6ee7b7', border: '1px solid rgba(52,211,153,0.25)' }}>
                {formatBytes(stats.total_chars)}
              </span>
            </>
          )}

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all"
            style={{
              background: uploading ? 'rgba(124,58,237,0.3)' : '#7c3aed',
              color: '#fff',
              border: '1px solid rgba(139,92,246,0.5)',
              cursor: uploading ? 'not-allowed' : 'pointer',
            }}>
            {uploading ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
                上传中…
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
                上传文档
              </>
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.txt,.md,.docx,.html"
            onChange={handleFileChange}
          />
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 min-h-0">

        {/* ── Left Panel: Document List ── */}
        <aside className="w-72 flex-shrink-0 flex flex-col border-r" style={{ background: '#0f172a', borderColor: '#1e293b' }}>

          {/* Drag-drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="mx-3 mt-3 rounded-xl flex flex-col items-center justify-center gap-1.5 cursor-pointer transition-all duration-200 py-4"
            style={{
              border: `2px dashed ${dragOver ? '#7c3aed' : '#334155'}`,
              background: dragOver ? 'rgba(124,58,237,0.08)' : 'transparent',
              minHeight: '80px',
            }}>
            <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
              style={{ color: dragOver ? '#a78bfa' : '#475569' }}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.338-2.32 5.75 5.75 0 011.048 11.09H6.75z" />
            </svg>
            <p className="text-xs text-center leading-relaxed" style={{ color: dragOver ? '#a78bfa' : '#64748b' }}>
              拖拽文件到此处或点击上传
              <br />
              <span style={{ opacity: 0.6 }}>支持 .pdf .txt .md .docx .html</span>
            </p>
          </div>

          {/* Document count header */}
          <div className="flex items-center justify-between px-4 pt-3 pb-1.5">
            <span className="text-xs font-medium uppercase tracking-wider" style={{ color: '#64748b' }}>
              文档列表
            </span>
            <span className="text-xs px-1.5 py-0.5 rounded-md font-mono"
              style={{ background: '#1e293b', color: '#94a3b8' }}>
              {documents.length}
            </span>
          </div>

          {/* Document list */}
          <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1.5">
            {documents.length === 0 ? (
              <div className="py-6 text-center text-xs" style={{ color: '#475569' }}>
                暂无文档
              </div>
            ) : (
              documents.map(doc => (
                <div
                  key={doc.doc_id}
                  onClick={() => setSelectedDocId(selectedDocId === doc.doc_id ? null : doc.doc_id)}
                  className="rounded-xl p-3 cursor-pointer transition-all duration-150 group"
                  style={{
                    background: selectedDocId === doc.doc_id ? 'rgba(124,58,237,0.12)' : '#1e293b',
                    border: `1px solid ${selectedDocId === doc.doc_id ? 'rgba(124,58,237,0.4)' : '#334155'}`,
                  }}>
                  <div className="flex items-start gap-2">
                    <span className="text-base flex-shrink-0 mt-0.5">{fileIcon(doc.doc_type, doc.filename)}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate" style={{ color: '#e2e8f0' }}
                        title={doc.filename}>
                        {doc.filename}
                      </p>
                      <div className="flex items-center gap-2 mt-1 flex-wrap">
                        <StatusBadge status={doc.status} />
                        {doc.chunk_count > 0 && (
                          <span className="text-xs" style={{ color: '#64748b' }}>
                            {doc.chunk_count} 块
                          </span>
                        )}
                        {doc.char_count > 0 && (
                          <span className="text-xs" style={{ color: '#64748b' }}>
                            {formatBytes(doc.char_count)}
                          </span>
                        )}
                      </div>
                      {doc.error_msg && (
                        <p className="text-xs mt-1 truncate" style={{ color: '#f87171' }}
                          title={doc.error_msg}>
                          {doc.error_msg}
                        </p>
                      )}
                      <p className="text-xs mt-1" style={{ color: '#475569' }}>
                        {formatDate(doc.created_at)}
                      </p>
                    </div>
                  </div>

                  {/* Action buttons */}
                  <div className="flex items-center gap-1.5 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    {doc.status === 'ready' && (
                      <button
                        onClick={(e) => { e.stopPropagation(); buildGraph(doc.doc_id) }}
                        disabled={graphBuilding.has(doc.doc_id)}
                        className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium transition-all"
                        style={{
                          background: 'rgba(124,58,237,0.2)',
                          color: '#a78bfa',
                          border: '1px solid rgba(124,58,237,0.35)',
                          cursor: graphBuilding.has(doc.doc_id) ? 'not-allowed' : 'pointer',
                          opacity: graphBuilding.has(doc.doc_id) ? 0.6 : 1,
                        }}>
                        {graphBuilding.has(doc.doc_id) ? (
                          <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                          </svg>
                        ) : (
                          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="5" r="2" /><circle cx="5" cy="19" r="2" /><circle cx="19" cy="19" r="2" />
                            <line x1="12" y1="7" x2="5" y2="17" /><line x1="12" y1="7" x2="19" y2="17" />
                          </svg>
                        )}
                        生成图谱
                      </button>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteDoc(doc.doc_id) }}
                      className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium transition-all ml-auto"
                      style={{
                        background: 'rgba(239,68,68,0.1)',
                        color: '#f87171',
                        border: '1px solid rgba(239,68,68,0.25)',
                      }}>
                      <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      删除
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* ── Main Panel ── */}
        <main className="flex-1 flex flex-col min-w-0" style={{ background: '#020617' }}>

          {!hasDocuments ? (
            // ── Empty State ──
            <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl"
                style={{ background: 'rgba(124,58,237,0.15)', border: '1px solid rgba(124,58,237,0.3)' }}>
                📚
              </div>
              <div className="text-center">
                <h2 className="text-xl font-semibold mb-2" style={{ color: '#f1f5f9' }}>知识库尚未包含文档</h2>
                <p className="text-sm" style={{ color: '#64748b' }}>
                  上传文档后，即可通过自然语言提问并获得基于文档内容的精准回答。
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-3 items-center">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium"
                  style={{ background: '#7c3aed', color: '#fff', border: '1px solid rgba(139,92,246,0.5)' }}>
                  <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                  </svg>
                  上传文档
                </button>
                <span className="text-xs" style={{ color: '#475569' }}>或者</span>
                <button
                  onClick={loadDemoDocuments}
                  disabled={loadingDemo}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all"
                  style={{
                    background: loadingDemo ? 'rgba(6,182,212,0.1)' : 'rgba(6,182,212,0.15)',
                    color: '#67e8f9',
                    border: '1px solid rgba(6,182,212,0.3)',
                    cursor: loadingDemo ? 'not-allowed' : 'pointer',
                  }}>
                  {loadingDemo ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                      </svg>
                      加载中…
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
                      </svg>
                      加载示例文档
                    </>
                  )}
                </button>
              </div>
              <p className="text-xs text-center" style={{ color: '#334155' }}>
                示例文档包含：星辰科技公司介绍 · 智星平台产品手册 · 常见问题解答
              </p>
            </div>
          ) : (
            // ── Q&A Interface ──
            <>
              {/* Tabs */}
              <div className="flex items-center gap-1 px-4 pt-3 pb-0 border-b flex-shrink-0"
                style={{ borderColor: '#1e293b' }}>
                {([['qa', '问答'], ['citations', '引用来源']] as const).map(([tab, label]) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className="px-4 py-2 text-sm font-medium rounded-t-lg transition-all border-b-2"
                    style={{
                      color: activeTab === tab ? '#a78bfa' : '#64748b',
                      borderBottomColor: activeTab === tab ? '#7c3aed' : 'transparent',
                      background: activeTab === tab ? 'rgba(124,58,237,0.08)' : 'transparent',
                    }}>
                    {label}
                    {tab === 'citations' && allCitations.length > 0 && (
                      <span className="ml-1.5 text-xs px-1.5 py-0.5 rounded-full"
                        style={{ background: 'rgba(124,58,237,0.2)', color: '#a78bfa' }}>
                        {allCitations.length}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {activeTab === 'qa' ? (
                <>
                  {/* Chat messages */}
                  <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">

                    {messages.length === 0 ? (
                      /* Demo prompts */
                      <div className="flex flex-col items-center justify-center h-full gap-6 pb-8">
                        <div className="text-center">
                          <div className="w-12 h-12 rounded-xl mx-auto mb-3 flex items-center justify-center text-2xl"
                            style={{ background: 'rgba(124,58,237,0.15)', border: '1px solid rgba(124,58,237,0.3)' }}>
                            💬
                          </div>
                          <h3 className="font-medium mb-1" style={{ color: '#e2e8f0' }}>开始提问</h3>
                          <p className="text-sm" style={{ color: '#64748b' }}>
                            基于已上传的文档回答您的问题
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                          {DEMO_PROMPTS.map(prompt => (
                            <button
                              key={prompt}
                              onClick={() => askQuestion(prompt)}
                              className="px-4 py-2 rounded-xl text-sm transition-all hover:scale-105"
                              style={{
                                background: '#1e293b',
                                color: '#94a3b8',
                                border: '1px solid #334155',
                              }}>
                              {prompt}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      messages.map(msg => (
                        <div key={msg.id}
                          className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>

                          {/* Avatar */}
                          <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs flex-shrink-0 mt-0.5"
                            style={{
                              background: msg.role === 'user'
                                ? 'rgba(124,58,237,0.3)'
                                : 'rgba(6,182,212,0.2)',
                              border: `1px solid ${msg.role === 'user' ? 'rgba(124,58,237,0.5)' : 'rgba(6,182,212,0.3)'}`,
                            }}>
                            {msg.role === 'user' ? '你' : '✦'}
                          </div>

                          <div className={`max-w-[72%] ${msg.role === 'user' ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
                            {/* Bubble */}
                            <div className="px-4 py-3 rounded-2xl text-sm leading-relaxed"
                              style={{
                                background: msg.role === 'user'
                                  ? '#7c3aed'
                                  : '#1e293b',
                                color: msg.role === 'user' ? '#fff' : '#e2e8f0',
                                borderRadius: msg.role === 'user'
                                  ? '18px 18px 4px 18px'
                                  : '18px 18px 18px 4px',
                                border: msg.role === 'user'
                                  ? '1px solid rgba(139,92,246,0.4)'
                                  : '1px solid #334155',
                              }}>
                              <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
                            </div>

                            {/* Citations chips */}
                            {msg.citations && msg.citations.length > 0 && (
                              <div className="flex flex-wrap gap-1.5 mt-1">
                                {msg.citations.map((cit, i) => {
                                  const key = `${msg.id}-${i}`
                                  const expanded = expandedCitations.has(key)
                                  return (
                                    <div key={key} className="flex flex-col gap-1">
                                      <button
                                        onClick={() => toggleCitation(key)}
                                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs transition-all"
                                        style={{
                                          background: 'rgba(124,58,237,0.12)',
                                          color: '#a78bfa',
                                          border: '1px solid rgba(124,58,237,0.25)',
                                        }}>
                                        <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                                          <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
                                        </svg>
                                        来源 {cit.index ?? i + 1}
                                        {cit.filename && (
                                          <span style={{ opacity: 0.7 }}>· {cit.filename}</span>
                                        )}
                                        {cit.score !== undefined && (
                                          <span style={{ opacity: 0.6 }}>
                                            {(cit.score * 100).toFixed(0)}%
                                          </span>
                                        )}
                                        <svg
                                          className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`}
                                          viewBox="0 0 20 20" fill="currentColor">
                                          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                                        </svg>
                                      </button>
                                      {expanded && (
                                        <div className="px-3 py-2 rounded-lg text-xs leading-relaxed max-w-sm"
                                          style={{
                                            background: '#0f172a',
                                            color: '#94a3b8',
                                            border: '1px solid #1e293b',
                                          }}>
                                          {cit.text_preview}
                                        </div>
                                      )}
                                    </div>
                                  )
                                })}
                              </div>
                            )}

                            {/* Timestamp */}
                            <span className="text-xs px-1" style={{ color: '#475569' }}>
                              {msg.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </div>
                        </div>
                      ))
                    )}

                    {/* Asking indicator */}
                    {asking && (
                      <div className="flex gap-3">
                        <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs"
                          style={{ background: 'rgba(6,182,212,0.2)', border: '1px solid rgba(6,182,212,0.3)' }}>
                          ✦
                        </div>
                        <div className="px-4 py-3 rounded-2xl text-sm"
                          style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '18px 18px 18px 4px' }}>
                          <div className="flex items-center gap-1.5">
                            {[0, 1, 2].map(i => (
                              <div key={i} className="w-1.5 h-1.5 rounded-full animate-bounce"
                                style={{ background: '#64748b', animationDelay: `${i * 150}ms` }} />
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    <div ref={chatBottomRef} />
                  </div>

                  {/* Input bar */}
                  <div className="px-4 pb-4 pt-2 flex-shrink-0 border-t" style={{ borderColor: '#1e293b' }}>
                    <div className="flex items-end gap-2 rounded-2xl px-4 py-3"
                      style={{ background: '#0f172a', border: '1px solid #334155' }}>
                      <textarea
                        className="flex-1 bg-transparent text-sm resize-none outline-none leading-relaxed"
                        style={{ color: '#e2e8f0', minHeight: '24px', maxHeight: '120px' }}
                        placeholder="输入问题，按 Enter 发送…"
                        rows={1}
                        value={inputText}
                        onChange={e => {
                          setInputText(e.target.value)
                          e.target.style.height = 'auto'
                          e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                        }}
                        onKeyDown={e => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            askQuestion(inputText)
                          }
                        }}
                      />
                      <button
                        onClick={() => askQuestion(inputText)}
                        disabled={asking || !inputText.trim()}
                        className="flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all"
                        style={{
                          background: asking || !inputText.trim() ? 'rgba(124,58,237,0.2)' : '#7c3aed',
                          color: '#fff',
                          cursor: asking || !inputText.trim() ? 'not-allowed' : 'pointer',
                        }}>
                        <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                        </svg>
                      </button>
                    </div>
                    <p className="text-xs mt-1.5 text-center" style={{ color: '#334155' }}>
                      Enter 发送 · Shift+Enter 换行
                    </p>
                  </div>
                </>
              ) : (
                // ── Citations Tab ──
                <div className="flex-1 overflow-y-auto px-4 py-4">
                  {allCitations.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: '#475569' }}>
                      <svg className="w-10 h-10 opacity-30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path strokeLinecap="round" strokeLinejoin="round"
                          d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                      </svg>
                      <p className="text-sm">提问后将在此显示引用来源</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <p className="text-xs font-medium uppercase tracking-wider" style={{ color: '#64748b' }}>
                        全部引用来源 ({allCitations.length})
                      </p>
                      {allCitations.map((cit, i) => (
                        <div key={i} className="rounded-xl p-4"
                          style={{ background: '#1e293b', border: '1px solid #334155' }}>
                          <div className="flex items-center gap-2 mb-2">
                            <span className="w-5 h-5 rounded-md flex items-center justify-center text-xs font-bold flex-shrink-0"
                              style={{ background: 'rgba(124,58,237,0.2)', color: '#a78bfa' }}>
                              {cit.index ?? i + 1}
                            </span>
                            <span className="text-sm font-medium truncate" style={{ color: '#e2e8f0' }}>
                              {cit.filename || cit.source}
                            </span>
                            {cit.score !== undefined && (
                              <span className="ml-auto text-xs px-2 py-0.5 rounded-full flex-shrink-0"
                                style={{ background: 'rgba(52,211,153,0.1)', color: '#6ee7b7', border: '1px solid rgba(52,211,153,0.2)' }}>
                                相关度 {(cit.score * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                          <p className="text-xs leading-relaxed" style={{ color: '#94a3b8' }}>
                            {cit.text_preview}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {/* Toast */}
      {toast && <Toast message={toast} onDone={() => setToast(null)} />}
    </div>
  )
}
