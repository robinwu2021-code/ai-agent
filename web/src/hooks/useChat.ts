'use client'
import { useState, useCallback, useRef, useEffect } from 'react'
import type { Message, SkillMode, SessionStats, ToolCall, DagStep, SubTaskInfo } from '@/types'
import { streamChat } from '@/lib/api'
import {
  genId,
  todayMidnightTs, tomorrowMidnightTs, daysMidnightTs,
  thisWeekStartTs, nextWeekStartTs, lastWeekStartTs,
  thisMonthStartTs, nextMonthStartTs,
} from '@/lib/utils'
import { parseStructuredData } from '@/lib/parseStructuredData'
import { useApp } from '@/contexts/AppContext'
import type { ChatSession } from '@/contexts/AppContext'

const SKILL_MAP: Record<SkillMode, string[] | null> = {
  general: null,
  weather: ['weather_current'],
  report: ['agent_bi'],
  marketing: ['marketing_advisor'],
  graph: null,
}

const QUICK_MAP: Record<SkillMode, string[]> = {
  general: ['今天北京天气怎么样？', '帮我写一首关于春天的短诗', '解释量子纠缠是什么'],
  weather: ['查询上海今天天气', '北京未来3天天气', '深圳今天下雨吗'],
  report: [
    '查询本店今日总销售额和订单数',
    '统计本周堂食和外带销售额对比',
    '查询昨天的会员订单数和新增会员数',
  ],
  marketing: [
    '我是一家烘焙店，想做一个国庆活动',
    '帮我做一个周年庆营销方案',
    '我的咖啡馆想吸引更多年轻顾客',
  ],
  graph: ['探索知识图谱', '查找实体关系', '分析图谱结构'],
}

export function useChat() {
  const appCtx = useApp()
  const { currentUser, selectedWorkspace, selectedProject, saveSession, currentSessionId, sessions } = appCtx

  const [messages, setMessages] = useState<Message[]>([])
  const [mode, setMode] = useState<SkillMode>('general')
  const [busy, setBusy] = useState(false)
  const [stats, setStats] = useState<SessionStats>({ calls: 0, totalTokens: 0, toolCalls: 0 })
  /** 当前报表门店 ID；空字符串 = 全部门店（后端自动回落到默认配置） */
  const [braId, setBraId] = useState<string>('')

  // Use a stable ref for session id; regenerated on clearSession
  const sessionId = useRef<string>('sess_' + genId())
  // Track session creation time for saveSession
  const sessionCreatedAt = useRef<number>(Date.now())

  // ── Load a saved session when currentSessionId changes ───────────────────
  const prevSessionIdRef = useRef<string | null>(null)

  useEffect(() => {
    if (currentSessionId === prevSessionIdRef.current) return
    prevSessionIdRef.current = currentSessionId

    if (currentSessionId === null) return

    const found = sessions.find(s => s.session_id === currentSessionId)
    if (!found) return

    loadSession(found)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, sessions])

  // ── loadSession ───────────────────────────────────────────────────────────
  function loadSession(session: ChatSession) {
    // Deserialise message timestamps back into Date objects
    const restored: Message[] = (session.messages ?? []).map((m: any) => ({
      ...m,
      timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
    }))
    setMessages(restored)
    sessionId.current = session.session_id
    sessionCreatedAt.current = session.created_at
    setStats({ calls: 0, totalTokens: 0, toolCalls: 0 })
  }

  // ── Helper: persist current conversation ─────────────────────────────────
  function persistSession(currentMessages: Message[]) {
    const firstUser = currentMessages.find(m => m.role === 'user')
    const title = firstUser
      ? firstUser.content.slice(0, 30)
      : '新对话'

    const session: ChatSession = {
      session_id: sessionId.current,
      title,
      mode,
      workspace_id: selectedWorkspace?.workspace_id,
      project_id: selectedProject?.project_id,
      messages: currentMessages,
      created_at: sessionCreatedAt.current,
      updated_at: Date.now(),
    }
    saveSession(session)
  }

  // ── send ──────────────────────────────────────────────────────────────────
  const send = useCallback(async (text: string) => {
    if (!text.trim() || busy) return
    setBusy(true)

    const userMsg: Message = {
      id: genId(),
      role: 'user',
      content: text,
      timestamp: new Date(),
      toolCalls: [],
    }

    let updatedMessages: Message[] = []
    setMessages(prev => {
      updatedMessages = [...prev, userMsg]
      return updatedMessages
    })

    const asstId = genId()
    const asstMsg: Message = {
      id: asstId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      toolCalls: [],
      isStreaming: true,
    }
    setMessages(prev => {
      updatedMessages = [...prev, asstMsg]
      return updatedMessages
    })

    // Build request — inject date range anchors for BI queries so the LLM
    // can pick rangeStart/rangeEnd directly without computing timestamps itself.
    let enrichedText = text
    if (mode === 'report') {
      const today = new Date()
      const todayStart  = todayMidnightTs()
      const weekStart   = thisWeekStartTs()
      const monthStart  = thisMonthStartTs()
      // YoY helpers: same period one year ago
      const msPerYear = 365.25 * 24 * 3600 * 1000
      const yoyOffset = Math.round(msPerYear)
      const dateCtx = [
        `[BI date context — ${today.toISOString().slice(0, 10)}]`,
        `RULE: for any single day, rangeEnd = rangeStart + 86400000 (exactly +24 h, never equal to rangeStart)`,
        `-- Current periods --`,
        `today:           rangeStart=${todayStart}, rangeEnd=${todayStart + 86400000}`,
        `yesterday:       rangeStart=${todayStart - 86400000}, rangeEnd=${todayStart}`,
        `this_week:       rangeStart=${weekStart}, rangeEnd=${nextWeekStartTs()}`,
        `last_week:       rangeStart=${lastWeekStartTs()}, rangeEnd=${weekStart}`,
        `last_7days:      rangeStart=${todayStart - 7 * 86400000}, rangeEnd=${todayStart + 86400000}`,
        `this_month:      rangeStart=${monthStart}, rangeEnd=${nextMonthStartTs()}`,
        `-- 同比 (yoy) comparison periods — same period last year --`,
        `yoy_today:       rangeStart=${todayStart - yoyOffset}, rangeEnd=${todayStart - yoyOffset + 86400000}`,
        `yoy_yesterday:   rangeStart=${todayStart - 86400000 - yoyOffset}, rangeEnd=${todayStart - yoyOffset}`,
        `yoy_this_week:   rangeStart=${weekStart - yoyOffset}, rangeEnd=${weekStart - yoyOffset + 7 * 86400000}`,
        `yoy_this_month:  rangeStart=${monthStart - yoyOffset}, rangeEnd=${nextMonthStartTs() - yoyOffset}`,
        `-- 环比 (pop) comparison periods — immediately preceding period --`,
        `pop_today:       rangeStart=${todayStart - 86400000}, rangeEnd=${todayStart}`,
        `pop_this_week:   rangeStart=${lastWeekStartTs()}, rangeEnd=${weekStart}`,
        `pop_this_month:  rangeStart=${monthStart - yoyOffset / 12}, rangeEnd=${monthStart}`,
        `-- Comparison instructions --`,
        `When user asks for 同比 (year-over-year), set comparison="yoy". When user asks for 环比 (period-over-period), set comparison="pop".`,
      ].join('\n')
      enrichedText = text + '\n\n' + dateCtx
    }

    const req = {
      text: enrichedText,
      user_id: currentUser.user_id,
      session_id: sessionId.current,
      max_steps: 15,
      skills: SKILL_MAP[mode] ?? undefined,
      mode: mode === 'marketing' ? 'marketing' : undefined,
      // bra_id passed as structured field; backend falls back to AGENT_BI_DEFAULT_BRA_ID
      bra_id: mode === 'report' && braId.trim() ? braId.trim() : undefined,
      workspace_id: selectedWorkspace?.workspace_id,
      project_id: selectedProject?.project_id,
    }

    try {
      let content = ''
      const toolMap = new Map<string, ToolCall>()
      const dagStepMap = new Map<string, DagStep>()
      const subTaskMap = new Map<string, SubTaskInfo>()
      let toolCallsCount = 0

      for await (const ev of streamChat(req)) {
        if (ev.type === 'delta') {
          content += ev.text
          setMessages(prev => {
            const next = prev.map(m => m.id === asstId ? { ...m, content } : m)
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'step') {
          const tcId = `tc-${ev.step}-${ev.tool}`
          const existing = toolMap.get(tcId)
          const tc: ToolCall = existing
            ? { ...existing, status: ev.status as ToolCall['status'], error: ev.error }
            : { id: tcId, name: ev.tool, status: ev.status as ToolCall['status'], error: ev.error }
          toolMap.set(tcId, tc)
          if (ev.status !== 'running') toolCallsCount++
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId
                ? { ...m, toolCalls: Array.from(toolMap.values()) }
                : m
            )
            updatedMessages = next
            return next
          })

        // ── DAG events ────────────────────────────────────────────
        } else if (ev.type === 'plan') {
          for (const s of ev.steps) {
            dagStepMap.set(s.id, { id: s.id, goal: s.goal, status: 'pending' })
          }
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'parallel_start') {
          for (const sid of ev.step_ids) {
            const s = dagStepMap.get(sid)
            if (s) dagStepMap.set(sid, { ...s, status: 'running', parallelGroup: ev.step_ids })
          }
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'step_start') {
          const s = dagStepMap.get(ev.step_id)
          if (s) dagStepMap.set(ev.step_id, { ...s, status: 'running' })
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'step_done') {
          const s = dagStepMap.get(ev.step_id)
          if (s) dagStepMap.set(ev.step_id, { ...s, status: 'done', result: ev.result })
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'step_failed') {
          const s = dagStepMap.get(ev.step_id)
          if (s) dagStepMap.set(ev.step_id, { ...s, status: 'error', error: ev.error })
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
            )
            updatedMessages = next
            return next
          })

        // ── MultiAgent events ─────────────────────────────────────
        } else if (ev.type === 'subtask_assign') {
          subTaskMap.set(ev.subtask_id, {
            id: ev.subtask_id, agent: ev.agent, goal: ev.goal,
            depends: ev.depends, status: 'pending',
          })
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'agent_start') {
          const st = subTaskMap.get(ev.subtask_id)
          if (st) subTaskMap.set(ev.subtask_id, { ...st, status: 'running' })
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'agent_done') {
          const st = subTaskMap.get(ev.subtask_id)
          if (st) subTaskMap.set(ev.subtask_id, { ...st, status: 'done', tokens: ev.tokens })
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'agent_error') {
          const st = subTaskMap.get(ev.subtask_id)
          if (st) subTaskMap.set(ev.subtask_id, { ...st, status: 'error', error: ev.error })
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
            )
            updatedMessages = next
            return next
          })

        } else if (ev.type === 'done') {
          const structured = parseStructuredData(content)
          setStats(s => ({
            calls: s.calls + 1,
            totalTokens: s.totalTokens + (ev.usage?.total_tokens ?? 0),
            toolCalls: s.toolCalls + toolCallsCount,
          }))
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId
                ? { ...m, isStreaming: false, structuredData: structured }
                : m
            )
            updatedMessages = next
            return next
          })
        } else if (ev.type === 'error') {
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === asstId
                ? { ...m, content: `⚠ ${ev.message}`, isStreaming: false }
                : m
            )
            updatedMessages = next
            return next
          })
        }
      }

      // ensure streaming flag is cleared
      setMessages(prev => {
        const next = prev.map(m => m.id === asstId ? { ...m, isStreaming: false } : m)
        updatedMessages = next
        return next
      })

      // Persist session after response completes
      // Use a small timeout so state flush is guaranteed
      setTimeout(() => {
        setMessages(latest => {
          persistSession(latest)
          return latest
        })
      }, 0)

    } catch (err) {
      const errMsg = err instanceof Error ? err.message : '未知错误'
      setMessages(prev => {
        const next = prev.map(m =>
          m.id === asstId
            ? { ...m, content: `**连接失败** — ${errMsg}\n\n请确保后端已启动 (\`python server.py\`)`, isStreaming: false }
            : m
        )
        updatedMessages = next
        return next
      })
    } finally {
      setBusy(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, mode, currentUser.user_id, selectedWorkspace, selectedProject])

  const clearSession = useCallback(() => {
    setMessages([])
    sessionId.current = 'sess_' + genId()
    sessionCreatedAt.current = Date.now()
    setStats({ calls: 0, totalTokens: 0, toolCalls: 0 })
  }, [])

  const quickPrompts = QUICK_MAP[mode]

  return {
    messages,
    mode,
    setMode,
    busy,
    send,
    clearSession,
    loadSession,
    stats,
    quickPrompts,
    sessionId: sessionId.current,
    /** 当前报表门店 ID（空 = 使用后端默认） */
    braId,
    setBraId,
  }
}
