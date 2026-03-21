'use client'
import { useState, useCallback, useRef } from 'react'
import type { Message, SkillMode, SessionStats, ToolCall, DagStep, SubTaskInfo } from '@/types'
import { streamChat } from '@/lib/api'
import { genId, daysMidnightTs, todayMidnightTs } from '@/lib/utils'
import { parseStructuredData } from '@/lib/parseStructuredData'

const SKILL_MAP: Record<SkillMode, string[] | null> = {
  general: null,
  weather: ['weather_current'],
  report: ['agent_bi'],
  marketing: ['marketing_advisor'],
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
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [mode, setMode] = useState<SkillMode>('general')
  const [busy, setBusy] = useState(false)
  const [stats, setStats] = useState<SessionStats>({ calls: 0, totalTokens: 0, toolCalls: 0 })
  const sessionId = useRef<string>('sess_' + genId())
  const userId = useRef<string>('user_' + genId())

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
    setMessages(prev => [...prev, userMsg])

    const asstId = genId()
    const asstMsg: Message = {
      id: asstId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      toolCalls: [],
      isStreaming: true,
    }
    setMessages(prev => [...prev, asstMsg])

    // Build request — inject date context for BI queries
    let enrichedText = text
    if (mode === 'report') {
      const today = new Date()
      const dateCtx = `（当前日期：${today.toLocaleDateString('zh-CN')}，今日凌晨时间戳：${todayMidnightTs()}，昨日凌晨时间戳：${daysMidnightTs(1)}，7天前凌晨时间戳：${daysMidnightTs(7)}）`
      enrichedText = text + '\n' + dateCtx
    }

    const req = {
      text: enrichedText,
      user_id: userId.current,
      session_id: sessionId.current,
      max_steps: 15,
      skills: SKILL_MAP[mode] ?? undefined,
      mode: mode === 'marketing' ? 'marketing' : undefined,
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
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, content } : m
          ))
        } else if (ev.type === 'step') {
          const tcId = `tc-${ev.step}-${ev.tool}`
          const existing = toolMap.get(tcId)
          const tc: ToolCall = existing
            ? { ...existing, status: ev.status as ToolCall['status'], error: ev.error }
            : { id: tcId, name: ev.tool, status: ev.status as ToolCall['status'], error: ev.error }
          toolMap.set(tcId, tc)
          if (ev.status !== 'running') toolCallsCount++
          setMessages(prev => prev.map(m =>
            m.id === asstId
              ? { ...m, toolCalls: Array.from(toolMap.values()) }
              : m
          ))

        // ── DAG events ────────────────────────────────────────────
        } else if (ev.type === 'plan') {
          for (const s of ev.steps) {
            dagStepMap.set(s.id, { id: s.id, goal: s.goal, status: 'pending' })
          }
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
          ))
        } else if (ev.type === 'parallel_start') {
          for (const sid of ev.step_ids) {
            const s = dagStepMap.get(sid)
            if (s) dagStepMap.set(sid, { ...s, status: 'running', parallelGroup: ev.step_ids })
          }
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
          ))
        } else if (ev.type === 'step_start') {
          const s = dagStepMap.get(ev.step_id)
          if (s) dagStepMap.set(ev.step_id, { ...s, status: 'running' })
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
          ))
        } else if (ev.type === 'step_done') {
          const s = dagStepMap.get(ev.step_id)
          if (s) dagStepMap.set(ev.step_id, { ...s, status: 'done', result: ev.result })
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
          ))
        } else if (ev.type === 'step_failed') {
          const s = dagStepMap.get(ev.step_id)
          if (s) dagStepMap.set(ev.step_id, { ...s, status: 'error', error: ev.error })
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, dagSteps: Array.from(dagStepMap.values()) } : m
          ))

        // ── MultiAgent events ─────────────────────────────────────
        } else if (ev.type === 'subtask_assign') {
          subTaskMap.set(ev.subtask_id, {
            id: ev.subtask_id, agent: ev.agent, goal: ev.goal,
            depends: ev.depends, status: 'pending',
          })
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
          ))
        } else if (ev.type === 'agent_start') {
          const st = subTaskMap.get(ev.subtask_id)
          if (st) subTaskMap.set(ev.subtask_id, { ...st, status: 'running' })
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
          ))
        } else if (ev.type === 'agent_done') {
          const st = subTaskMap.get(ev.subtask_id)
          if (st) subTaskMap.set(ev.subtask_id, { ...st, status: 'done', tokens: ev.tokens })
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
          ))
        } else if (ev.type === 'agent_error') {
          const st = subTaskMap.get(ev.subtask_id)
          if (st) subTaskMap.set(ev.subtask_id, { ...st, status: 'error', error: ev.error })
          setMessages(prev => prev.map(m =>
            m.id === asstId ? { ...m, subTasks: Array.from(subTaskMap.values()) } : m
          ))

        } else if (ev.type === 'done') {
          const structured = parseStructuredData(content)
          setStats(s => ({
            calls: s.calls + 1,
            totalTokens: s.totalTokens + (ev.usage?.total_tokens ?? 0),
            toolCalls: s.toolCalls + toolCallsCount,
          }))
          setMessages(prev => prev.map(m =>
            m.id === asstId
              ? { ...m, isStreaming: false, structuredData: structured }
              : m
          ))
        } else if (ev.type === 'error') {
          setMessages(prev => prev.map(m =>
            m.id === asstId
              ? { ...m, content: `⚠ ${ev.message}`, isStreaming: false }
              : m
          ))
        }
      }
      // ensure streaming stops
      setMessages(prev => prev.map(m =>
        m.id === asstId ? { ...m, isStreaming: false } : m
      ))
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : '未知错误'
      setMessages(prev => prev.map(m =>
        m.id === asstId
          ? { ...m, content: `**连接失败** — ${errMsg}\n\n请确保后端已启动 (\`python server.py\`)`, isStreaming: false }
          : m
      ))
    } finally {
      setBusy(false)
    }
  }, [busy, mode])

  const clearSession = useCallback(() => {
    setMessages([])
    sessionId.current = 'sess_' + genId()
    setStats({ calls: 0, totalTokens: 0, toolCalls: 0 })
  }, [])

  const quickPrompts = QUICK_MAP[mode]

  return { messages, mode, setMode, busy, send, clearSession, stats, quickPrompts, sessionId: sessionId.current }
}
