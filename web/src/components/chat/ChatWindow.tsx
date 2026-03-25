'use client'
import { useEffect, useRef } from 'react'
import type { Message } from '@/types'
import MessageBubble from './MessageBubble'

interface Props {
  messages: Message[]
  busy: boolean
  quickPrompts: string[]
  onQuick: (text: string) => void
}

export default function ChatWindow({ messages, busy, quickPrompts, onQuick }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto py-6" style={{ scrollbarWidth: 'thin' }}>
      {messages.length === 0 ? (
        <Empty quickPrompts={quickPrompts} onQuick={onQuick} />
      ) : (
        <div className="max-w-3xl mx-auto px-4 space-y-4">
          {messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {busy && messages[messages.length - 1]?.role !== 'assistant' && (
            <ThinkingRow />
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  )
}

function ThinkingRow() {
  return (
    <div className="flex items-center gap-2 px-1" style={{ color: 'var(--muted)' }}>
      <div className="flex gap-1">
        {[0, 0.15, 0.3].map((d, i) => (
          <span key={i} className="w-1.5 h-1.5 rounded-full bg-current animate-blink"
            style={{ animationDelay: `${d}s` }} />
        ))}
      </div>
      <span className="text-xs">推理中…</span>
    </div>
  )
}

function Empty({ quickPrompts, onQuick }: { quickPrompts: string[]; onQuick: (t: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 px-6 text-center" style={{ minHeight: '60vh' }}>
      <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl"
        style={{ background: 'var(--elevated)', border: '1px solid var(--border-strong)' }}>
        ✦
      </div>
      <div>
        <div className="text-lg font-semibold mb-1" style={{ color: '#f1f5f9' }}>AI Agent 已就绪</div>
        <div className="text-sm max-w-sm leading-relaxed" style={{ color: 'var(--muted)' }}>
          天气查询 · 报表分析 · 营销方案 · 多轮对话
        </div>
      </div>
      <div className="flex flex-wrap gap-2 justify-center max-w-lg">
        {quickPrompts.map(p => (
          <button key={p} onClick={() => onQuick(p)}
            className="text-xs px-3 py-2 rounded-full transition-all duration-150 hover:scale-105"
            style={{
              background: 'var(--elevated)',
              border: '1px solid var(--border-strong)',
              color: '#cbd5e1',
            }}
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  )
}
