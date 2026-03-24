'use client'
import { useRef, useState, useCallback } from 'react'
import { cn } from '@/lib/utils'
import type { SkillMode } from '@/types'

interface Props {
  onSend: (text: string) => void
  busy: boolean
  /** 当前模式，report 模式下显示门店 ID 输入 */
  mode?: SkillMode
  /** 当前门店 ID */
  braId?: string
  /** 门店 ID 变更回调 */
  onBraIdChange?: (v: string) => void
}

export default function InputBar({ onSend, busy, mode, braId = '', onBraIdChange }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const resize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [])

  const send = useCallback(() => {
    const text = value.trim()
    if (!text || busy) return
    onSend(text)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }, [value, busy, onSend])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const canSend = value.trim().length > 0 && !busy
  const isReport = mode === 'report'

  return (
    <div className="flex-shrink-0 px-4 py-3 border-t" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
      <div className="max-w-3xl mx-auto space-y-2">

        {/* ── 报表模式：门店 ID 输入行 ──────────────────────── */}
        {isReport && (
          <div className="flex items-center gap-2 px-1">
            <span className="text-xs flex-shrink-0" style={{ color: 'var(--muted)' }}>
              门店 ID
            </span>
            <div className="relative flex-1 max-w-[240px]">
              <input
                type="text"
                value={braId}
                onChange={e => onBraIdChange?.(e.target.value)}
                placeholder="留空使用默认门店"
                className="w-full text-xs rounded-lg px-3 py-1.5 outline-none transition-colors"
                style={{
                  background: 'var(--elevated)',
                  border: '1px solid var(--border-strong)',
                  color: braId ? '#f1f5f9' : 'var(--muted)',
                }}
              />
              {braId && (
                <button
                  onClick={() => onBraIdChange?.('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-xs"
                  style={{ color: 'var(--muted)' }}
                  title="清空门店ID"
                >
                  ✕
                </button>
              )}
            </div>
            {braId ? (
              <span className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: 'rgba(52,211,153,0.1)', color: '#34d399', border: '1px solid rgba(52,211,153,0.25)' }}>
                {braId}
              </span>
            ) : (
              <span className="text-xs" style={{ color: 'var(--muted)', fontStyle: 'italic' }}>
                使用配置默认
              </span>
            )}
          </div>
        )}

        {/* ── 主输入框 ──────────────────────────────────────── */}
        <div className="relative rounded-xl overflow-hidden"
          style={{ background: 'var(--elevated)', border: '1px solid var(--border-strong)' }}>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={e => { setValue(e.target.value); resize() }}
            onKeyDown={onKeyDown}
            placeholder="输入消息… Enter 发送，Shift+Enter 换行"
            rows={1}
            className="w-full bg-transparent resize-none outline-none px-4 py-3 pr-14 text-sm leading-relaxed"
            style={{ color: 'var(--text)', minHeight: '46px', maxHeight: '160px' }}
          />
          <button
            onClick={send}
            disabled={!canSend}
            className={cn(
              'absolute right-2.5 bottom-2.5 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150',
              canSend ? 'opacity-100 hover:scale-105' : 'opacity-30 cursor-not-allowed'
            )}
            style={{ background: canSend ? '#7c3aed' : 'var(--border-strong)' }}
          >
            {busy ? (
              <div className="w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
              </svg>
            )}
          </button>
        </div>

        <div className="flex items-center justify-between px-1">
          <div className="text-xs" style={{ color: 'var(--muted)' }}>
            Enter 发送 · Shift+Enter 换行
          </div>
          <div className="text-xs font-mono" style={{ color: 'var(--muted)' }}>
            {value.length > 0 ? `${value.length}` : ''}
          </div>
        </div>
      </div>
    </div>
  )
}
