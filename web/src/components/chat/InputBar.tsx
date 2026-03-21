'use client'
import { useRef, useState, useCallback } from 'react'
import { cn } from '@/lib/utils'

interface Props {
  onSend: (text: string) => void
  busy: boolean
}

export default function InputBar({ onSend, busy }: Props) {
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

  return (
    <div className="flex-shrink-0 px-4 py-3 border-t" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
      <div className="max-w-3xl mx-auto">
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
        <div className="flex items-center justify-between mt-1.5 px-1">
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
