import type { SSEEvent, ChatRequest, SkillInfo } from '@/types'

const BASE = '/api/agent'

export async function fetchHealth() {
  const r = await fetch(`${BASE}/health`)
  if (!r.ok) throw new Error('unhealthy')
  return r.json() as Promise<{ status: string; skills?: number }>
}

export async function fetchSkills(): Promise<SkillInfo[]> {
  const r = await fetch(`${BASE}/skills`)
  if (!r.ok) throw new Error('Failed to fetch skills')
  return r.json()
}

export async function* streamChat(req: ChatRequest): AsyncGenerator<SSEEvent> {
  const r = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!r.ok) {
    yield { type: 'error', message: `HTTP ${r.status}: ${r.statusText}` }
    return
  }
  const reader = r.body!.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      const raw = line.slice(5).trim()
      if (raw === '[DONE]') return
      try { yield JSON.parse(raw) as SSEEvent } catch { /* ignore */ }
    }
  }
}
