'use client'
import { useEffect } from 'react'
import { useChat } from '@/hooks/useChat'
import { useApp } from '@/contexts/AppContext'
import Sidebar from '@/components/layout/Sidebar'
import ChatWindow from '@/components/chat/ChatWindow'
import InputBar from '@/components/chat/InputBar'
import Header from '@/components/layout/Header'
import { useState } from 'react'

export default function Home() {
  const chat = useChat()
  const appCtx = useApp()
  const { currentSessionId, sessions, selectedWorkspace, selectedProject } = appCtx
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // When currentSessionId changes, load that session's messages
  useEffect(() => {
    if (!currentSessionId) return
    const session = sessions.find(s => s.session_id === currentSessionId)
    if (session) {
      chat.loadSession(session)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId])

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>
      <Sidebar
        mode={chat.mode}
        setMode={chat.setMode}
        stats={chat.stats}
        sessionId={chat.sessionId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onClear={chat.clearSession}
      />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Header
          mode={chat.mode}
          busy={chat.busy}
          onToggleSidebar={() => setSidebarOpen(o => !o)}
        />

        {/* Project context bar */}
        {selectedProject && (
          <div
            className="px-4 py-1.5 text-xs flex items-center gap-2 border-b flex-shrink-0"
            style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
          >
            <span style={{ color: 'var(--muted)' }}>当前项目</span>
            <span className="text-purple-400 font-medium">
              {selectedWorkspace?.name ?? '—'}
            </span>
            <span style={{ color: 'var(--muted)' }}>/</span>
            <span className="text-purple-300 font-medium">{selectedProject.name}</span>
          </div>
        )}

        <ChatWindow
          messages={chat.messages}
          busy={chat.busy}
          quickPrompts={chat.quickPrompts}
          onQuick={chat.send}
        />
        <InputBar
          onSend={chat.send}
          busy={chat.busy}
          mode={chat.mode}
          braId={chat.braId}
          onBraIdChange={chat.setBraId}
        />
      </div>
    </div>
  )
}
