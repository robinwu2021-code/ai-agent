'use client'
import { useChat } from '@/hooks/useChat'
import Sidebar from '@/components/layout/Sidebar'
import ChatWindow from '@/components/chat/ChatWindow'
import InputBar from '@/components/chat/InputBar'
import Header from '@/components/layout/Header'
import { useState } from 'react'

export default function Home() {
  const chat = useChat()
  const [sidebarOpen, setSidebarOpen] = useState(true)

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
        <ChatWindow
          messages={chat.messages}
          busy={chat.busy}
          quickPrompts={chat.quickPrompts}
          onQuick={chat.send}
        />
        <InputBar onSend={chat.send} busy={chat.busy} />
      </div>
    </div>
  )
}
