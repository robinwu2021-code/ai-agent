import type { Message } from '@/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { formatTimestamp } from '@/lib/utils'
import ToolCallCards from './ToolCallCard'
import WeatherCard from '../cards/WeatherCard'
import BiReportCard from '../cards/BiReportCard'
import MarketingPlanCard from '../cards/MarketingPlanCard'

interface Props { message: Message }

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[75%]">
          <div className="px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-relaxed"
            style={{ background: '#7c3aed', color: '#fff' }}>
            {message.content}
          </div>
          <div className="text-right mt-1 text-xs" style={{ color: 'var(--muted)' }}>
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-fade-in">
      <ToolCallCards toolCalls={message.toolCalls} />

      {/* Structured data cards */}
      {message.structuredData?.type === 'weather' && (
        <div className="mb-3">
          <WeatherCard data={message.structuredData.data} />
        </div>
      )}
      {message.structuredData?.type === 'bi' && (
        <div className="mb-3">
          <BiReportCard data={message.structuredData.data} />
        </div>
      )}
      {message.structuredData?.type === 'marketing' && (
        <div className="mb-3">
          <MarketingPlanCard data={message.structuredData.data} />
        </div>
      )}

      {/* Text content */}
      {message.content && (
        <div className="rounded-2xl rounded-tl-sm px-4 py-3"
          style={{ background: 'var(--elevated)', border: '1px solid var(--border)' }}>
          <div className="prose text-sm" style={{ color: '#cbd5e1' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
          {message.isStreaming && (
            <span className="inline-block w-0.5 h-4 ml-0.5 bg-purple-400 animate-blink align-middle" />
          )}
        </div>
      )}

      {!message.content && message.isStreaming && !message.toolCalls.length && (
        <div className="flex items-center gap-2 px-1" style={{ color: 'var(--muted)' }}>
          <div className="flex gap-1">
            {[0, 0.15, 0.3].map((d, i) => (
              <span key={i} className="w-1.5 h-1.5 rounded-full bg-current animate-blink"
                style={{ animationDelay: `${d}s` }} />
            ))}
          </div>
        </div>
      )}

      <div className="mt-1 text-xs" style={{ color: 'var(--muted)' }}>
        {formatTimestamp(message.timestamp)}
      </div>
    </div>
  )
}
