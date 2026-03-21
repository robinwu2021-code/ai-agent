export type SkillMode = 'general' | 'weather' | 'report' | 'marketing'

export type MessageRole = 'user' | 'assistant'

export interface ToolCall {
  id: string
  name: string
  status: 'running' | 'done' | 'error'
  error?: string
}

export interface WeatherData {
  city?: string
  temperature?: number
  feels_like?: number
  weather_condition?: string
  humidity?: number
  wind_speed?: number
  description?: string
  forecast?: Array<{
    date: string
    high?: number
    low?: number
    condition?: string
  }>
}

export interface BiMetricRow {
  [key: string]: { value: number | string; label: string } | number | string
}

export interface BiData {
  metrics: BiMetricRow[]
  aggregation?: string
  bra_id?: string
  range_start?: number
  range_end?: number
}

export interface MarketingActivity {
  name: string
  type?: string
  channels?: string[]
  description?: string
  timeline?: string
  budget?: string
  expected_outcome?: string
}

export interface MarketingData {
  plan_name?: string
  merchant?: string
  goal?: string
  activities?: MarketingActivity[]
  summary?: string
  next_steps?: string[]
}

export type StructuredData =
  | { type: 'weather'; data: WeatherData }
  | { type: 'bi'; data: BiData }
  | { type: 'marketing'; data: MarketingData }

export interface Message {
  id: string
  role: MessageRole
  content: string
  timestamp: Date
  toolCalls: ToolCall[]
  isStreaming?: boolean
  structuredData?: StructuredData | null
}

export interface ChatRequest {
  text: string
  user_id: string
  session_id?: string
  max_steps?: number
  skills?: string[]
  mode?: string
}

export type SSEEvent =
  | { type: 'thinking' }
  | { type: 'step'; tool: string; status: 'running' | 'done' | 'error'; step: number; error?: string }
  | { type: 'planning' }
  | { type: 'plan'; steps: { goal: string }[] }
  | { type: 'step_start'; step: number; goal: string }
  | { type: 'step_done'; step: number }
  | { type: 'delta'; text: string }
  | { type: 'done'; usage?: { total_tokens: number } }
  | { type: 'error'; message: string }

export interface SkillInfo {
  name: string
  description: string
  source: 'builtin' | 'hub'
}

export interface SessionStats {
  calls: number
  totalTokens: number
  toolCalls: number
}
