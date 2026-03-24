export type SkillMode = 'general' | 'weather' | 'report' | 'marketing' | 'graph'

export type MessageRole = 'user' | 'assistant'

export interface ToolCall {
  id: string
  name: string
  status: 'running' | 'done' | 'error'
  error?: string
}

/** DAG 中单个步骤 */
export interface DagStep {
  id: string
  goal: string
  status: 'pending' | 'running' | 'done' | 'error'
  result?: string
  error?: string
  /** 同批并行执行的兄弟步骤 ids */
  parallelGroup?: string[]
}

/** 多 Agent 中单个子任务 */
export interface SubTaskInfo {
  id: string
  agent: string
  goal: string
  depends: string[]
  status: 'pending' | 'running' | 'done' | 'error'
  tokens?: number
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
  /** 未经标签加工的原始指标数组（与 metrics 一一对应） */
  raw_metrics?: Array<Record<string, number | string>>
  aggregation?: string
  bra_id?: string
  range_start?: number
  range_end?: number
}

/** 报表图表视图类型 */
export type BiChartTab = 'overview' | 'table' | 'line' | 'pie' | 'comparison'

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
  /** DAG 步骤列表（orchestrator_type=dag 时填充） */
  dagSteps?: DagStep[]
  /** 多 Agent 子任务列表（orchestrator_type=multiagent 时填充） */
  subTasks?: SubTaskInfo[]
}

export interface ChatRequest {
  text: string
  user_id: string
  session_id?: string
  max_steps?: number
  skills?: string[]
  mode?: string
  /** Store ID for BI report queries; backend falls back to AGENT_BI_DEFAULT_BRA_ID if omitted */
  bra_id?: string
  workspace_id?: string
  project_id?: string
  orchestrator_type?: 'react' | 'plan_execute' | 'dag' | 'multiagent'
  agent_specs?: Array<{
    name: string
    description: string
    skills: string[]
    system_prompt: string
    max_steps: number
  }>
}

export type SSEEvent =
  | { type: 'thinking' }
  | { type: 'step'; tool: string; status: 'running' | 'done' | 'error'; step: number; error?: string }
  // DAG events
  | { type: 'planning' }
  | { type: 'plan'; steps: Array<{ id: string; goal: string; depends?: string[] }> }
  | { type: 'parallel_start'; step_ids: string[]; goals: string[] }
  | { type: 'step_start'; step_id: string; step: number; goal: string }
  | { type: 'step_done'; step_id: string; result: string }
  | { type: 'step_failed'; step_id: string; error: string }
  // MultiAgent events
  | { type: 'orchestrating'; agents: string[] }
  | { type: 'subtask_assign'; agent: string; goal: string; depends: string[]; subtask_id: string }
  | { type: 'agent_start'; agent: string; subtask_id: string }
  | { type: 'agent_done'; agent: string; subtask_id: string; tokens: number }
  | { type: 'agent_error'; agent: string; subtask_id: string; error: string }
  | { type: 'agent_event'; agent: string; event: SSEEvent }
  // Common
  | { type: 'delta'; text: string }
  | { type: 'done'; usage?: { total_tokens: number } }
  | { type: 'error'; message: string }
  | { type: 'session'; session_id: string; workspace_id?: string; project_id?: string }

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
