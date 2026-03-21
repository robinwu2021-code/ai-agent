import type { StructuredData, WeatherData, BiData, MarketingData } from '@/types'

/** Try to extract the first JSON block from markdown text */
function extractJson(text: string): unknown | null {
  const codeBlock = text.match(/```(?:json)?\s*\n([\s\S]*?)\n```/)
  if (codeBlock) {
    try { return JSON.parse(codeBlock[1]) } catch { /* ignore */ }
  }
  // Try bare JSON objects
  const bare = text.match(/(\{[\s\S]*\})/)
  if (bare) {
    try { return JSON.parse(bare[1]) } catch { /* ignore */ }
  }
  return null
}

function isWeather(obj: unknown): obj is WeatherData {
  if (!obj || typeof obj !== 'object') return false
  const o = obj as Record<string, unknown>
  return 'temperature' in o || 'weather_condition' in o || 'weather' in o
}

function isBi(obj: unknown): obj is BiData {
  if (!obj || typeof obj !== 'object') return false
  const o = obj as Record<string, unknown>
  return 'metrics' in o && Array.isArray(o.metrics)
}

function isMarketing(obj: unknown): obj is MarketingData {
  if (!obj || typeof obj !== 'object') return false
  const o = obj as Record<string, unknown>
  return 'activities' in o || 'plan_name' in o
}

export function parseStructuredData(content: string): StructuredData | null {
  const obj = extractJson(content)
  if (!obj) return null
  if (isWeather(obj)) return { type: 'weather', data: obj }
  if (isBi(obj)) return { type: 'bi', data: obj }
  if (isMarketing(obj)) return { type: 'marketing', data: obj }
  return null
}
