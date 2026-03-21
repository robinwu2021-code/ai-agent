import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function genId() {
  return Math.random().toString(36).slice(2, 10)
}

export function formatTimestamp(date: Date): string {
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

export function todayMidnightTs(): number {
  const d = new Date()
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

export function daysMidnightTs(daysAgo: number): number {
  return todayMidnightTs() - daysAgo * 86400000
}
