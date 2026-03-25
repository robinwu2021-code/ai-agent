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

export function tomorrowMidnightTs(): number {
  return todayMidnightTs() + 86400000
}

/** Monday of the current week at midnight */
export function thisWeekStartTs(): number {
  const d = new Date()
  d.setHours(0, 0, 0, 0)
  const day = d.getDay() === 0 ? 6 : d.getDay() - 1  // Mon=0 … Sun=6
  d.setDate(d.getDate() - day)
  return d.getTime()
}

/** Monday of next week — serves as this week's range end */
export function nextWeekStartTs(): number {
  return thisWeekStartTs() + 7 * 86400000
}

/** Monday of last week */
export function lastWeekStartTs(): number {
  return thisWeekStartTs() - 7 * 86400000
}

/** First day of current month at midnight */
export function thisMonthStartTs(): number {
  const d = new Date()
  d.setDate(1)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

/** First day of next month — serves as this month's range end */
export function nextMonthStartTs(): number {
  const d = new Date()
  d.setDate(1)
  d.setHours(0, 0, 0, 0)
  d.setMonth(d.getMonth() + 1)
  return d.getTime()
}
