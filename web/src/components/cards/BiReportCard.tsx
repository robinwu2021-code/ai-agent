'use client'
import { useState } from 'react'
import type { BiData, BiChartTab, BiDelta } from '@/types'
import {
  BarChart, Bar,
  LineChart, Line,
  PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

// ── 常量 ──────────────────────────────────────────────────────────────────────

const METRIC_LABELS: Record<string, string> = {
  turnover:              '总销售额',
  actual_income:         '实际收入',
  actual_income_no_tips: '实际收入(不含小费)',
  tips_amount:           '小费总额',
  order_quantity:        '订单数',
  average_order_price:   '均单价',
  customer_number:       '顾客数',
  dining_room_amount:    '堂食',
  togo_amount:           '外带',
  store_delivery_amount: '配送',
  discount_amount:       '折扣额',
  discount_ratio:        '折扣比率',
  member_quantity:       '会员订单',
  new_member_number:     '新会员',
  first_member_number:   '首次消费',
  second_member_number:  '回头客',
  regular_member_number: '常客',
  refund_amount:         '退款额',
  refunded_quantity:     '退款单数',
  pay_by_cash:           '现金',
  pay_by_card:           '卡支付',
  pay_by_online:         '在线支付',
  pay_by_savings:        '储值支付',
  pos_amount:            'POS卡',
  amount_not_taxed:      '不含税额',
  vat:                   '增值税',
}

const MONEY_FIELDS = new Set([
  'turnover', 'actual_income', 'actual_income_no_tips', 'tips_amount',
  'average_order_price', 'dining_room_amount', 'togo_amount',
  'store_delivery_amount', 'discount_amount', 'refund_amount',
  'pay_by_cash', 'pay_by_card', 'pay_by_online', 'pay_by_savings',
  'pos_amount', 'amount_not_taxed', 'vat',
])

const PAYMENT_KEYS    = ['pay_by_cash', 'pay_by_card', 'pay_by_online', 'pay_by_savings', 'pos_amount']
const ORDER_TYPE_KEYS = ['dining_room_amount', 'togo_amount', 'store_delivery_amount']

const CHART_COLORS = [
  '#34d399', '#7c3aed', '#f59e0b', '#3b82f6',
  '#ec4899', '#10b981', '#f97316', '#6366f1',
  '#14b8a6', '#a78bfa',
]

const TOOLTIP_STYLE = {
  background: '#1a1a2e',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '8px',
  fontSize: '11px',
}

// ── 工具函数 ──────────────────────────────────────────────────────────────────

function getVal(v: unknown): number {
  if (typeof v === 'object' && v !== null && 'value' in v)
    return Number((v as { value: unknown }).value) || 0
  return Number(v) || 0
}

function getLabel(key: string, v: unknown): string {
  if (typeof v === 'object' && v !== null && 'label' in v)
    return String((v as { label: unknown }).label)
  return METRIC_LABELS[key] ?? key
}

function fmtMoney(n: number): string {
  return `¥${n.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function fmtVal(key: string, v: unknown): string {
  if (v == null) return '—'
  const n = getVal(v)
  if (isNaN(n)) return String(v)
  if (MONEY_FIELDS.has(key)) return fmtMoney(n)
  if (key === 'discount_ratio') return `${(n * 100).toFixed(1)}%`
  return n.toLocaleString('zh-CN')
}

function fmtDeltaPct(d: BiDelta | undefined): string {
  if (!d || d.delta_pct == null) return '—'
  const sign = d.delta_pct >= 0 ? '+' : ''
  return `${sign}${d.delta_pct.toFixed(1)}%`
}

function deltaColor(d: BiDelta | undefined): string {
  if (!d || d.delta_pct == null) return 'var(--muted)'
  return d.delta_pct >= 0 ? '#34d399' : '#f87171'
}

function tsToDate(ms?: number): string {
  if (!ms) return '—'
  return new Date(ms).toLocaleDateString('zh-CN', { month: 'long', day: 'numeric' })
}

function tsToDateShort(ms?: number): string {
  if (!ms) return '—'
  return new Date(ms).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}

// ── 概览 Tab ──────────────────────────────────────────────────────────────────

interface OverviewTabProps {
  rows: BiData['metrics']
  allKeys: string[]
  deltas?: BiData['deltas']
}

function OverviewTab({ rows, allKeys, deltas }: OverviewTabProps) {
  const row = rows[0] ?? {}
  const topKeys = ['turnover', 'order_quantity', 'customer_number', 'average_order_price']
    .filter(k => allKeys.includes(k))
  const restKeys = allKeys.filter(k => !topKeys.includes(k))

  return (
    <div className="space-y-3">
      {topKeys.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {topKeys.map(key => {
            const d = deltas?.[key]
            return (
              <div key={key} className="rounded-xl p-3"
                style={{ background: 'rgba(52,211,153,0.07)', border: '1px solid rgba(52,211,153,0.2)' }}>
                <div className="text-xs mb-1" style={{ color: '#34d399' }}>
                  {getLabel(key, row[key])}
                </div>
                <div className="text-base font-bold" style={{ color: '#f1f5f9' }}>
                  {fmtVal(key, row[key])}
                </div>
                {d && (
                  <div className="text-xs mt-1 font-medium" style={{ color: deltaColor(d) }}>
                    {d.delta_pct != null && (d.delta_pct >= 0 ? '▲ ' : '▼ ')}{fmtDeltaPct(d)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
      {restKeys.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {restKeys.map(key => {
            const d = deltas?.[key]
            return (
              <div key={key} className="rounded-xl p-3"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
                <div className="text-xs mb-1 truncate" style={{ color: 'var(--muted)' }}>
                  {getLabel(key, row[key])}
                </div>
                <div className="font-semibold text-sm truncate" style={{ color: '#f1f5f9' }}>
                  {fmtVal(key, row[key])}
                </div>
                {d && (
                  <div className="text-xs mt-0.5" style={{ color: deltaColor(d) }}>
                    {fmtDeltaPct(d)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── 表格 Tab ──────────────────────────────────────────────────────────────────

interface TableTabProps {
  rows: BiData['metrics']
  allKeys: string[]
  prevRows?: BiData['previous_metrics']
  deltas?: BiData['deltas']
}

function TableTab({ rows, allKeys, prevRows, deltas }: TableTabProps) {
  const hasPrev = !!prevRows?.length
  return (
    <div className="overflow-x-auto rounded-xl" style={{ border: '1px solid var(--border)' }}>
      <table className="w-full text-xs">
        <thead>
          <tr style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid var(--border)' }}>
            <th className="text-left px-3 py-2.5 font-medium" style={{ color: 'var(--muted)' }}>指标</th>
            {rows.map((_, i) => (
              <th key={i} className="text-right px-3 py-2.5 font-medium" style={{ color: 'var(--muted)' }}>
                {rows.length > 1 ? `当期 ${i + 1}` : '当期值'}
              </th>
            ))}
            {hasPrev && (
              <th className="text-right px-3 py-2.5 font-medium" style={{ color: 'var(--muted)' }}>对比期</th>
            )}
            {hasPrev && (
              <th className="text-right px-3 py-2.5 font-medium" style={{ color: 'var(--muted)' }}>变化率</th>
            )}
          </tr>
        </thead>
        <tbody>
          {allKeys.map((key, ri) => {
            const d = deltas?.[key]
            const prevRow = prevRows?.[0]
            return (
              <tr key={key}
                style={{
                  borderBottom: ri < allKeys.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                  background: ri % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)',
                }}>
                <td className="px-3 py-2" style={{ color: 'var(--muted)' }}>
                  {METRIC_LABELS[key] ?? key}
                </td>
                {rows.map((row, ci) => (
                  <td key={ci} className="px-3 py-2 text-right font-medium tabular-nums"
                    style={{ color: '#f1f5f9' }}>
                    {fmtVal(key, row[key])}
                  </td>
                ))}
                {hasPrev && (
                  <td className="px-3 py-2 text-right tabular-nums" style={{ color: 'var(--muted)' }}>
                    {prevRow ? fmtVal(key, prevRow[key]) : '—'}
                  </td>
                )}
                {hasPrev && (
                  <td className="px-3 py-2 text-right font-medium tabular-nums"
                    style={{ color: deltaColor(d) }}>
                    {d ? (d.delta_pct != null && (d.delta_pct >= 0 ? '▲ ' : '▼ ')) + fmtDeltaPct(d) : '—'}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── 折线/趋势 Tab ─────────────────────────────────────────────────────────────

interface LineTabProps {
  rows: BiData['metrics']
  allKeys: string[]
  prevRows?: BiData['previous_metrics']
  rangeStart?: number
  rangeEnd?: number
  prevRangeStart?: number
  comparisonType?: 'yoy' | 'pop'
}

function LineTab({ rows, allKeys, prevRows, rangeStart, prevRangeStart, comparisonType }: LineTabProps) {
  const numericKeys = allKeys.filter(k => {
    const total = rows.reduce((s, r) => s + getVal(r[k]), 0)
    return total > 0
  }).slice(0, 5)

  const hasPrev = !!prevRows?.length

  // Comparison line chart: current vs previous period, one line per metric
  if (hasPrev && rows.length <= 1) {
    const periodLabel  = rangeStart    ? tsToDateShort(rangeStart) : '当期'
    const prevLabel    = prevRangeStart ? tsToDateShort(prevRangeStart) : '对比期'
    const compLabel    = comparisonType === 'yoy' ? '同比' : comparisonType === 'pop' ? '环比' : '对比'
    const row          = rows[0] ?? {}
    const prevRow      = prevRows![0] ?? {}

    const chartData = numericKeys.map(k => ({
      name:   METRIC_LABELS[k] ?? k,
      当期:   getVal(row[k]),
      对比期: getVal(prevRow[k]),
      key: k,
    }))

    return (
      <div>
        <p className="text-xs mb-3" style={{ color: 'var(--muted)' }}>
          {compLabel}对比：<span style={{ color: '#34d399' }}>{periodLabel}</span>
          {' vs '}<span style={{ color: '#7c3aed' }}>{prevLabel}</span>
        </p>
        <ResponsiveContainer width="100%" height={Math.max(200, numericKeys.length * 42)}>
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 40, left: 8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--muted)' }}
              tickFormatter={v => MONEY_FIELDS.has(chartData[0]?.key ?? '') ? `¥${(v / 1000).toFixed(0)}k` : String(v)} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: 'var(--muted)' }} width={64} />
            <Tooltip contentStyle={TOOLTIP_STYLE}
              formatter={(v: number, name: string, entry) => {
                const k = (entry.payload as { key?: string })?.key ?? ''
                return [MONEY_FIELDS.has(k) ? fmtMoney(v) : v.toLocaleString('zh-CN'), name]
              }} />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
            <Bar dataKey="当期"   fill="#34d399" fillOpacity={0.85} radius={[0, 4, 4, 0]} />
            <Bar dataKey="对比期" fill="#7c3aed" fillOpacity={0.6}  radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    )
  }

  // Multi-row: line chart (single period trend)
  if (rows.length > 1) {
    const lineData = rows.map((row, i) => {
      // Use period_label if present (injected by LLM for multi-period queries)
      const label = row['period_label'] != null ? String(row['period_label']) : `#${i + 1}`
      const pt: Record<string, number | string> = { name: label }
      numericKeys.forEach(k => { pt[k] = getVal(row[k]) })
      return pt
    })
    return (
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={lineData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
          <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
          {numericKeys.map((k, i) => (
            <Line key={k} type="monotone" dataKey={k}
              name={METRIC_LABELS[k] ?? k}
              stroke={CHART_COLORS[i % CHART_COLORS.length]}
              strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    )
  }

  // Single-row, no comparison: horizontal bar chart
  const chartData = numericKeys.map(k => ({
    name: METRIC_LABELS[k] ?? k,
    value: getVal(rows[0]?.[k] ?? 0),
    key: k,
  }))
  return (
    <div>
      <p className="text-xs mb-3" style={{ color: 'var(--muted)' }}>
        单期数据以柱状图展示；多期或含对比数据时自动切换为对比图/折线图
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: 'var(--muted)' }} width={72} />
          <Tooltip contentStyle={TOOLTIP_STYLE}
            formatter={(v: number, _: string, entry: { payload?: { key?: string } }) =>
              [MONEY_FIELDS.has(entry?.payload?.key ?? '') ? fmtMoney(v) : v.toLocaleString('zh-CN'), '']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} fill="#34d399" fillOpacity={0.85} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── 饼图 Tab ──────────────────────────────────────────────────────────────────

type PieGroupKey = 'payment' | 'order_type'

interface PieTabProps {
  rows: BiData['metrics']
  prevRows?: BiData['previous_metrics']
  comparisonType?: 'yoy' | 'pop'
}

function PieTab({ rows, prevRows, comparisonType }: PieTabProps) {
  const [group, setGroup] = useState<PieGroupKey>('payment')
  const [showPrev, setShowPrev] = useState(false)
  const hasPrev = !!prevRows?.length

  const activeRows = showPrev ? (prevRows ?? []) : rows
  const row   = activeRows[0] ?? {}
  const keys  = group === 'payment' ? PAYMENT_KEYS : ORDER_TYPE_KEYS
  const pieData = keys
    .map(k => ({ name: METRIC_LABELS[k] ?? k, value: getVal(row[k]), key: k }))
    .filter(d => d.value > 0)
  const total = pieData.reduce((s, d) => s + d.value, 0)

  if (!pieData.length) {
    return (
      <p className="text-center text-sm py-8" style={{ color: 'var(--muted)' }}>
        当前数据中没有{group === 'payment' ? '支付方式' : '订单类型'}分布数据
      </p>
    )
  }

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-3">
        {([['payment', '支付方式'], ['order_type', '订单类型']] as [PieGroupKey, string][]).map(([g, label]) => (
          <button key={g} onClick={() => setGroup(g)}
            className="text-xs px-3 py-1 rounded-full transition-colors"
            style={{
              background: group === g ? 'rgba(52,211,153,0.15)' : 'rgba(255,255,255,0.05)',
              color: group === g ? '#34d399' : 'var(--muted)',
              border: `1px solid ${group === g ? 'rgba(52,211,153,0.35)' : 'var(--border)'}`,
            }}>
            {label}
          </button>
        ))}
        {hasPrev && (
          <button onClick={() => setShowPrev(v => !v)}
            className="text-xs px-3 py-1 rounded-full transition-colors ml-auto"
            style={{
              background: showPrev ? 'rgba(124,58,237,0.15)' : 'rgba(255,255,255,0.05)',
              color: showPrev ? '#a78bfa' : 'var(--muted)',
              border: `1px solid ${showPrev ? 'rgba(124,58,237,0.35)' : 'var(--border)'}`,
            }}>
            {showPrev ? (comparisonType === 'yoy' ? '去年同期' : '上一时段') : '当期'}
          </button>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div style={{ flexShrink: 0 }}>
          <PieChart width={160} height={160}>
            <Pie data={pieData} cx={75} cy={75} innerRadius={40} outerRadius={70}
              dataKey="value" paddingAngle={2}>
              {pieData.map((_, i) => (
                <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.9} />
              ))}
            </Pie>
            <Tooltip contentStyle={TOOLTIP_STYLE}
              formatter={(v: number) => [fmtMoney(v), '']} />
          </PieChart>
        </div>
        <div className="flex-1 space-y-2 min-w-0">
          {pieData.map((d, i) => {
            const pct = total > 0 ? ((d.value / total) * 100).toFixed(1) : '0'
            return (
              <div key={d.key} className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ background: CHART_COLORS[i % CHART_COLORS.length] }} />
                  <span className="text-xs truncate" style={{ color: 'var(--muted)' }}>{d.name}</span>
                </div>
                <div className="text-right flex-shrink-0">
                  <span className="text-xs font-medium tabular-nums" style={{ color: '#f1f5f9' }}>
                    {pct}%
                  </span>
                  <div className="text-xs tabular-nums" style={{ color: 'var(--muted)', fontSize: '10px' }}>
                    {fmtMoney(d.value)}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ── 指标对比 Tab ──────────────────────────────────────────────────────────────

interface ComparisonTabProps { rows: BiData['metrics']; allKeys: string[] }

function ComparisonTab({ rows, allKeys }: ComparisonTabProps) {
  const [showMoney, setShowMoney] = useState(true)
  const row = rows[0] ?? {}

  const filteredKeys = allKeys.filter(k => {
    const v = getVal(row[k])
    if (v === 0) return false
    return showMoney ? MONEY_FIELDS.has(k) : !MONEY_FIELDS.has(k)
  })

  const chartData = filteredKeys.map(k => ({
    name: METRIC_LABELS[k] ?? k,
    value: getVal(row[k]),
    key: k,
  })).sort((a, b) => b.value - a.value).slice(0, 10)

  return (
    <div>
      <div className="flex gap-2 mb-3">
        {([true, false] as const).map(isMoney => (
          <button key={String(isMoney)} onClick={() => setShowMoney(isMoney)}
            className="text-xs px-3 py-1 rounded-full transition-colors"
            style={{
              background: showMoney === isMoney ? 'rgba(124,58,237,0.15)' : 'rgba(255,255,255,0.05)',
              color: showMoney === isMoney ? '#a78bfa' : 'var(--muted)',
              border: `1px solid ${showMoney === isMoney ? 'rgba(124,58,237,0.35)' : 'var(--border)'}`,
            }}>
            {isMoney ? '金额指标' : '数量指标'}
          </button>
        ))}
      </div>

      {chartData.length === 0 ? (
        <p className="text-center text-sm py-6" style={{ color: 'var(--muted)' }}>
          暂无{showMoney ? '金额' : '数量'}指标数据
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={Math.max(180, chartData.length * 32)}>
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 40, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--muted)' }}
              tickFormatter={v => showMoney ? `¥${(v / 1000).toFixed(0)}k` : String(v)} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: 'var(--muted)' }} width={72} />
            <Tooltip contentStyle={TOOLTIP_STYLE}
              formatter={(v: number) => [showMoney ? fmtMoney(v) : v.toLocaleString('zh-CN'), '']} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} fillOpacity={0.85}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

// ── 同比/环比 Tab ─────────────────────────────────────────────────────────────

interface YoYPopTabProps {
  rows: BiData['metrics']
  allKeys: string[]
  prevRows: BiData['previous_metrics']
  deltas: BiData['deltas']
  comparisonType: 'yoy' | 'pop'
  rangeStart?: number
  prevRangeStart?: number
}

function YoYPopTab({ rows, allKeys, prevRows, deltas, comparisonType, rangeStart, prevRangeStart }: YoYPopTabProps) {
  const [showMoney, setShowMoney] = useState(true)
  const [chartType, setChartType] = useState<'bar' | 'delta'>('bar')

  const row     = rows[0]     ?? {}
  const prevRow = (prevRows ?? [])[0] ?? {}
  const compLabel = comparisonType === 'yoy' ? '同比' : '环比'
  const currentLabel = rangeStart    ? tsToDateShort(rangeStart)     : '当期'
  const prevLabel    = prevRangeStart ? tsToDateShort(prevRangeStart) : '对比期'

  const filteredKeys = allKeys.filter(k => {
    const v = getVal(row[k])
    if (v === 0 && getVal(prevRow[k]) === 0) return false
    return showMoney ? MONEY_FIELDS.has(k) : !MONEY_FIELDS.has(k)
  })

  const barData = filteredKeys.map(k => ({
    name:   METRIC_LABELS[k] ?? k,
    当期:   getVal(row[k]),
    对比期: getVal(prevRow[k]),
    key: k,
  })).sort((a, b) => b['当期'] - a['当期']).slice(0, 10)

  const deltaData = filteredKeys
    .filter(k => deltas?.[k]?.delta_pct != null)
    .map(k => ({
      name:       METRIC_LABELS[k] ?? k,
      delta_pct:  deltas![k].delta_pct!,
      key: k,
    }))
    .sort((a, b) => b.delta_pct - a.delta_pct)
    .slice(0, 10)

  return (
    <div>
      {/* Controls */}
      <div className="flex flex-wrap gap-2 mb-4">
        <div className="flex gap-1">
          {([['bar', '对比柱图'], ['delta', '变化率']] as const).map(([ct, label]) => (
            <button key={ct} onClick={() => setChartType(ct)}
              className="text-xs px-3 py-1 rounded-full transition-colors"
              style={{
                background: chartType === ct ? 'rgba(52,211,153,0.15)' : 'rgba(255,255,255,0.05)',
                color: chartType === ct ? '#34d399' : 'var(--muted)',
                border: `1px solid ${chartType === ct ? 'rgba(52,211,153,0.35)' : 'var(--border)'}`,
              }}>
              {label}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {([true, false] as const).map(isMoney => (
            <button key={String(isMoney)} onClick={() => setShowMoney(isMoney)}
              className="text-xs px-3 py-1 rounded-full transition-colors"
              style={{
                background: showMoney === isMoney ? 'rgba(124,58,237,0.15)' : 'rgba(255,255,255,0.05)',
                color: showMoney === isMoney ? '#a78bfa' : 'var(--muted)',
                border: `1px solid ${showMoney === isMoney ? 'rgba(124,58,237,0.35)' : 'var(--border)'}`,
              }}>
              {isMoney ? '金额' : '数量'}
            </button>
          ))}
        </div>
      </div>

      {/* Delta summary cards */}
      {chartType === 'bar' && barData.length > 0 && (
        <div className="grid grid-cols-3 gap-1.5 mb-4">
          {barData.slice(0, 3).map(item => {
            const d = deltas?.[item.key]
            return (
              <div key={item.key} className="rounded-lg p-2.5 text-center"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
                <div className="text-xs truncate mb-1" style={{ color: 'var(--muted)' }}>{item.name}</div>
                <div className="text-sm font-bold" style={{ color: '#f1f5f9' }}>
                  {showMoney ? fmtMoney(item['当期']) : item['当期'].toLocaleString('zh-CN')}
                </div>
                {d && (
                  <div className="text-xs font-medium mt-0.5" style={{ color: deltaColor(d) }}>
                    {d.delta_pct != null && (d.delta_pct >= 0 ? '▲ ' : '▼ ')}{fmtDeltaPct(d)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Bar comparison chart */}
      {chartType === 'bar' && (
        barData.length === 0 ? (
          <p className="text-center text-sm py-6" style={{ color: 'var(--muted)' }}>暂无数据</p>
        ) : (
          <div>
            <p className="text-xs mb-2" style={{ color: 'var(--muted)' }}>
              {compLabel}：<span style={{ color: '#34d399' }}>{currentLabel}</span>
              {' vs '}<span style={{ color: '#7c3aed' }}>{prevLabel}</span>
            </p>
            <ResponsiveContainer width="100%" height={Math.max(200, barData.length * 44)}>
              <BarChart data={barData} layout="vertical" margin={{ top: 0, right: 8, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--muted)' }}
                  tickFormatter={v => showMoney ? `¥${(v / 1000).toFixed(0)}k` : String(v)} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: 'var(--muted)' }} width={64} />
                <Tooltip contentStyle={TOOLTIP_STYLE}
                  formatter={(v: number, name: string, entry) => {
                    const k = (entry.payload as { key?: string })?.key ?? ''
                    return [MONEY_FIELDS.has(k) ? fmtMoney(v) : v.toLocaleString('zh-CN'), name]
                  }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }}
                  formatter={(v) => v === '当期' ? currentLabel : prevLabel} />
                <Bar dataKey="当期"   name="当期"   fill="#34d399" fillOpacity={0.85} radius={[0, 3, 3, 0]} />
                <Bar dataKey="对比期" name="对比期" fill="#7c3aed" fillOpacity={0.6}  radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )
      )}

      {/* Delta % chart */}
      {chartType === 'delta' && (
        deltaData.length === 0 ? (
          <p className="text-center text-sm py-6" style={{ color: 'var(--muted)' }}>暂无变化率数据</p>
        ) : (
          <div>
            <p className="text-xs mb-2" style={{ color: 'var(--muted)' }}>
              {compLabel}变化率（%），正值为增长，负值为下降
            </p>
            <ResponsiveContainer width="100%" height={Math.max(200, deltaData.length * 36)}>
              <BarChart data={deltaData} layout="vertical" margin={{ top: 0, right: 48, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--muted)' }}
                  tickFormatter={v => `${v}%`} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: 'var(--muted)' }} width={64} />
                <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" />
                <Tooltip contentStyle={TOOLTIP_STYLE}
                  formatter={(v: number) => [`${v >= 0 ? '+' : ''}${v.toFixed(1)}%`, compLabel + '变化']} />
                <Bar dataKey="delta_pct" radius={[0, 3, 3, 0]}>
                  {deltaData.map((d, i) => (
                    <Cell key={i} fill={d.delta_pct >= 0 ? '#34d399' : '#f87171'} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )
      )}
    </div>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────────────────

interface Props { data: BiData }

export default function BiReportCard({ data }: Props) {
  const hasComparison = !!(data.comparison_type && data.previous_metrics?.length)
  const [tab, setTab] = useState<BiChartTab>(hasComparison ? 'yoy_pop' : 'overview')

  const rows    = data.metrics ?? []
  // Exclude metadata fields that are not metrics
  const META_KEYS = new Set(['period_label', 'bra_id', 'aggregation'])
  const allKeys = Array.from(new Set(rows.flatMap(r => Object.keys(r)))).filter(k => !META_KEYS.has(k))

  if (!rows.length) return null

  const compLabel = data.comparison_type === 'yoy' ? '同比' : data.comparison_type === 'pop' ? '环比' : null

  const ALL_TABS: { id: BiChartTab; label: string; show?: boolean }[] = [
    { id: 'overview',   label: '概览' },
    { id: 'table',      label: '表格' },
    { id: 'line',       label: '折线图' },
    { id: 'pie',        label: '饼图' },
    { id: 'comparison', label: '对比图' },
    { id: 'yoy_pop',    label: compLabel ?? '同比/环比', show: hasComparison },
  ]
  const TABS = ALL_TABS.filter(t => t.show !== false)

  return (
    <div className="rounded-2xl overflow-hidden"
      style={{ background: 'var(--elevated)', border: '1px solid rgba(52,211,153,0.25)' }}>

      {/* ── Header ──────────────────────────────────────────── */}
      <div className="px-5 py-3.5 flex items-center justify-between"
        style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2">
          <span className="text-lg">📊</span>
          <div>
            <div className="font-semibold text-sm" style={{ color: '#f1f5f9' }}>
              业务报表{compLabel && <span className="ml-1.5 text-xs font-normal" style={{ color: '#a78bfa' }}>({compLabel}分析)</span>}
            </div>
            <div className="text-xs" style={{ color: 'var(--muted)' }}>
              {data.bra_id ? `门店 ${data.bra_id}` : '全部门店'}
              {(data.range_start || data.range_end) && (
                <> · {tsToDate(data.range_start)} ~ {tsToDate(data.range_end)}</>
              )}
              {hasComparison && (
                <span style={{ color: '#7c3aed' }}>
                  {' vs '}{tsToDate(data.previous_range_start)} ~ {tsToDate(data.previous_range_end)}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <span className="text-xs px-2 py-1 rounded-full"
            style={{ background: 'rgba(52,211,153,0.1)', color: '#34d399', border: '1px solid rgba(52,211,153,0.25)' }}>
            {data.aggregation ?? 'SUM'}
          </span>
          {rows.length > 1 && (
            <span className="text-xs px-2 py-1 rounded-full"
              style={{ background: 'rgba(124,58,237,0.1)', color: '#a78bfa', border: '1px solid rgba(124,58,237,0.25)' }}>
              {rows.length} 条记录
            </span>
          )}
        </div>
      </div>

      {/* ── Tab Bar ─────────────────────────────────────────── */}
      <div className="flex overflow-x-auto" style={{ borderBottom: '1px solid var(--border)' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className="flex-shrink-0 px-4 py-2.5 text-xs font-medium transition-colors"
            style={{
              color: tab === t.id ? '#34d399' : 'var(--muted)',
              borderBottom: tab === t.id ? '2px solid #34d399' : '2px solid transparent',
              background: 'transparent',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Tab Content ─────────────────────────────────────── */}
      <div className="p-4">
        {tab === 'overview' && (
          <OverviewTab rows={rows} allKeys={allKeys} deltas={data.deltas} />
        )}
        {tab === 'table' && (
          <TableTab rows={rows} allKeys={allKeys}
            prevRows={data.previous_metrics} deltas={data.deltas} />
        )}
        {tab === 'line' && (
          <LineTab rows={rows} allKeys={allKeys}
            prevRows={data.previous_metrics}
            rangeStart={data.range_start}
            prevRangeStart={data.previous_range_start}
            comparisonType={data.comparison_type} />
        )}
        {tab === 'pie' && (
          <PieTab rows={rows}
            prevRows={data.previous_metrics}
            comparisonType={data.comparison_type} />
        )}
        {tab === 'comparison' && (
          <ComparisonTab rows={rows} allKeys={allKeys} />
        )}
        {tab === 'yoy_pop' && hasComparison && (
          <YoYPopTab
            rows={rows}
            allKeys={allKeys}
            prevRows={data.previous_metrics}
            deltas={data.deltas ?? {}}
            comparisonType={data.comparison_type!}
            rangeStart={data.range_start}
            prevRangeStart={data.previous_range_start}
          />
        )}
      </div>
    </div>
  )
}
