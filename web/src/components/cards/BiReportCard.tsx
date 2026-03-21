'use client'
import type { BiData } from '@/types'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const METRIC_LABELS: Record<string, string> = {
  turnover: '总销售额', actual_income: '实际收入', order_quantity: '订单数',
  average_order_price: '均单价', customer_number: '顾客数',
  dining_room_amount: '堂食', togo_amount: '外带',
  store_delivery_amount: '配送', discount_amount: '折扣额',
  member_quantity: '会员订单', new_member_number: '新会员',
  tips_amount: '小费', refund_amount: '退款',
}

const MONEY_FIELDS = new Set(['turnover','actual_income','actual_income_no_tips','tips_amount','average_order_price','dining_room_amount','togo_amount','store_delivery_amount','discount_amount','refund_amount','pay_by_cash','pay_by_card','pay_by_online','pay_by_savings','pos_amount','amount_not_taxed','vat'])

function fmt(key: string, val: unknown): string {
  if (typeof val === 'object' && val !== null && 'value' in val) {
    return fmt(key, (val as { value: unknown }).value)
  }
  if (val == null) return '—'
  const n = Number(val)
  if (isNaN(n)) return String(val)
  if (MONEY_FIELDS.has(key)) return `¥${n.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  return n.toLocaleString('zh-CN')
}

function getVal(v: unknown): number {
  if (typeof v === 'object' && v !== null && 'value' in v) return Number((v as { value: unknown }).value) || 0
  return Number(v) || 0
}

interface Props { data: BiData }

export default function BiReportCard({ data }: Props) {
  const rows = data.metrics ?? []
  if (!rows.length) return null

  // Flatten all metrics for display
  const allKeys = Array.from(new Set(rows.flatMap(r => Object.keys(r))))
  const chartData = allKeys.map(k => ({
    name: METRIC_LABELS[k] ?? k,
    value: rows.reduce((sum, r) => sum + getVal(r[k]), 0),
    key: k,
  })).filter(d => d.value > 0).slice(0, 8)

  return (
    <div className="rounded-2xl overflow-hidden"
      style={{ background: 'var(--elevated)', border: '1px solid rgba(52,211,153,0.25)' }}>
      {/* Header */}
      <div className="px-5 py-4 flex items-center justify-between"
        style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2">
          <span className="text-lg">📊</span>
          <div>
            <div className="font-semibold text-sm" style={{ color: '#f1f5f9' }}>业务报表</div>
            {data.bra_id && (
              <div className="text-xs" style={{ color: 'var(--muted)' }}>门店 {data.bra_id}</div>
            )}
          </div>
        </div>
        {data.aggregation && (
          <span className="text-xs px-2 py-1 rounded-full"
            style={{ background: 'rgba(52,211,153,0.1)', color: '#34d399', border: '1px solid rgba(52,211,153,0.25)' }}>
            {data.aggregation}
          </span>
        )}
      </div>

      {/* Metric cards */}
      <div className="p-4">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {allKeys.map(key => {
            const row = rows[0]
            const raw = row?.[key]
            const label = METRIC_LABELS[key] ?? key
            const displayVal = fmt(key, raw)
            return (
              <div key={key} className="rounded-xl p-3"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
                <div className="text-xs mb-1 truncate" style={{ color: 'var(--muted)' }}>{label}</div>
                <div className="font-semibold text-sm truncate" style={{ color: '#f1f5f9' }}>{displayVal}</div>
              </div>
            )
          })}
        </div>

        {/* Chart */}
        {chartData.length >= 2 && (
          <div className="mt-4">
            <div className="text-xs mb-2" style={{ color: 'var(--muted)' }}>数据对比</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                  itemStyle={{ color: '#34d399' }}
                />
                <Bar dataKey="value" fill="#34d399" radius={[4, 4, 0, 0]} fillOpacity={0.85} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}
