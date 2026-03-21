import type { WeatherData } from '@/types'

const CONDITION_ICON: Record<string, string> = {
  晴: '☀️', clear: '☀️',
  多云: '⛅', cloudy: '⛅', partly_cloudy: '⛅',
  阴: '☁️', overcast: '☁️',
  小雨: '🌦', rain: '🌧', 中雨: '🌧', 大雨: '🌧',
  雷雨: '⛈️', thunderstorm: '⛈️',
  雪: '❄️', snow: '❄️',
  雾: '🌫️', fog: '🌫️',
  霾: '😷', haze: '😷',
}

function getIcon(condition?: string): string {
  if (!condition) return '🌡️'
  const lower = condition.toLowerCase()
  for (const [k, v] of Object.entries(CONDITION_ICON)) {
    if (lower.includes(k.toLowerCase())) return v
  }
  return '🌡️'
}

interface Props { data: WeatherData }

export default function WeatherCard({ data }: Props) {
  const icon = getIcon(data.weather_condition)

  return (
    <div className="rounded-2xl overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, #0c1445 0%, #1a237e 50%, #0d47a1 100%)',
        border: '1px solid rgba(56,189,248,0.3)',
      }}>
      <div className="p-5">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs font-medium mb-1" style={{ color: 'rgba(186,230,253,0.8)' }}>
              📍 {data.city ?? '当前位置'}
            </div>
            <div className="flex items-end gap-2">
              <span className="text-5xl font-thin" style={{ color: '#fff' }}>
                {data.temperature != null ? `${data.temperature}°` : '—'}
              </span>
              {data.feels_like != null && (
                <span className="text-sm mb-2" style={{ color: 'rgba(186,230,253,0.7)' }}>
                  体感 {data.feels_like}°
                </span>
              )}
            </div>
            {data.weather_condition && (
              <div className="text-sm mt-1" style={{ color: 'rgba(186,230,253,0.9)' }}>
                {data.weather_condition}
              </div>
            )}
          </div>
          <div className="text-6xl leading-none">{icon}</div>
        </div>

        {/* Stats row */}
        {(data.humidity != null || data.wind_speed != null) && (
          <div className="flex gap-4 mt-4 pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            {data.humidity != null && (
              <div className="flex items-center gap-1.5 text-sm" style={{ color: 'rgba(186,230,253,0.8)' }}>
                <span>💧</span> 湿度 {data.humidity}%
              </div>
            )}
            {data.wind_speed != null && (
              <div className="flex items-center gap-1.5 text-sm" style={{ color: 'rgba(186,230,253,0.8)' }}>
                <span>💨</span> 风速 {data.wind_speed} km/h
              </div>
            )}
          </div>
        )}

        {/* Description */}
        {data.description && (
          <div className="mt-3 text-xs" style={{ color: 'rgba(186,230,253,0.6)' }}>
            {data.description}
          </div>
        )}
      </div>

      {/* Forecast */}
      {data.forecast && data.forecast.length > 0 && (
        <div className="px-5 pb-4">
          <div className="text-xs mb-2" style={{ color: 'rgba(186,230,253,0.5)' }}>未来预报</div>
          <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(data.forecast.length, 5)}, 1fr)` }}>
            {data.forecast.slice(0, 5).map((f, i) => (
              <div key={i} className="text-center rounded-lg py-2 px-1"
                style={{ background: 'rgba(255,255,255,0.07)' }}>
                <div className="text-xs mb-1" style={{ color: 'rgba(186,230,253,0.6)' }}>{f.date}</div>
                <div className="text-sm">{getIcon(f.condition)}</div>
                <div className="text-xs font-medium mt-1" style={{ color: '#fff' }}>
                  {f.high != null ? `${f.high}°` : '—'}
                </div>
                {f.low != null && (
                  <div className="text-xs" style={{ color: 'rgba(186,230,253,0.5)' }}>{f.low}°</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
