import { ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import type { ModelMetrics as ModelMetricsData } from '@/types'

const percent = new Intl.NumberFormat('ru-RU', { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 })

export function ModelMetrics({ metrics }: { metrics: ModelMetricsData | null }) {
  const validation = metrics?.validation
  const known = validation?.known_queries
  const unknown = validation?.unknown_queries
  const queryCount = typeof known === 'number' && typeof unknown === 'number' ? known + unknown : null
  const scores = [
    { label: 'mAP@10', value: validation?.mAP_at_10, description: 'Качество первых 10 результатов' },
    { label: 'F1', value: validation?.candidate_F1, description: 'Баланс точности и полноты кандидатов' },
    { label: 'TNR', value: validation?.TNR, description: 'Доля верных отказов для запросов без пары' },
  ]

  return (
    <Collapsible className="min-w-0">
      <CollapsibleTrigger render={<Button variant="ghost" className="h-auto max-w-full" />}>
        <span className="min-w-0 whitespace-normal text-left">Метрики и сведения о модели</span><ChevronDown data-icon="inline-end" />
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-3">
        {metrics && validation ? (
          <div className="flex min-w-0 flex-col gap-4">
            <div className="grid gap-4 sm:grid-cols-3">
              {scores.map((score) => (
                <dl key={score.label} className="flex flex-col gap-1">
                  <dt className="text-sm text-muted-foreground">{score.label}</dt>
                  <dd className="text-2xl font-medium tabular-nums" data-testid={`metric-${score.label}`}>{typeof score.value === 'number' && Number.isFinite(score.value) ? percent.format(score.value) : '—'}</dd>
                  <dd className="text-xs text-muted-foreground">{score.description}</dd>
                </dl>
              ))}
            </div>
            <dl className="grid gap-2 text-sm">
              <div><dt className="text-muted-foreground">Модель</dt><dd className="break-words">{metrics.model}</dd></div>
              <div><dt className="text-muted-foreground">Запросов в validation</dt><dd>{queryCount ?? 'Не указано'}</dd></div>
              <div><dt className="text-muted-foreground">Калиброванный порог cosine</dt><dd>{Number.isFinite(metrics.threshold) ? metrics.threshold.toFixed(4) : 'Не указан'} · не вероятность</dd></div>
            </dl>
            <p className="text-sm text-muted-foreground">Локальная validation, не скрытый тест организаторов; использовалась при выборе модели. Отказы проверены с синтетическими запросами без пары.</p>
            <Collapsible>
              <CollapsibleTrigger render={<Button type="button" variant="outline" size="sm" />}>
                Технические подробности · JSON<ChevronDown data-icon="inline-end" />
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-3">
                <pre className="max-h-72 overflow-auto rounded-lg bg-muted p-4 text-xs break-all whitespace-pre-wrap">{JSON.stringify(metrics, null, 2)}</pre>
              </CollapsibleContent>
            </Collapsible>
          </div>
        ) : <p className="text-sm text-muted-foreground">Метрики недоступны. Это не мешает поиску.</p>}
      </CollapsibleContent>
    </Collapsible>
  )
}
