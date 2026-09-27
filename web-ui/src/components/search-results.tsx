import { ChevronDown, Download, ExternalLink, ScanSearch, SearchX, TriangleAlert } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button, buttonVariants } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import type { Candidate, SearchResult } from '@/types'

function CandidateCard({ item, onCompare }: { item: Candidate; onCompare: (trigger: HTMLElement) => void }) {
  return (
    <Card size="sm" data-testid="candidate-card" className="min-w-0">
      <CardHeader>
        <CardTitle>Кандидат {item.rank}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Button type="button" variant="ghost" onClick={(event) => onCompare(event.currentTarget)} aria-label={`Открыть сравнение кандидата ${item.rank}`} className="h-auto w-full cursor-zoom-in overflow-hidden p-0">
          <img className="h-48 w-full object-contain" src={item.crop_url} alt={`Автомобиль, кандидат ${item.rank}`} loading="lazy" />
        </Button>
        <Button type="button" variant="outline" aria-label={`Сравнить кандидата ${item.rank}`} onClick={(event) => onCompare(event.currentTarget)}>Сравнить</Button>
        <Collapsible>
          <CollapsibleTrigger render={<Button variant="ghost" size="sm" />}><ChevronDown data-icon="inline-start" />Подробности</CollapsibleTrigger>
          <CollapsibleContent className="pt-2">
            <p className="mb-2 break-all text-xs text-muted-foreground">ID: {item.image_id}</p>
            <dl className="grid grid-cols-[1fr_auto] gap-1 text-xs tabular-nums">
              <dt className="text-muted-foreground">Cosine</dt><dd>{item.similarity.toFixed(4)}</dd>
              <dt className="text-muted-foreground">Rerank</dt><dd>{item.rerank_score.toFixed(4)}</dd>
            </dl>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
      <CardFooter>
        <a className={buttonVariants({ variant: 'ghost', size: 'sm' })} href={`/api/images/gallery/${item.image_id}`} target="_blank" rel="noreferrer">
          Полный кадр<ExternalLink data-icon="inline-end" />
        </a>
      </CardFooter>
    </Card>
  )
}

export function SearchResults({ result, searching, error, onDownload, onCompare, onCompareAccepted }: {
  result: SearchResult | null
  searching: boolean
  error: string | null
  onDownload: () => void
  onCompare: (index: number, trigger: HTMLElement) => void
  onCompareAccepted: (trigger: HTMLElement) => void
}) {
  const ranking = result?.mode === 'ranking'
  return (
    <Card className="min-w-0" data-testid="search-results">
      <CardHeader>
        <CardTitle>02 / Результаты поиска</CardTitle>
        <CardDescription>Сравните найденные автомобили с исходным кадром.</CardDescription>
        <CardAction>
          <Button variant="ghost" size="icon" aria-label="Скачать JSON" disabled={!result} onClick={onDownload}><Download data-icon="inline-start" /></Button>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-5" aria-busy={searching}>
        {error && <Alert variant="destructive"><TriangleAlert /><AlertTitle>Не удалось выполнить действие</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>}
        {searching ? (
          <div role="status" aria-label="Выполняется поиск" className="flex flex-col gap-4">
            <p className="text-sm text-muted-foreground">Сравниваем с галереей…</p>
            <Skeleton className="h-20 w-full" />
            <div className="grid grid-cols-2 gap-3"><Skeleton className="h-56" /><Skeleton className="h-56" /></div>
          </div>
        ) : result ? (
          <>
            <Alert data-testid="search-decision" role="status">
              <AlertTitle>{ranking ? 'Похожие автомобили · без решения о совпадении' : result.refused ? 'Отказ: порог не пройден' : 'Порог пройден · проверьте кандидатов'}</AlertTitle>
              <AlertDescription>
                {ranking ? 'Это ранжированный список, а не подтверждение идентичности автомобиля.' : result.refused ? 'В галерее нет кандидатов, прошедших выбранный порог.' : 'Максимальный raw cosine прошёл порог. Показан ранжированный список, а не подтверждение совпадения. Проверьте кадры визуально.'}
                <dl className="mt-3 grid grid-cols-[1fr_auto] gap-1 text-xs tabular-nums">
                  <dt>Максимальный raw cosine</dt><dd>{result.confidence.toFixed(4)}</dd>
                  <dt>Порог</dt><dd>{ranking ? 'не применяется' : result.threshold?.toFixed(4) ?? 'не задан'}</dd>
                  {!ranking && <><dt>Источник порога</dt><dd>{result.threshold_source === 'manual' ? 'ручной' : 'калибровка модели'}</dd></>}
                </dl>
              </AlertDescription>
            </Alert>
            {result.demo_threshold_override && <Alert><AlertTitle>Ручной порог · только демо</AlertTitle><AlertDescription>Конкурсная конфигурация и замороженный порог не изменены.</AlertDescription></Alert>}
            {!ranking && result.accepted_candidate && <section data-testid="accepted-candidate" className="flex flex-col gap-3 rounded-lg border p-4">
              <h3 className="text-sm font-medium">Принятый кандидат · {result.candidate_policy}</h3>
              <p className="break-all text-xs text-muted-foreground">ID: {result.accepted_candidate.image_id} · место в ranking: {result.accepted_candidate.rank}</p>
              <img className="h-48 w-full object-contain" src={result.accepted_candidate.crop_url} alt="Автомобиль принятого кандидата" />
              <p className="text-xs">Cosine этого изображения: {result.accepted_candidate.similarity.toFixed(4)}. Не вероятность совпадения.</p>
              <Button type="button" variant="outline" onClick={(event) => onCompareAccepted(event.currentTarget)}>Сравнить принятого кандидата</Button>
            </section>}
            <p className="text-xs text-muted-foreground">{result.profile} · ranking: {result.ranking_policy} · candidate: {result.candidate_policy}</p>
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
              <Badge variant="secondary">{result.results.length} результатов</Badge><span>Обработка запроса: {result.elapsed_ms.toFixed(0)} мс</span>
            </div>
            {result.results.length ? <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">{result.results.map((item, index) => <CandidateCard key={item.image_id} item={item} onCompare={(trigger) => onCompare(index, trigger)} />)}</div> : (
              <Empty className="min-h-48"><EmptyHeader><EmptyMedia variant="icon"><SearchX /></EmptyMedia><EmptyTitle>Кандидатов нет</EmptyTitle><EmptyDescription>Можно проверить область автомобиля или просмотреть ранжирование без порога.</EmptyDescription></EmptyHeader></Empty>
            )}
          </>
        ) : (
          <Empty className="min-h-80"><EmptyHeader><EmptyMedia variant="icon"><ScanSearch /></EmptyMedia><EmptyTitle>Здесь появятся результаты</EmptyTitle><EmptyDescription>Загрузите изображение или выберите пример, выделите автомобиль и запустите поиск.</EmptyDescription></EmptyHeader></Empty>
        )}
      </CardContent>
      <CardFooter><p className="text-xs leading-relaxed text-muted-foreground">Rerank задаёт порядок. Максимальный raw cosine используется для отказа; ни один из этих показателей не является вероятностью совпадения.</p></CardFooter>
    </Card>
  )
}
