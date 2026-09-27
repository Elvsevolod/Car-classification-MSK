// The UI uses only /api; model inference and gallery access stay on the server.
import { useEffect, useRef, useState, type FormEvent, type PointerEvent } from 'react'
import { CarFront, ExternalLink, ImagePlus, RotateCcw } from 'lucide-react'
import { SearchForm } from '@/components/search-form'
import { SearchResults } from '@/components/search-results'
import { ComparisonDialog } from '@/components/comparison-dialog'
import { QueryCrop } from '@/components/query-crop'
import { ModelMetrics } from '@/components/model-metrics'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'
import { isValidBox } from '@/lib/image-crop'
import { boxKeys, type Box, type Health, type ModelMetrics as ModelMetricsData, type Query, type SearchMode, type SearchResult } from '@/types'

const initialBox: Box = { x: 0, y: 0, w: 0, h: 0 }

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const objectUrl = useRef<string | null>(null)
  const sourceVersion = useRef(0)
  const searchVersion = useRef(0)
  const imageRequest = useRef<AbortController | null>(null)
  const searchRequest = useRef<AbortController | null>(null)
  const dragStart = useRef<[number, number] | null>(null)
  const resultsRef = useRef<HTMLElement>(null)
  const returnFocus = useRef<HTMLElement | null>(null)
  const [comparisonIndex, setComparisonIndex] = useState<number | null>(null)
  const [sourceImage, setSourceImage] = useState<HTMLImageElement | null>(null)
  const [box, setBox] = useState<Box>(initialBox)
  const [blob, setBlob] = useState<Blob | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [sourceName, setSourceName] = useState('')
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [boxInvalid, setBoxInvalid] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [serviceError, setServiceError] = useState<string | null>(null)
  const [serviceRetry, setServiceRetry] = useState(0)
  const [queries, setQueries] = useState<Query[]>([])
  const [metrics, setMetrics] = useState<ModelMetricsData | null>(null)
  const [selectedQuery, setSelectedQuery] = useState('')
  const [result, setResult] = useState<SearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sourceError, setSourceError] = useState<string | null>(null)
  const [mode, setMode] = useState<SearchMode>('candidates')
  const [topK, setTopK] = useState(10)
  const [threshold, setThreshold] = useState('')
  const [searching, setSearching] = useState(false)
  const [loadingImage, setLoadingImage] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    const options = { signal: controller.signal }
    void Promise.all([fetch('/api/health', options), fetch('/api/queries?limit=1110', options)])
      .then(async ([healthResponse, queryResponse]) => {
        if (!healthResponse.ok || !queryResponse.ok) throw new Error('Сервис недоступен. Проверьте, что backend запущен.')
        const nextHealth = await healthResponse.json() as Health
        const nextQueries = await queryResponse.json() as { items: Query[] }
        if (controller.signal.aborted) return
        setHealth(nextHealth)
        setQueries(nextQueries.items)
        setServiceError(null)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setServiceError(reason instanceof Error ? reason.message : 'Сервис недоступен.')
      })
    void fetch('/api/metrics', options).then(async (response) => {
      if (!response.ok) return
      const payload = await response.json() as ModelMetricsData
      if (!controller.signal.aborted) setMetrics(payload)
    }).catch(() => { /* Search remains available if the optional metrics request fails. */ })
    return () => controller.abort()
  }, [serviceRetry])

  useEffect(() => {
    if (!result || !window.matchMedia('(max-width: 1023px)').matches) return
    resultsRef.current?.focus({ preventScroll: true })
    resultsRef.current?.scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' })
  }, [result])

  useEffect(() => () => {
    sourceVersion.current += 1
    searchVersion.current += 1
    imageRequest.current?.abort()
    searchRequest.current?.abort()
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !blob) return
    canvas.width = image.naturalWidth
    canvas.height = image.naturalHeight
    const context = canvas.getContext('2d')
    if (!context) return
    context.drawImage(image, 0, 0)
    if (box.w > 0 && box.h > 0) {
      const lineWidth = Math.max(2, canvas.width / 350)
      context.lineWidth = lineWidth * 2
      context.strokeStyle = '#171717'
      context.strokeRect(box.x, box.y, box.w, box.h)
      context.lineWidth = lineWidth
      context.strokeStyle = '#ffffff'
      context.strokeRect(box.x, box.y, box.w, box.h)
    }
  }, [blob, box])

  const invalidateSearch = () => {
    searchVersion.current += 1
    searchRequest.current?.abort()
    setSearching(false)
    setResult(null)
    setComparisonIndex(null)
    setError(null)
    setSourceError(null)
  }

  const changeBox = (nextBox: Box) => {
    invalidateSearch()
    setBoxInvalid(false)
    setBox(nextBox)
  }

  const clearSource = () => {
    sourceVersion.current += 1
    imageRequest.current?.abort()
    invalidateSearch()
    imageRef.current = null
    setSourceImage(null)
    dragStart.current = null
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = null
    setPreviewUrl(null)
    setBlob(null)
    setBox(initialBox)
    setSourceName('')
    setImageSize({ width: 0, height: 0 })
    setBoxInvalid(false)
    setLoadingImage(false)
    return sourceVersion.current
  }

  const loadBlob = async (nextBlob: Blob, version: number, name: string, nextBox: Box = initialBox) => {
    const url = URL.createObjectURL(nextBlob)
    try {
      const image = new Image()
      image.src = url
      await image.decode()
      if (sourceVersion.current !== version) return
      if (image.naturalWidth * image.naturalHeight > 25_000_000) throw new Error('Изображение превышает 25 мегапикселей.')
      imageRef.current = image
      setSourceImage(image)
      objectUrl.current = url
      setPreviewUrl(url)
      setBlob(nextBlob)
      setBox(nextBox)
      setSourceName(name)
      setImageSize({ width: image.naturalWidth, height: image.naturalHeight })
    } catch (reason) {
      if (sourceVersion.current === version) setSourceError(reason instanceof Error ? reason.message : 'Не удалось прочитать изображение.')
    } finally {
      if (objectUrl.current !== url) URL.revokeObjectURL(url)
      if (sourceVersion.current === version) setLoadingImage(false)
    }
  }

  const selectFile = async (file?: File) => {
    if (!file) return
    const version = clearSource()
    setSelectedQuery('')
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setSourceError('Выберите изображение JPEG или PNG.')
      return
    }
    if (file.size > 15 * 1024 * 1024) {
      setSourceError('Максимальный размер файла — 15 МиБ.')
      return
    }
    setLoadingImage(true)
    await loadBlob(file, version, file.name)
  }

  const selectQuery = async (queryId: string) => {
    const query = queries.find((item) => item.image_id === queryId)
    if (!query) return
    const version = clearSource()
    const controller = new AbortController()
    imageRequest.current = controller
    setSelectedQuery(queryId)
    setLoadingImage(true)
    try {
      const response = await fetch(`/api/images/query/${queryId}`, { signal: controller.signal })
      if (!response.ok) throw new Error('Не удалось загрузить пример. Попробуйте выбрать его ещё раз.')
      const nextBlob = await response.blob()
      if (sourceVersion.current !== version) return
      await loadBlob(nextBlob, version, queryId, query)
    } catch (reason) {
      if (sourceVersion.current === version && !controller.signal.aborted) {
        setSourceError(reason instanceof Error ? reason.message : 'Не удалось загрузить пример.')
        setLoadingImage(false)
      }
    }
  }

  // CSS scales the canvas; pointer coordinates must use source-image pixels.
  const getPoint = (event: PointerEvent<HTMLCanvasElement>): [number, number] | null => {
    const canvas = canvasRef.current
    if (!canvas || !imageRef.current) return null
    const rect = canvas.getBoundingClientRect()
    return [
      Math.max(0, Math.min(canvas.width, Math.round((event.clientX - rect.left) * canvas.width / rect.width))),
      Math.max(0, Math.min(canvas.height, Math.round((event.clientY - rect.top) * canvas.height / rect.height))),
    ]
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    invalidateSearch()
    const image = imageRef.current
    if (!blob || !image || loadingImage) return
    if (!isValidBox(box, image)) {
      setBoxInvalid(true)
      setSourceError('Выделите автомобиль. BBox должен иметь положительный размер и находиться внутри кадра.')
      return
    }
    if (!Number.isInteger(topK) || topK < 1 || topK > 100) {
      setSourceError('Количество результатов должно быть целым числом от 1 до 100.')
      return
    }
    const version = searchVersion.current
    const controller = new AbortController()
    searchRequest.current = controller
    setSearching(true)
    try {
      const form = new FormData()
      form.append('image', blob, blob.type === 'image/png' ? 'query.png' : 'query.jpg')
      boxKeys.forEach((key) => form.append(key, String(box[key])))
      form.append('top_k', String(topK))
      form.append('mode', mode)
      if (mode === 'candidates' && threshold.trim()) form.append('threshold', threshold)
      const response = await fetch('/api/search', { method: 'POST', body: form, signal: controller.signal })
      if (!response.ok) {
        const payload = await response.json().catch(() => null) as { detail?: unknown } | null
        if ([413, 415, 422].includes(response.status)) {
          if (searchVersion.current === version) setSourceError(typeof payload?.detail === 'string' ? payload.detail : 'Проверьте изображение, BBox и параметры поиска.')
          return
        }
        throw new Error(typeof payload?.detail === 'string' ? payload.detail : `Ошибка поиска (${response.status}). Проверьте параметры и повторите запрос.`)
      }
      const payload = await response.json() as SearchResult
      if (searchVersion.current === version) setResult(payload)
    } catch (reason) {
      if (searchVersion.current === version && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Ошибка поиска.')
    } finally {
      if (searchVersion.current === version) setSearching(false)
    }
  }

  const download = () => {
    if (!result) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'search-result.json'
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  const comparisonCandidates = result ? [...result.results] : []
  const accepted = result?.accepted_candidate
  if (accepted && !comparisonCandidates.some((item) => item.image_id === accepted.image_id)) comparisonCandidates.push(accepted)

  return (
    <main className="min-h-screen bg-background px-4 pb-8 text-foreground sm:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-7">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b py-5">
          <div className="flex items-center gap-3"><CarFront className="size-6" /><span className="text-sm font-semibold tracking-tight">ASU Team <span className="font-normal text-muted-foreground">/ Vehicle Re-ID</span></span></div>
        </header>

        <section className="flex flex-wrap items-end justify-between gap-4">
          <div className="flex flex-col gap-2"><h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Поиск автомобиля</h1><p className="max-w-xl text-sm leading-6 text-muted-foreground">Один кадр. Область автомобиля. Похожие изображения из галереи.</p></div>
          {health && <p className="text-xs text-muted-foreground">{health.gallery_size} кадров в галерее · {health.device}</p>}
        </section>
        {health?.profile && <details className="text-xs text-muted-foreground" data-testid="active-profile">
          <summary>Профиль: {health.profile} · {health.embedding_dim}D · {health.ranking_policy} / {health.candidate_policy}</summary>
          <p className="mt-2 break-all">Fingerprint: {health.profile_fingerprint}. Профиль выбирается при запуске сервиса.</p>
        </details>}

        {serviceError && <Alert variant="destructive"><AlertTitle>Нет связи с сервисом</AlertTitle><AlertDescription>{serviceError}</AlertDescription><Button type="button" variant="outline" className="mt-2 w-fit" onClick={() => setServiceRetry((value) => value + 1)}>Повторить подключение</Button></Alert>}

        <section className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
          <div id="source-workspace" tabIndex={-1} className="flex min-w-0 flex-col gap-5 self-stretch scroll-mt-4 outline-none">
          <SearchForm box={box} boxInvalid={boxInvalid} sourceError={sourceError} hasImage={!!blob} canSearch={!!blob && health?.status === 'ready'} searching={searching} loadingImage={loadingImage} mode={mode} queries={queries} selectedQuery={selectedQuery} threshold={threshold} defaultThreshold={health?.default_threshold ?? null} topK={topK} onBoxChange={(key, value) => changeBox({ ...box, [key]: Number(value) })} onModeChange={(value) => { invalidateSearch(); setMode(value) }} onSelectFile={selectFile} onSelectQuery={selectQuery} onSubmit={submit} onThresholdChange={(value) => { invalidateSearch(); setThreshold(value) }} onTopKChange={(value) => { invalidateSearch(); setTopK(value) }}>
            <div className="flex flex-col gap-3">
              <div className="overflow-hidden rounded-lg border bg-muted/40">
                <canvas ref={canvasRef} data-testid="source-canvas" aria-label="Кадр запроса. Выделите автомобиль указателем или задайте координаты в дополнительных настройках." className={cn('block h-auto w-full touch-none', !blob && 'hidden')}
                  onPointerDown={(event) => {
                    const point = getPoint(event)
                    if (!point) return
                    dragStart.current = point
                    event.currentTarget.setPointerCapture(event.pointerId)
                    changeBox({ x: point[0], y: point[1], w: 0, h: 0 })
                  }}
                  onPointerMove={(event) => {
                    const start = dragStart.current
                    const end = getPoint(event)
                    if (start && end) changeBox({ x: Math.min(start[0], end[0]), y: Math.min(start[1], end[1]), w: Math.abs(end[0] - start[0]), h: Math.abs(end[1] - start[1]) })
                  }}
                  onPointerUp={() => { dragStart.current = null }} onPointerCancel={() => { dragStart.current = null }}
                />
                {!blob && <Empty className="min-h-64"><EmptyHeader><EmptyMedia variant="icon">{loadingImage ? <Spinner aria-label="Загрузка изображения" /> : <ImagePlus />}</EmptyMedia><EmptyTitle>{loadingImage ? 'Загружаем кадр…' : 'Добавьте исходный кадр'}</EmptyTitle><EmptyDescription>Выберите файл или готовый пример выше.</EmptyDescription></EmptyHeader></Empty>}
              </div>
              {blob && <>
                <p className="truncate text-xs text-muted-foreground" title={sourceName}>{sourceName}</p>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-xs text-muted-foreground" role="status">{imageSize.width} × {imageSize.height} px · BBox {box.w} × {box.h}</p>
                  <Button type="button" variant="ghost" size="sm" onClick={() => changeBox(initialBox)}><RotateCcw data-icon="inline-start" />Сбросить BBox</Button>
                </div>
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span className="text-muted-foreground">Обведите автомобиль на кадре.</span>
                  {previewUrl && <a href={previewUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 underline underline-offset-4">Исходный кадр<ExternalLink className="size-3" /></a>}
                </div>
              </>}
            </div>
          </SearchForm>
          {result && sourceImage && isValidBox(box, sourceImage) && <div className="sticky top-4 hidden lg:block">
            <Card size="sm"><CardHeader><CardTitle>Исходный автомобиль</CardTitle></CardHeader><CardContent><QueryCrop image={sourceImage} box={box} testId="sticky-query-crop" className="h-40 w-full object-contain" /></CardContent></Card>
          </div>}
          </div>
          <section ref={resultsRef} tabIndex={-1} aria-label="Результаты поиска" className="min-w-0 scroll-mt-4 outline-none">
            <SearchResults result={result} searching={searching} error={error} onDownload={download} onCompare={(index, trigger) => { returnFocus.current = trigger; setComparisonIndex(index) }} onCompareAccepted={(trigger) => { returnFocus.current = trigger; setComparisonIndex(comparisonCandidates.findIndex((item) => item.image_id === accepted?.image_id)) }} />
            {result && <Button variant="outline" className="mt-4 w-full lg:hidden" onClick={() => {
              const source = document.getElementById('source-workspace')
              source?.focus({ preventScroll: true })
              source?.scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' })
            }}>К исходному кадру</Button>}
          </section>
        </section>

        <ComparisonDialog image={sourceImage} box={box} sourceUrl={previewUrl} candidates={comparisonCandidates} selectedIndex={comparisonIndex} onSelect={setComparisonIndex} onClose={() => setComparisonIndex(null)} returnFocus={returnFocus} />

        <footer className="flex min-w-0 flex-col gap-4 border-t pt-4">
          <ModelMetrics metrics={metrics} />
          <nav className="flex gap-5 text-xs text-muted-foreground" aria-label="Документация API"><a href="/docs" className="underline underline-offset-4">Swagger API</a><a href="/openapi.json" className="underline underline-offset-4">OpenAPI JSON</a></nav>
        </footer>
      </div>
    </main>
  )
}

export default App
