// Демо-клиент использует только существующий /api: здесь нет ML-логики и прямого доступа к PostgreSQL.
import { useEffect, useMemo, useRef, useState, type FormEvent, type PointerEvent } from 'react'
import { SquareDashedMousePointer } from 'lucide-react'

import { SearchForm } from '@/components/search-form'
import { SearchResults } from '@/components/search-results'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { boxKeys, type Box, type Health, type Query, type SearchResult } from '@/types'

const initialBox: Box = { x: 0, y: 0, w: 0, h: 0 }

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const dragStart = useRef<[number, number] | null>(null)
  const [box, setBox] = useState<Box>(initialBox)
  const [blob, setBlob] = useState<Blob | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [queries, setQueries] = useState<Query[]>([])
  const [metrics, setMetrics] = useState<object | null>(null)
  const [queryFilter, setQueryFilter] = useState('')
  const [selectedQuery, setSelectedQuery] = useState('')
  const [status, setStatus] = useState('Загрузите изображение или выберите query из списка.')
  const [result, setResult] = useState<SearchResult | null>(null)
  const [mode, setMode] = useState<'ranking' | 'candidates'>('ranking')
  const [topK, setTopK] = useState(10)
  const [threshold, setThreshold] = useState('')
  const [busy, setBusy] = useState(false)

  const visibleQueries = useMemo(() => {
    const normalized = queryFilter.trim().toLowerCase()
    const matching = normalized
      ? queries.filter((query) => query.image_id.toLowerCase().includes(normalized))
      : queries
    return matching.slice(0, 8)
  }, [queries, queryFilter])

  const draw = (nextBox: Box) => {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image) return
    const context = canvas.getContext('2d')
    if (!context) return

    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0)
    if (nextBox.w && nextBox.h) {
      context.strokeStyle = '#ffffff'
      context.lineWidth = Math.max(2, canvas.width / 400)
      context.strokeRect(nextBox.x, nextBox.y, nextBox.w, nextBox.h)
    }
  }

  useEffect(() => { draw(box) }, [box])

  // Загружаем неизменяемые для сессии данные один раз, чтобы UI сразу показывал состояние сервиса и query.
  useEffect(() => {
    void Promise.all([fetch('/api/health'), fetch('/api/queries?limit=1110'), fetch('/api/metrics')])
      .then(async ([healthResponse, queryResponse, metricsResponse]) => {
        if (!healthResponse.ok || !queryResponse.ok) throw new Error('Сервис недоступен')
        setHealth(await healthResponse.json() as Health)
        setQueries((await queryResponse.json() as { items: Query[] }).items)
        if (metricsResponse.ok) setMetrics(await metricsResponse.json() as object)
      })
      .catch((error: Error) => setStatus(error.message))
  }, [])

  const loadBlob = async (nextBlob: Blob, nextBox: Box = initialBox) => {
    const url = URL.createObjectURL(nextBlob)
    try {
      const image = new Image()
      image.src = url
      await image.decode()
      if (image.naturalWidth * image.naturalHeight > 25_000_000) {
        throw new Error('Изображение превышает 25 мегапикселей')
      }
      imageRef.current = image
      if (canvasRef.current) {
        canvasRef.current.width = image.naturalWidth
        canvasRef.current.height = image.naturalHeight
      }
      setBlob(nextBlob)
      setBox(nextBox)
      setResult(null)
      setStatus(`Исходный размер: ${image.naturalWidth}×${image.naturalHeight}. Выделите автомобиль.`)
      requestAnimationFrame(() => draw(nextBox))
    } catch (error) {
      setBlob(null)
      setStatus(error instanceof Error ? error.message : 'Не удалось загрузить изображение')
    } finally {
      URL.revokeObjectURL(url)
    }
  }

  const selectFile = async (file?: File) => {
    if (!file) return
    if (file.size > 15 * 1024 * 1024) {
      setStatus('Максимальный размер файла — 15 МиБ')
      return
    }
    setSelectedQuery('')
    setQueryFilter('')
    await loadBlob(file)
  }

  const selectQuery = async (queryId: string) => {
    setSelectedQuery(queryId)
    setQueryFilter(queryId)
    const query = queries.find((item) => item.image_id === queryId)
    if (!query) return

    setBusy(true)
    try {
      const response = await fetch(`/api/images/query/${query.image_id}`)
      if (!response.ok) throw new Error('Не удалось загрузить пример')
      await loadBlob(await response.blob(), query)
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Не удалось загрузить пример')
    } finally {
      setBusy(false)
    }
  }

  // Canvas может быть уменьшен CSS; переводим координаты указателя обратно в пиксели исходного изображения.
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
    const image = imageRef.current
    if (!blob || !image) return
    if (box.w <= 0 || box.h <= 0 || box.x + box.w > image.naturalWidth || box.y + box.h > image.naturalHeight) {
      setStatus('BBox должен находиться внутри исходного изображения.')
      return
    }

    setBusy(true)
    setResult(null)
    setStatus('Поиск по gallery…')
    try {
      const form = new FormData()
      form.append('image', blob, 'query.jpg')
      boxKeys.forEach((key) => form.append(key, String(box[key])))
      form.append('top_k', String(topK))
      form.append('mode', mode)
      if (threshold) form.append('threshold', threshold)

      const response = await fetch('/api/search', { method: 'POST', body: form })
      const payload = await response.json() as SearchResult & { detail?: string }
      if (!response.ok) throw new Error(payload.detail ?? 'Ошибка поиска')

      setResult(payload)
      setStatus(
        payload.refused
          ? `Отказ: нет кандидатов выше порога ${payload.threshold?.toFixed(4) ?? ''}.`
          : `Найдено ${payload.results.length} результатов за ${payload.elapsed_ms} мс.`,
      )
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Ошибка поиска')
    } finally {
      setBusy(false)
    }
  }

  const download = () => {
    if (!result) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'search-result.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="dark min-h-screen bg-[#080808] px-4 py-5 text-white sm:px-8">
      <div className="mx-auto max-w-7xl">
        <header className="flex items-center justify-between border-b border-white/20 pb-4 text-[11px] font-semibold tracking-[.14em] uppercase">
          <span className="flex items-center gap-3">
            <span className="grid size-6 place-items-center bg-white text-black">V</span>
            Кейс от ASU_TEAM
          </span>
          <span className={health ? 'hidden text-white/60 sm:block' : 'hidden text-amber-200 sm:block'}>
            {health ? '● Система поиска готова' : '○ Проверяем сервис'}
          </span>
        </header>

        <section className="grid gap-8 border-b border-white/20 py-10 lg:grid-cols-[1.4fr_.6fr]">
          <div>
            <p className="mb-3 text-[10px] tracking-[.16em] text-white/55 uppercase">Computer vision · re-identification</p>
            <h1 className="text-5xl leading-[.86] font-semibold tracking-[-.08em] uppercase sm:text-7xl">Поиск<br />автомобиля</h1>
            <p className="mt-6 border-t border-white/20 pt-4 text-xs text-white/80">
              {health ? `${health.model} · ${health.device} · gallery: ${health.gallery_size}` : 'Проверка сервиса…'}
            </p>
          </div>
          <p className="self-end text-sm leading-6 text-white/55">Загрузите кадр, выделите автомобиль и получите ранжированный Top‑10 либо обоснованный отказ.</p>
        </section>

        <section className="grid gap-5 py-5 lg:grid-cols-[360px_minmax(0,1fr)]">
          <SearchForm box={box} blob={blob} busy={busy} hasResult={result !== null} mode={mode} queryFilter={queryFilter} selectedQuery={selectedQuery} threshold={threshold} topK={topK} visibleQueries={visibleQueries} onBoxChange={(key, value) => { setBox({ ...box, [key]: Math.max(0, Number(value) || 0) }); setResult(null) }} onDownload={download} onModeChange={setMode} onQueryFilterChange={setQueryFilter} onSelectFile={selectFile} onSelectQuery={selectQuery} onSubmit={submit} onThresholdChange={setThreshold} onTopKChange={setTopK} />

          <Card className="min-w-0 bg-white/[.04] text-white ring-white/15">
            <CardHeader className="border-b border-white/15">
              <CardTitle className="flex items-center gap-2 text-xs tracking-[.12em] uppercase"><SquareDashedMousePointer />Кадр запроса</CardTitle>
              <CardDescription>Выделите автомобиль указателем или введите BBox.</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <canvas
                ref={canvasRef}
                className="block min-h-80 w-full bg-black touch-none"
                onPointerDown={(event) => { dragStart.current = getPoint(event); event.currentTarget.setPointerCapture(event.pointerId) }}
                onPointerMove={(event) => { const start = dragStart.current; const end = getPoint(event); if (!start || !end) return; setBox({ x: Math.min(start[0], end[0]), y: Math.min(start[1], end[1]), w: Math.abs(end[0] - start[0]), h: Math.abs(end[1] - start[1]) }) }}
                onPointerUp={() => { dragStart.current = null }}
                onPointerCancel={() => { dragStart.current = null }}
              />
            </CardContent>
          </Card>
        </section>

        <p className="bg-white px-4 py-3 text-xs text-black">{status}</p>

        <SearchResults result={result} />

        <section className="mt-5 grid gap-4 border-t border-white/20 pt-5 lg:grid-cols-[1fr_auto]">
          <details className="bg-white/[.04] p-4 text-xs">
            <summary className="cursor-pointer tracking-[.1em] uppercase">Метрики модели на локальной validation</summary>
            <pre className="mt-4 max-h-72 overflow-auto whitespace-pre-wrap text-white/55">{metrics ? JSON.stringify(metrics, null, 2) : 'Метрики пока не рассчитаны.'}</pre>
          </details>
          <nav className="flex items-start gap-5 pt-3 text-xs tracking-[.1em] uppercase"><a href="/docs">Swagger API</a><a href="/openapi.json">OpenAPI JSON</a></nav>
        </section>
      </div>
    </main>
  )
}

export default App
