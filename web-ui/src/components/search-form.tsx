import { ChevronDown, CircleAlert, Search, SlidersHorizontal, Upload } from 'lucide-react'
import { useRef, type FormEvent, type ReactNode } from 'react'
import { QueryPicker } from '@/components/query-picker'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldLegend, FieldSet, FieldTitle } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui/spinner'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { boxKeys, type Box, type Query, type SearchMode } from '@/types'

type Props = {
  box: Box
  boxInvalid: boolean
  canSearch: boolean
  searching: boolean
  loadingImage: boolean
  hasImage: boolean
  mode: SearchMode
  queries: Query[]
  selectedQuery: string
  sourceError: string | null
  threshold: string
  defaultThreshold: number | null
  topK: number
  children: ReactNode
  onBoxChange: (key: keyof Box, value: string) => void
  onModeChange: (value: SearchMode) => void
  onSelectFile: (file?: File) => Promise<void>
  onSelectQuery: (id: string) => Promise<void>
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onThresholdChange: (value: string) => void
  onTopKChange: (value: number) => void
}

export function SearchForm(p: Props) {
  const uploadRef = useRef<HTMLInputElement>(null)

  return (
    <form onSubmit={p.onSubmit} className="min-w-0">
      <Card>
        <CardHeader>
          <CardTitle>01 / Исходный кадр</CardTitle>
          <CardDescription>Загрузите фото и выделите автомобиль.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="image-upload">Изображение</FieldLabel>
              <Input ref={uploadRef} id="image-upload" className="hidden" type="file" accept="image/jpeg,image/png" onChange={(event) => {
                void p.onSelectFile(event.target.files?.[0])
                event.target.value = ''
              }} />
              <Button type="button" variant="outline" onClick={() => uploadRef.current?.click()}>
                <Upload data-icon="inline-start" />{p.hasImage ? 'Заменить изображение' : 'Загрузить изображение'}
              </Button>
              <FieldDescription>JPEG или PNG · до 15 МиБ и 25 Мп</FieldDescription>
            </Field>
          </FieldGroup>

          <QueryPicker queries={p.queries} selectedQuery={p.selectedQuery} onSelectQuery={p.onSelectQuery} />

          {p.children}

          {p.sourceError && (
            <Alert variant="destructive">
              <CircleAlert />
              <AlertTitle>Проверьте исходный кадр</AlertTitle>
              <AlertDescription>{p.sourceError}</AlertDescription>
            </Alert>
          )}

          <FieldGroup>
            <Field>
              <FieldTitle id="mode-label">Режим поиска</FieldTitle>
              <ToggleGroup aria-labelledby="mode-label" variant="outline" value={[p.mode]} onValueChange={(values) => {
                if (values[0]) p.onModeChange(values[0] as SearchMode)
              }}>
                <ToggleGroupItem value="candidates">С порогом</ToggleGroupItem>
                <ToggleGroupItem value="ranking">Ранжирование</ToggleGroupItem>
              </ToggleGroup>
              <FieldDescription>{p.mode === 'candidates'
                ? 'Кандидаты или отказ по выбранному порогу.'
                : 'Похожие автомобили по порядку. Без решения о совпадении.'}</FieldDescription>
            </Field>
          </FieldGroup>

          <Collapsible>
            <CollapsibleTrigger render={<Button variant="ghost" type="button" />}>
              <SlidersHorizontal data-icon="inline-start" />Дополнительные настройки<ChevronDown data-icon="inline-end" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-4">
              <FieldGroup>
                <FieldSet>
                  <FieldLegend variant="label">Координаты BBox</FieldLegend>
                  <FieldDescription>В пикселях исходного кадра. Можно задать с клавиатуры.</FieldDescription>
                  <FieldGroup className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {boxKeys.map((key) => (
                      <Field key={key} data-invalid={p.boxInvalid}>
                        <FieldLabel htmlFor={`bbox-${key}`}>{key}</FieldLabel>
                        <Input id={`bbox-${key}`} type="number" min="0" step="1" aria-invalid={p.boxInvalid} value={p.box[key]} onChange={(event) => p.onBoxChange(key, event.target.value)} />
                      </Field>
                    ))}
                  </FieldGroup>
                </FieldSet>
                <Field>
                  <FieldLabel htmlFor="top-k">Количество результатов</FieldLabel>
                  <Input id="top-k" type="number" min="1" max="100" step="1" value={p.topK || ''} onChange={(event) => p.onTopKChange(Number(event.target.value))} />
                </Field>
                <Field data-disabled={p.mode !== 'candidates'}>
                  <FieldLabel htmlFor="threshold">Порог cosine</FieldLabel>
                  <Input id="threshold" type="number" min="-1" max="1" step="any" disabled={p.mode !== 'candidates'} value={p.threshold} placeholder={p.defaultThreshold?.toFixed(4) ?? 'Автоматически'} onChange={(event) => p.onThresholdChange(event.target.value)} />
                  <FieldDescription>Пустое поле — калиброванный порог модели{p.defaultThreshold !== null ? ` (${p.defaultThreshold.toFixed(4)})` : ''}. Это не вероятность.</FieldDescription>
                </Field>
              </FieldGroup>
            </CollapsibleContent>
          </Collapsible>
        </CardContent>
        <CardFooter>
          <Button className="w-full" size="lg" disabled={!p.canSearch || p.searching || p.loadingImage} type="submit">
            {p.searching || p.loadingImage ? <Spinner data-icon="inline-start" aria-label="Загрузка" /> : <Search data-icon="inline-start" />}
            {p.searching ? 'Ищем автомобиль…' : p.loadingImage ? 'Загружаем кадр…' : 'Найти автомобиль'}
          </Button>
        </CardFooter>
      </Card>
    </form>
  )
}
