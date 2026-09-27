import { ImageOff, Images } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { Field, FieldGroup, FieldLabel } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import type { Query } from '@/types'

type Props = {
  queries: Query[]
  selectedQuery: string
  onSelectQuery: (id: string) => Promise<void>
}

function QueryThumbnail({ id }: { id: string }) {
  const [failed, setFailed] = useState(false)

  return failed ? (
    <span className="flex aspect-4/3 w-full flex-col items-center justify-center gap-2 rounded-md bg-muted text-muted-foreground">
      <ImageOff aria-hidden="true" />
      <span className="text-xs whitespace-normal">Фото недоступно</span>
    </span>
  ) : (
    <img className="aspect-4/3 w-full rounded-md bg-muted object-contain" src={`/api/images/query/${encodeURIComponent(id)}?crop=true`} alt="" loading="lazy" onError={() => setFailed(true)} />
  )
}

export function QueryPicker({ queries, selectedQuery, onSelectQuery }: Props) {
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [limit, setLimit] = useState(8)
  const filtered = queries.filter((query) => query.image_id.toLowerCase().includes(filter.trim().toLowerCase()))
  const visible = filtered.slice(0, limit)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger render={<Button variant="outline" type="button" />}>
        <Images data-icon="inline-start" />Выбрать пример
      </DialogTrigger>
      <DialogContent className="flex max-h-[85dvh] flex-col sm:max-w-3xl">
        <DialogHeader className="pr-7">
          <DialogTitle>Выберите исходный автомобиль</DialogTitle>
          <DialogDescription>Загрузим полный кадр и область автомобиля из разметки.</DialogDescription>
        </DialogHeader>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="query-filter">Поиск примера по ID</FieldLabel>
            <Input id="query-filter" placeholder="Введите часть ID" value={filter} onChange={(event) => {
              setFilter(event.target.value)
              setLimit(8)
            }} />
          </Field>
        </FieldGroup>
        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto p-1">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4" aria-label="Примеры запросов">
            {visible.map((query) => (
              <Button key={query.image_id} type="button" variant={selectedQuery === query.image_id ? 'secondary' : 'outline'} className="h-auto min-w-0 flex-col gap-2 p-2" data-testid="query-option" data-query-id={query.image_id} aria-label={`Выбрать пример ${query.image_id}`} aria-pressed={selectedQuery === query.image_id} title={query.image_id} onClick={() => {
                void onSelectQuery(query.image_id)
                setOpen(false)
              }}>
                <QueryThumbnail id={query.image_id} />
                <span className="w-full truncate">{query.image_id}</span>
                {selectedQuery === query.image_id && <span className="text-xs">Выбран</span>}
              </Button>
            ))}
          </div>
          {!visible.length && (
            <Empty>
              <EmptyHeader>
                <EmptyTitle>Примеры не найдены</EmptyTitle>
                <EmptyDescription>{queries.length ? 'Попробуйте другую часть ID.' : 'Можно загрузить собственное изображение.'}</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
          <p className="text-sm text-muted-foreground" aria-live="polite">Показано {visible.length} из {filtered.length}</p>
          {visible.length < filtered.length && <Button type="button" variant="outline" onClick={() => setLimit((value) => value + 8)}>Показать ещё</Button>}
        </div>
      </DialogContent>
    </Dialog>
  )
}
