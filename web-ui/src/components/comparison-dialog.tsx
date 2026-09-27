import { useState, type RefObject } from 'react'
import { ArrowLeft, ArrowRight, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { QueryCrop } from '@/components/query-crop'
import type { Box, Candidate } from '@/types'

function ComparisonImages({ image, box, candidate, sourceUrl }: {
  image: HTMLImageElement; box: Box; candidate: Candidate; sourceUrl: string
}) {
  const [sourceZoom, setSourceZoom] = useState('1')
  const [candidateZoom, setCandidateZoom] = useState('1')
  const [imageFailed, setImageFailed] = useState(false)
  return (
    <div className="grid min-w-0 gap-5 md:grid-cols-2">
      <section className="flex min-w-0 flex-col gap-3" aria-label="Исходный автомобиль">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-medium">Исходный автомобиль</h3>
          <ToggleGroup aria-label="Масштаб исходного автомобиля" variant="outline" value={[sourceZoom]} onValueChange={(values) => { if (values[0]) setSourceZoom(values[0]) }}>
            <ToggleGroupItem value="1">1×</ToggleGroupItem><ToggleGroupItem value="2">2×</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div key={sourceZoom} className="h-52 overflow-auto rounded-lg border bg-muted/40 sm:h-80" tabIndex={0} aria-label="Просмотр исходного автомобиля">
          <div style={{ width: `${Number(sourceZoom) * 100}%`, height: `${Number(sourceZoom) * 100}%` }}>
            <QueryCrop image={image} box={box} testId="query-comparison-crop" className="block h-full w-full object-contain" />
          </div>
        </div>
        <a href={sourceUrl} target="_blank" rel="noreferrer" className="inline-flex w-fit items-center gap-1 text-sm underline underline-offset-4">Полный исходный кадр<ExternalLink className="size-3" /></a>
      </section>
      <section className="flex min-w-0 flex-col gap-3" aria-label={`Кандидат ${candidate.rank}`}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-medium">Кандидат {candidate.rank}</h3>
          <ToggleGroup aria-label="Масштаб кандидата" variant="outline" value={[candidateZoom]} onValueChange={(values) => { if (values[0]) setCandidateZoom(values[0]) }}>
            <ToggleGroupItem value="1">1×</ToggleGroupItem><ToggleGroupItem value="2">2×</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div key={candidateZoom} className="h-52 overflow-auto rounded-lg border bg-muted/40 sm:h-80" tabIndex={0} aria-label="Просмотр кандидата">
          {imageFailed ? <p role="status" className="p-4 text-sm text-muted-foreground">Не удалось загрузить изображение. Попробуйте открыть полный кадр.</p> : (
            <div style={{ width: `${Number(candidateZoom) * 100}%`, height: `${Number(candidateZoom) * 100}%` }}>
              <img data-testid="compare-candidate-image" src={candidate.crop_url} alt={`Автомобиль, кандидат ${candidate.rank}`} className="block h-full w-full object-contain" onError={() => setImageFailed(true)} />
            </div>
          )}
        </div>
        <a href={`/api/images/gallery/${candidate.image_id}`} target="_blank" rel="noreferrer" className="inline-flex w-fit items-center gap-1 text-sm underline underline-offset-4">Полный кадр кандидата<ExternalLink className="size-3" /></a>
      </section>
    </div>
  )
}

export function ComparisonDialog({ image, box, sourceUrl, candidates, selectedIndex, onSelect, onClose, returnFocus }: {
  image: HTMLImageElement | null; box: Box; sourceUrl: string | null; candidates: Candidate[]
  selectedIndex: number | null; onSelect: (index: number) => void; onClose: () => void
  returnFocus: RefObject<HTMLElement | null>
}) {
  const candidate = selectedIndex === null ? null : candidates[selectedIndex]
  return (
    <Dialog open={!!candidate && !!image && !!sourceUrl} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="max-h-[90dvh] w-[calc(100%-2rem)] max-w-5xl overflow-y-auto sm:max-w-5xl" finalFocus={() => returnFocus.current?.isConnected ? returnFocus.current : document.getElementById('source-workspace')}>
        <DialogHeader>
          <DialogTitle>Сравнение автомобилей</DialogTitle>
          <DialogDescription>Сравните детали кузова и внешний вид. Позиция в выдаче не подтверждает идентичность автомобиля.</DialogDescription>
        </DialogHeader>
        {candidate && image && sourceUrl && <ComparisonImages key={candidate.image_id} image={image} box={box} candidate={candidate} sourceUrl={sourceUrl} />}
        <DialogFooter className="flex-wrap gap-2 sm:justify-between">
          <Button variant="outline" disabled={selectedIndex === null || selectedIndex <= 0} onClick={() => { if (selectedIndex !== null) onSelect(selectedIndex - 1) }}><ArrowLeft data-icon="inline-start" />Предыдущий кандидат</Button>
          <Button variant="outline" disabled={selectedIndex === null || selectedIndex >= candidates.length - 1} onClick={() => { if (selectedIndex !== null) onSelect(selectedIndex + 1) }}>Следующий кандидат<ArrowRight data-icon="inline-end" /></Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
