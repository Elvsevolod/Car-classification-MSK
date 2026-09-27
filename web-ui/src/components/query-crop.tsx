import { useEffect, useRef } from 'react'
import type { Box } from '@/types'

export function QueryCrop({ image, box, className, testId }: {
  image: HTMLImageElement
  box: Box
  className?: string
  testId?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    // Decode has already applied EXIF orientation. Never copy the annotated canvas.
    const context = canvas.getContext('2d')
    context?.clearRect(0, 0, box.w, box.h)
    context?.drawImage(image, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h)
  }, [image, box])
  return <canvas ref={canvasRef} width={box.w} height={box.h} role="img" aria-label="Область исходного автомобиля" data-testid={testId} className={className} />
}
