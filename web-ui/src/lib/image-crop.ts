import { boxKeys, type Box } from '@/types'

export function isValidBox(box: Box, image: HTMLImageElement) {
  return boxKeys.every((key) => Number.isInteger(box[key]) && box[key] >= 0)
    && box.w > 0 && box.h > 0
    && box.x + box.w <= image.naturalWidth && box.y + box.h <= image.naturalHeight
}
