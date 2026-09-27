import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'
import type { SearchResult } from '../src/types'

async function chooseExample(page: Page) {
  await page.goto('/')
  await expect(page.getByText(/\d+ кадров в галерее ·/)).toBeVisible({ timeout: 30_000 })
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  await page.getByTestId('query-option').first().click()
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await expect(page.getByRole('button', { name: 'Найти автомобиль', exact: true })).toBeEnabled()
}

async function ranking(page: Page): Promise<SearchResult> {
  await page.getByRole('button', { name: 'Ранжирование', exact: true }).click()
  const response = page.waitForResponse((item) => item.url().endsWith('/api/search') && item.request().method() === 'POST')
  await page.getByRole('button', { name: 'Найти автомобиль', exact: true }).click()
  const result = await response
  expect(result.ok()).toBe(true)
  await expect(page.getByTestId('candidate-card')).toHaveCount(10)
  return result.json()
}

test('comparison: clean source crop, correct candidate, zoom reset and keyboard focus', async ({ page }) => {
  await chooseExample(page)
  await page.getByRole('button', { name: 'Дополнительные настройки' }).click()
  const box = Object.fromEntries(await Promise.all(['x', 'y', 'w', 'h'].map(async (key) => [key, Number(await page.getByLabel(key, { exact: true }).inputValue())])))
  const sourceUrl = await page.getByRole('link', { name: 'Исходный кадр', exact: true }).getAttribute('href')
  if (!sourceUrl) throw new Error('Missing source image')
  const expectedCrop = await page.evaluate(async ({ sourceUrl, box }) => {
    const image = new Image()
    image.src = sourceUrl
    await image.decode()
    const canvas = document.createElement('canvas')
    canvas.width = box.w
    canvas.height = box.h
    canvas.getContext('2d')!.drawImage(image, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h)
    return canvas.toDataURL()
  }, { sourceUrl, box })
  const result = await ranking(page)
  await expect(page.getByText(/Обработка запроса: \d+ мс/)).toBeVisible()
  const card = page.getByTestId('candidate-card').first()
  await expect(card.getByText('Cosine', { exact: true })).not.toBeVisible()
  await card.getByRole('button', { name: 'Подробности', exact: true }).click()
  await expect(card.getByText(`ID: ${result.results[0].image_id}`, { exact: true })).toBeVisible()
  await expect(card.getByText('Cosine', { exact: true })).toBeVisible()
  const trigger = page.getByRole('button', { name: 'Сравнить кандидата 1', exact: true })
  await trigger.click()
  const dialog = page.getByRole('dialog', { name: 'Сравнение автомобилей' })
  await expect(dialog).toBeVisible()
  expect(await page.getByTestId('query-comparison-crop').evaluate((canvas: HTMLCanvasElement) => canvas.toDataURL())).toBe(expectedCrop)
  const candidate = page.getByTestId('compare-candidate-image')
  await expect(candidate).toHaveAttribute('src', result.results[0].crop_url)
  const sourceZoom = page.getByRole('group', { name: 'Масштаб исходного автомобиля' })
  const candidateZoom = page.getByRole('group', { name: 'Масштаб кандидата' })
  const sourceWidth = await page.getByTestId('query-comparison-crop').evaluate((element) => element.clientWidth)
  const candidateWidth = await candidate.evaluate((element) => element.clientWidth)
  await expect(dialog.getByRole('button', { name: 'Предыдущий кандидат' })).toBeDisabled()
  await sourceZoom.getByRole('button', { name: '2×', exact: true }).click()
  await candidateZoom.getByRole('button', { name: '2×', exact: true }).click()
  await expect(sourceZoom.getByRole('button', { name: '2×', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(candidateZoom.getByRole('button', { name: '2×', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect.poll(() => page.getByTestId('query-comparison-crop').evaluate((element) => element.clientWidth)).toBe(sourceWidth * 2)
  await expect.poll(() => candidate.evaluate((element) => element.clientWidth)).toBe(candidateWidth * 2)
  await dialog.getByRole('button', { name: 'Следующий кандидат' }).click()
  await expect(candidate).toHaveAttribute('src', result.results[1].crop_url)
  await expect(sourceZoom.getByRole('button', { name: '1×', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(candidateZoom.getByRole('button', { name: '1×', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await dialog.getByRole('button', { name: 'Предыдущий кандидат' }).click()
  await expect(candidate).toHaveAttribute('src', result.results[0].crop_url)
  await page.keyboard.press('Escape')
  await expect(dialog).not.toBeVisible()
  await expect(trigger).toBeFocused()
  await page.getByRole('button', { name: 'Открыть сравнение кандидата 2', exact: true }).click()
  await expect(candidate).toHaveAttribute('src', result.results[1].crop_url)
  await page.keyboard.press('Escape')
  await expect(page.getByRole('button', { name: 'Открыть сравнение кандидата 2', exact: true })).toBeFocused()
  await page.getByLabel('x', { exact: true }).fill('0')
  await expect(dialog).not.toBeVisible()
  await expect(page.getByTestId('candidate-card')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Скачать JSON' })).toBeDisabled()
})

test('comparison: EXIF rotation is applied before cropping and the crop has no BBox border', async ({ page }) => {
  await chooseExample(page)
  const result = await ranking(page)
  await page.getByLabel('Изображение', { exact: true }).setInputFiles(fileURLToPath(new URL('./fixtures/rotated-colors.jpg', import.meta.url)))
  await expect(page.getByText('16 × 32 px · BBox 0 × 0', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Дополнительные настройки' }).click()
  for (const [key, value] of Object.entries({ x: 0, y: 0, w: 16, h: 16 })) await page.getByLabel(key, { exact: true }).fill(String(value))
  await page.route('**/api/search', (route) => route.fulfill({ json: result }), { times: 1 })
  await page.getByRole('button', { name: 'Найти автомобиль', exact: true }).click()
  await page.getByRole('button', { name: 'Сравнить кандидата 1', exact: true }).click()
  const crop = await page.getByTestId('query-comparison-crop').evaluate((canvas: HTMLCanvasElement) => ({
    width: canvas.width, height: canvas.height,
    corner: Array.from(canvas.getContext('2d')!.getImageData(0, 0, 1, 1).data),
    center: Array.from(canvas.getContext('2d')!.getImageData(8, 8, 1, 1).data),
  }))
  expect(crop.width).toBe(16)
  expect(crop.height).toBe(16)
  for (const color of [crop.corner, crop.center]) {
    expect(color[0]).toBeGreaterThan(240)
    expect(color[1]).toBeLessThan(15)
    expect(color[2]).toBeLessThan(15)
    expect(color[3]).toBe(255)
  }
})

test('comparison: replacing the source closes an open comparison and removes stale candidates', async ({ page }) => {
  await chooseExample(page)
  await ranking(page)
  await page.getByRole('button', { name: 'Сравнить кандидата 1', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Сравнение автомобилей' })
  await expect(dialog).toBeVisible()
  await page.getByLabel('Изображение', { exact: true }).setInputFiles(fileURLToPath(new URL('./fixtures/rotated-colors.jpg', import.meta.url)))
  await expect(dialog).not.toBeVisible()
  await expect(page.getByTestId('candidate-card')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Скачать JSON' })).toBeDisabled()
  await expect(page.getByText('16 × 32 px · BBox 0 × 0', { exact: true })).toBeVisible()
})

test('comparison: missing candidate image keeps navigation and full-frame access available', async ({ page }) => {
  await chooseExample(page)
  const result = await ranking(page)
  // Use a distinct URL so a previously decoded card image cannot hide the simulated error.
  result.results[0].crop_url += '&ui_fixture=missing'
  await page.route(`**${result.results[0].crop_url}`, (route) => route.fulfill({ status: 404 }))
  await page.route('**/api/search', (route) => route.fulfill({ json: result }), { times: 1 })
  await page.getByRole('button', { name: 'Найти автомобиль', exact: true }).click()
  await page.getByRole('button', { name: 'Сравнить кандидата 1', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Сравнение автомобилей' })
  await expect(dialog.getByText('Не удалось загрузить изображение. Попробуйте открыть полный кадр.', { exact: true })).toBeVisible()
  await expect(dialog.getByRole('link', { name: 'Полный кадр кандидата', exact: true })).toHaveAttribute('href', `/api/images/gallery/${result.results[0].image_id}`)
  await dialog.getByRole('button', { name: 'Следующий кандидат' }).click()
  const candidate = page.getByTestId('compare-candidate-image')
  await expect(candidate).toHaveAttribute('src', result.results[1].crop_url)
  await expect.poll(() => candidate.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true)
})

test('examples: eight thumbnails, more, ID filter, selected state and failed thumbnail fallback', async ({ page }, testInfo) => {
  const { items } = await (await page.request.get('/api/queries?limit=1')).json()
  const failedThumbnailId = items[0].image_id
  let failedThumbnailRequested = false
  const requests = new Set<string>()
  page.on('request', (request) => {
    if (/\/api\/images\/query\/.+\?crop=true$/.test(request.url())) requests.add(request.url())
  })
  await page.route(`**/api/images/query/${failedThumbnailId}?crop=true`, (route) => {
    failedThumbnailRequested = true
    return route.fulfill({ status: 404 })
  })
  await page.goto('/')
  await expect(page.getByText(/\d+ кадров в галерее ·/)).toBeVisible()
  expect(requests.size).toBe(0)
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Выберите исходный автомобиль' })
  await expect(dialog.getByTestId('query-option')).toHaveCount(8)
  const firstId = await dialog.getByTestId('query-option').first().getAttribute('data-query-id')
  if (!firstId) throw new Error('Missing example ID')
  expect(firstId).toBe(failedThumbnailId)
  await expect.poll(() => failedThumbnailRequested).toBe(true)
  await expect(dialog.getByTestId('query-option').first().getByText('Фото недоступно', { exact: true })).toBeVisible()
  await expect.poll(() => requests.size).toBeGreaterThan(0)
  expect(requests.size).toBeLessThanOrEqual(8)
  for (const image of await dialog.locator('img').all()) {
    await expect.poll(() => image.evaluate((element: HTMLImageElement) => element.complete && element.naturalWidth > 0)).toBe(true)
  }
  await dialog.screenshot({ path: testInfo.outputPath('query-picker.png') })
  await dialog.getByRole('button', { name: 'Показать ещё' }).click()
  await expect(dialog.getByTestId('query-option')).toHaveCount(16)
  expect(requests.size).toBeLessThanOrEqual(16)
  await dialog.getByLabel('Поиск примера по ID').fill(firstId)
  await expect(dialog.getByTestId('query-option')).toHaveCount(1)
  await dialog.getByTestId('query-option').click()
  await expect(dialog).not.toBeVisible()
  await expect(page.locator(`p[title="${firstId}"]`)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Найти автомобиль', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  await expect(dialog.locator(`[data-query-id="${firstId}"]`)).toHaveAttribute('aria-pressed', 'true')
  await dialog.getByLabel('Поиск примера по ID').fill('definitely-no-such-query')
  await expect(dialog.getByText('Примеры не найдены', { exact: true })).toBeVisible()
})

test('metrics: summary uses validation values and keeps raw JSON nested', async ({ page }, testInfo) => {
  const metrics = {
    model: 'Контрольная модель', threshold: 0.4321,
    partition_identity_counts: { train: 900, calibration: 200, validation: 123 },
    calibration: { mAP_at_10: 0.9999, candidate_F1: 0.9998, TNR: 0.9997 },
    validation: { mAP_at_10: 0.8123, candidate_F1: 0.6543, TNR: 0.9876, known_queries: 210, unknown_queries: 32 },
  }
  await page.route('**/api/metrics', (route) => route.fulfill({ json: metrics }))
  await chooseExample(page)
  await page.getByRole('button', { name: 'Метрики и сведения о модели' }).click()
  const footer = page.locator('footer')
  await expect(footer).toContainText('Контрольная модель')
  for (const value of [/81[,.]23\s*%/, /65[,.]43\s*%/, /98[,.]76\s*%/, /242/, /0[,.]4321/]) await expect(footer).toContainText(value)
  await expect(footer).toContainText('не скрытый тест организаторов')
  await expect(footer).toContainText('использовалась при выборе модели')
  await expect(footer.locator('pre')).not.toBeVisible()
  await footer.screenshot({ path: testInfo.outputPath('model-metrics.png') })
  await footer.getByRole('button', { name: 'Технические подробности · JSON' }).click()
  expect(JSON.parse(await footer.locator('pre').innerText())).toEqual(metrics)
})

test('metrics: missing optional report does not block searching', async ({ page }) => {
  await page.route('**/api/metrics', (route) => route.fulfill({ status: 503, json: { detail: 'Unavailable' } }))
  await chooseExample(page)
  await page.getByRole('button', { name: 'Метрики и сведения о модели' }).click()
  await expect(page.locator('footer')).toContainText('Метрики недоступны')
  await ranking(page)
  await expect(page.getByTestId('search-decision')).toContainText('без решения о совпадении')
})

test.describe('200% browser-zoom equivalent: 1440 physical pixels / 720 CSS pixels', () => {
  test.use({ viewport: { width: 720, height: 500 }, deviceScaleFactor: 2 })

  test('reflow, modal controls and mobile return-to-source stay accessible', async ({ page }, testInfo) => {
    await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' })
    await chooseExample(page)
    await ranking(page)
    await expect(page.getByRole('region', { name: 'Результаты поиска', exact: true })).toBeFocused()
    await page.getByRole('button', { name: 'К исходному кадру', exact: true }).click()
    await expect(page.getByTestId('source-canvas')).toBeInViewport()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.getByRole('button', { name: 'Сравнить кандидата 1', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Сравнение автомобилей' })
    await expect(dialog).toBeVisible()
    await expect(page.getByTestId('query-comparison-crop')).toBeVisible()
    await expect(page.getByTestId('compare-candidate-image')).toBeVisible()
    expect(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
    await dialog.getByRole('button', { name: 'Следующий кандидат' }).click()
    await page.screenshot({ path: testInfo.outputPath('white-ui-200-percent-comparison.png'), fullPage: true })
    await page.keyboard.press('Escape')
    await expect(dialog).not.toBeVisible()
    await page.screenshot({ path: testInfo.outputPath('white-ui-200-percent.png'), fullPage: true })
  })
})
