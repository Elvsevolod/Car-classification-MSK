import { expect, test, type Page } from '@playwright/test'

async function selectOfficialQuery(page: Page) {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Поиск автомобиля' })).toBeVisible()
  await expect(page.getByText(/\d+ кадров в галерее ·/)).toBeVisible({ timeout: 30_000 })
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  await page.getByTestId('query-option').first().click()
  await expect(page.getByRole('dialog', { name: 'Выберите исходный автомобиль' })).not.toBeVisible()
  await expect(page.getByTestId('source-canvas')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Найти автомобиль' })).toBeEnabled()
}

async function search(page: Page) {
  const responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/search') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Найти автомобиль' }).click()
  const response = await responsePromise
  expect(response.status()).toBe(200)
  await expect(page.getByTestId('search-decision')).toBeVisible()
  return response.json()
}

test('real API: calibrated default, result photos, JSON and metrics', async ({ page }) => {
  const health = await (await page.request.get('/api/health')).json()
  await selectOfficialQuery(page)
  await expect(page).toHaveTitle('Поиск автомобиля · ASU Team')
  await expect(page.locator('header')).toHaveText('ASU Team / Vehicle Re-ID')
  await expect(page.getByRole('button', { name: 'С порогом', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByLabel('Порог cosine')).not.toBeVisible()
  const payload = await search(page)
  expect(payload.mode).toBe('candidates')
  expect(health.profile).toBe('MVP_fusion_v25')
  expect(health.embedding_dim).toBe(2048)
  expect(health.default_threshold).toBeCloseTo(0.534365177154541)
  expect(payload.threshold).toBe(health.default_threshold)
  expect(payload.threshold_source).not.toBe('manual')
  const decision = page.getByTestId('search-decision')
  await expect(decision).toContainText(payload.refused ? 'Отказ: порог не пройден' : 'Порог пройден · проверьте кандидатов')
  await expect(decision).not.toContainText('совпадение найдено')
  await expect(decision).toContainText('Максимальный raw cosine')
  if (payload.results.length) {
    await expect(page.getByTestId('candidate-card')).toHaveCount(payload.results.length)
    const photo = page.getByTestId('candidate-card').first().locator('img')
    await expect(photo).toBeVisible()
    await expect.poll(() => photo.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true)
    await expect(photo).toHaveCSS('filter', 'none')
    const fullFrame = await page.request.get(await page.getByRole('link', { name: 'Полный кадр' }).first().getAttribute('href') as string)
    expect(fullFrame.ok()).toBe(true)
  }
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Скачать JSON' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('search-result.json')
  const stream = await download.createReadStream()
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  expect(JSON.parse(Buffer.concat(chunks).toString())).toEqual(payload)
  await page.getByRole('button', { name: 'Метрики и сведения о модели' }).click()
  await page.getByRole('button', { name: 'Технические подробности · JSON' }).click()
  await expect(page.locator('pre')).toContainText('threshold')
})

test('real API: ranking never claims an identity match, parameter changes clear results', async ({ page }) => {
  await selectOfficialQuery(page)
  await page.getByRole('button', { name: 'Ранжирование', exact: true }).click()
  const payload = await search(page)
  expect(payload.mode).toBe('ranking')
  expect(payload.threshold).toBeNull()
  await expect(page.getByTestId('candidate-card')).toHaveCount(10)
  await expect(page.getByTestId('search-decision')).toContainText('без решения о совпадении')
  await page.getByRole('button', { name: 'Дополнительные настройки' }).click()
  await page.getByLabel('Количество результатов').fill('3')
  await expect(page.getByTestId('search-decision')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Скачать JSON' })).toBeDisabled()
  const limited = await search(page)
  expect(limited.results).toHaveLength(3)
  await page.getByRole('button', { name: 'С порогом', exact: true }).click()
  await expect(page.getByTestId('search-decision')).toHaveCount(0)
  await page.getByLabel('Порог cosine').fill('1')
  const refusal = await search(page)
  expect(refusal.refused).toBe(true)
  await expect(page.getByTestId('candidate-card')).toHaveCount(3) // keep the requested ranking size on refusal
  await expect(page.getByTestId('accepted-candidate')).toHaveCount(0)
  await page.getByLabel('Порог cosine').fill('-1')
  await expect(page.getByTestId('search-decision')).toHaveCount(0)
  const accepted = await search(page)
  expect(accepted.refused).toBe(false)
  expect(accepted.threshold_source).toBe('manual')
  await expect(page.getByTestId('search-decision')).toContainText('ручной')
  await page.getByLabel('x', { exact: true }).fill('0')
  await expect(page.getByTestId('search-decision')).toHaveCount(0)
})

test('real API: upload, BBox validation, keyboard coordinates and pointer selection', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText(/\d+ кадров в галерее ·/)).toBeVisible()
  const queries = await (await page.request.get('/api/queries?limit=1')).json()
  const query = queries.items[0]
  const image = await page.request.get(`/api/images/query/${query.image_id}`)
  await page.getByLabel('Изображение', { exact: true }).setInputFiles({ name: 'vehicle.jpg', mimeType: 'image/jpeg', buffer: await image.body() })
  await expect(page.getByTestId('source-canvas')).toBeVisible()
  await page.getByRole('button', { name: 'Найти автомобиль' }).click()
  await expect(page.getByRole('alert')).toContainText('BBox должен иметь положительный размер')
  await expect(page.locator('form').getByRole('alert')).toContainText('BBox должен иметь положительный размер')
  await page.getByRole('button', { name: 'Дополнительные настройки' }).click()
  for (const key of ['x', 'y', 'w', 'h']) await page.getByLabel(key, { exact: true }).fill(String(query[key]))
  const firstInput = page.getByLabel('x', { exact: true })
  await firstInput.focus()
  await page.keyboard.press('Tab')
  await expect(page.getByLabel('y', { exact: true })).toBeFocused()
  await search(page)
  const canvas = page.getByTestId('source-canvas')
  await canvas.scrollIntoViewIfNeeded()
  const bounds = await canvas.boundingBox()
  if (!bounds) throw new Error('Canvas bounds unavailable')
  await page.mouse.move(bounds.x + bounds.width * 0.2, bounds.y + bounds.height * 0.2)
  await page.mouse.down()
  await page.mouse.move(bounds.x + bounds.width * 0.7, bounds.y + bounds.height * 0.7)
  await page.mouse.up()
  await expect(page.getByTestId('search-decision')).toHaveCount(0)
  expect(Number(await page.getByLabel('w', { exact: true }).inputValue())).toBeGreaterThan(0)
  expect(Number(await page.getByLabel('h', { exact: true }).inputValue())).toBeGreaterThan(0)
  await page.getByLabel('Изображение', { exact: true }).setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('not an image') })
  await expect(page.getByTestId('source-canvas')).not.toBeVisible()
  await expect(page.getByRole('button', { name: 'Найти автомобиль' })).toBeDisabled()
  await expect(page.getByRole('alert')).toBeVisible()
  await expect(page.locator('form').getByRole('alert')).toBeVisible()
})

for (const width of [390, 768, 1440]) {
  test(`real API: white UI and no horizontal overflow at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 })
    await page.emulateMedia({ colorScheme: 'dark' })
    await selectOfficialQuery(page)
    await page.getByRole('button', { name: 'Ранжирование', exact: true }).click()
    await search(page)
    await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
    await expect(page.getByRole('dialog', { name: 'Выберите исходный автомобиль' })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog', { name: 'Выберите исходный автомобиль' })).not.toBeVisible()
    await page.evaluate(() => window.scrollTo(0, 0))
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await expect(page.locator('html')).toHaveAttribute('lang', 'ru')
    await expect(page.locator('html')).toHaveCSS('color-scheme', 'light')
    // White is defined in OKLCH tokens; computed style must remain white, regardless of OS theme.
    const background = await page.locator('main').evaluate((element) => getComputedStyle(element).backgroundColor)
    expect(['oklch(1 0 0)', 'rgb(255, 255, 255)']).toContain(background)
    for (const photo of await page.getByTestId('candidate-card').locator('img').all()) {
      await photo.scrollIntoViewIfNeeded()
      await expect.poll(() => photo.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true)
    }
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.screenshot({ path: testInfo.outputPath(`white-ui-${width}.png`), fullPage: true })
    if (width === 390) {
      await page.screenshot({ path: testInfo.outputPath('white-ui-390-source-viewport.png') })
      await page.getByTestId('search-results').evaluate((element) => element.scrollIntoView({ block: 'start' }))
      await page.screenshot({ path: testInfo.outputPath('white-ui-390-results-viewport.png') })
    }
    await page.getByRole('button', { name: 'Сравнить кандидата 1', exact: true }).click()
    const comparison = page.getByRole('dialog', { name: 'Сравнение автомобилей' })
    await expect(comparison).toBeVisible()
    const candidate = page.getByTestId('compare-candidate-image')
    await expect.poll(() => candidate.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true)
    expect(await comparison.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`white-comparison-${width}.png`), fullPage: true })
  })
}

test('controlled race: changing mode while search is pending discards the late response', async ({ page }) => {
  await selectOfficialQuery(page)
  let release!: () => void
  let finish!: () => void
  let begin!: () => void
  const gate = new Promise<void>((resolve) => { release = resolve })
  const completed = new Promise<void>((resolve) => { finish = resolve })
  const entered = new Promise<void>((resolve) => { begin = resolve })
  await page.route('**/api/search', async (route) => {
    begin()
    await gate
    await route.fulfill({ json: { mode: 'candidates', threshold_source: 'manual', threshold: -1, confidence: 0.99, refused: false, results: [], elapsed_ms: 10 } }).catch(() => {})
    finish()
  })
  await page.getByRole('button', { name: 'Найти автомобиль' }).click()
  await entered
  await expect(page.getByRole('status', { name: 'Выполняется поиск' })).toBeVisible()
  await page.getByRole('button', { name: 'Ранжирование', exact: true }).click()
  release()
  await completed
  await expect(page.getByRole('status', { name: 'Выполняется поиск' })).toHaveCount(0)
  await expect(page.getByTestId('search-decision')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Скачать JSON' })).toBeDisabled()
})

test('controlled image fetch race: the latest selected example wins', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText(/\d+ кадров в галерее ·/)).toBeVisible()
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  const first = page.getByTestId('query-option').nth(0)
  const second = page.getByTestId('query-option').nth(1)
  const firstId = await first.getAttribute('data-query-id')
  const secondId = await second.getAttribute('data-query-id')
  expect(firstId).toBeTruthy()
  expect(secondId).toBeTruthy()
  let release!: () => void
  let finish!: () => void
  let begin!: () => void
  const gate = new Promise<void>((resolve) => { release = resolve })
  const completed = new Promise<void>((resolve) => { finish = resolve })
  const entered = new Promise<void>((resolve) => { begin = resolve })
  await page.route(`**/api/images/query/${firstId}`, async (route) => {
    const response = await route.fetch()
    begin()
    await gate
    await route.fulfill({ response }).catch(() => {})
    finish()
  })
  await first.click()
  await entered
  await expect(page.getByRole('dialog', { name: 'Выберите исходный автомобиль' })).not.toBeVisible()
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  await page.locator(`[data-testid="query-option"][data-query-id="${secondId}"]`).click()
  await expect(page.getByTestId('source-canvas')).toBeVisible()
  await expect(page.locator(`p[title="${secondId}"]`)).toBeVisible()
  release()
  await completed
  await expect(page.locator(`p[title="${secondId}"]`)).toBeVisible()
  await expect(page.locator(`p[title="${firstId}"]`)).toHaveCount(0)
})

test('controlled image decode race: an older decode cannot replace the current frame', async ({ page }) => {
  await page.addInitScript(() => {
    const original = HTMLImageElement.prototype.decode
    let calls = 0
    HTMLImageElement.prototype.decode = async function () {
      await original.call(this)
      calls += 1
      if (calls === 1) await new Promise<void>((resolve) => {
        Object.assign(window, { releaseFirstDecode: resolve })
      })
    }
  })
  await page.goto('/')
  await expect(page.getByText(/\d+ кадров в галерее ·/)).toBeVisible()
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  await page.getByTestId('query-option').nth(0).click()
  await page.waitForFunction(() => 'releaseFirstDecode' in window)
  await expect(page.getByRole('dialog', { name: 'Выберите исходный автомобиль' })).not.toBeVisible()
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  const second = page.getByTestId('query-option').nth(1)
  const secondId = await second.getAttribute('data-query-id')
  expect(secondId).toBeTruthy()
  await second.click()
  await expect(page.locator(`p[title="${secondId}"]`)).toBeVisible()
  await page.evaluate(() => (window as unknown as { releaseFirstDecode: () => void }).releaseFirstDecode())
  await expect(page.locator(`p[title="${secondId}"]`)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Найти автомобиль' })).toBeEnabled()
})

test('controlled API error: show the error and allow retry', async ({ page }) => {
  await selectOfficialQuery(page)
  await page.route('**/api/search', (route) => route.fulfill({ status: 503, json: { detail: 'Сервис временно недоступен. Повторите запрос.' } }), { times: 1 })
  await page.getByRole('button', { name: 'Найти автомобиль' }).click()
  await expect(page.getByRole('alert')).toContainText('Сервис временно недоступен')
  await expect(page.getByTestId('search-results').getByRole('alert')).toContainText('Сервис временно недоступен')
  await expect(page.getByRole('button', { name: 'Найти автомобиль' })).toBeEnabled()
  await search(page)
  await expect(page.getByRole('alert')).toHaveCount(0)
})
