import { expect, test, type Page } from '@playwright/test'

async function selectOfficialQuery(page: Page) {
  await page.goto('/')
  await expect(page.getByText('Кейс от ASU_TEAM')).toBeVisible()
  await expect(page.getByText('Система поиска готова')).toBeVisible({ timeout: 30_000 })

  const query = page.getByTestId('query-option').first()
  await expect(query).toBeVisible()
  await query.click()
  await expect(page.locator('canvas')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Запустить поиск' })).toBeEnabled()
}

test('production React UI loads a query and returns ranked candidates', async ({ page }) => {
  await selectOfficialQuery(page)
  await page.getByRole('button', { name: 'Запустить поиск' }).click()

  await expect(page.getByTestId('search-decision')).toBeVisible()
  await expect(page.getByTestId('candidate-card').first()).toBeVisible()
  await expect(page.getByText('Confidence:')).toBeVisible()
})

test('mobile UI fits the viewport and shows a refusal in candidates mode', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await selectOfficialQuery(page)

  await page.getByLabel('Режим').selectOption('candidates')
  await page.getByLabel('Порог cosine').fill('1')
  await page.getByRole('button', { name: 'Запустить поиск' }).click()

  await expect(page.getByTestId('search-decision')).toContainText('Решение / отказ')
  await expect(page.getByTestId('candidate-card')).toHaveCount(0)

  const layout = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, width: window.innerWidth }))
  expect(layout.scrollWidth).toBeLessThanOrEqual(layout.width)
})
