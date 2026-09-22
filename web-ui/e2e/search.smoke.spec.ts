import { expect, test } from '@playwright/test'

test('production React UI loads a query and returns ranked candidates', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByText('Кейс от ASU_TEAM')).toBeVisible()
  await expect(page.getByText('Система поиска готова')).toBeVisible({ timeout: 30_000 })

  const query = page.getByTestId('query-option').first()
  await expect(query).toBeVisible()
  await query.click()

  await expect(page.locator('canvas')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Запустить поиск' })).toBeEnabled()
  await page.getByRole('button', { name: 'Запустить поиск' }).click()

  await expect(page.getByTestId('search-decision')).toBeVisible()
  await expect(page.getByTestId('candidate-card').first()).toBeVisible()
  await expect(page.getByText('Confidence:')).toBeVisible()
})
