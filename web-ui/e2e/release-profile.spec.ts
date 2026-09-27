import { expect, test } from '@playwright/test'

test('raw accepted candidate remains separate from ranking top1, with its own image and score', async ({ page }) => {
  const response = await page.request.get('/api/queries?limit=2')
  const rows = (await response.json()).items
  const imageUrl = `/api/images/query/${rows[0].image_id}?crop=true`
  const item = (id: string, rank: number, similarity: number) => ({
    image_id: id, rank, similarity, rerank_score: -.1, x: 0, y: 0, w: 1, h: 1, crop_url: imageUrl,
  })
  await page.route('**/api/search', (route) => route.fulfill({ json: {
    mode: 'candidates', threshold_source: 'calibration', threshold: .53, confidence: .91,
    refused: false, results: [item('ranking-first', 1, .61)], accepted_candidate: item('raw-candidate', 12, .91),
    profile: 'RC_R1_equal3_v18', ranking_policy: 'less_graph', candidate_policy: 'raw_top1',
    demo_threshold_override: false, elapsed_ms: 20,
  } }))
  await page.goto('/')
  await page.getByRole('button', { name: 'Выбрать пример', exact: true }).click()
  await page.getByTestId('query-option').first().click()
  await page.getByRole('button', { name: 'Найти автомобиль' }).click()
  const accepted = page.getByTestId('accepted-candidate')
  await expect(accepted).toContainText('raw-candidate')
  await expect(accepted).toContainText('0.9100')
  await expect(accepted).not.toContainText('0.6100')
  await expect(accepted.locator('img')).toHaveAttribute('src', imageUrl)
  await expect(page.getByTestId('candidate-card')).toHaveCount(1)
  await page.getByRole('button', { name: 'Сравнить принятого кандидата' }).click()
  await expect(page.getByRole('dialog').getByRole('link', { name: 'Полный кадр кандидата' })).toHaveAttribute('href', '/api/images/gallery/raw-candidate')
  await expect(page.getByTestId('compare-candidate-image')).toHaveAttribute('src', imageUrl)
})
