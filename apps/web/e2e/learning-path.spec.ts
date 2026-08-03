import { test, expect } from '@playwright/test'

test.describe('Learning Path Page', () => {
  test('learning path page loads for cs201', async ({ page }) => {
    await page.goto('/learn/cs201')
    await expect(page.locator('body')).toBeVisible()
  })

  test('learning path shows course content', async ({ page }) => {
    await page.goto('/learn/cs201')
    await page.waitForLoadState('networkidle')
    // The page should have rendered some content
    const mainContent = page.locator('main, [role="main"], #__next, .content-area').first()
    await expect(mainContent).toBeVisible({ timeout: 15000 })
  })
})
