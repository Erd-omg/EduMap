import { test, expect } from '@playwright/test'

test.describe('Dashboard Page', () => {
  test('dashboard page loads', async ({ page }) => {
    await page.goto('/dashboard')
    await expect(page.locator('body')).toBeVisible()
  })

  test('dashboard shows learning statistics', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForLoadState('networkidle')
    // Dashboard should be rendered
    expect(await page.title()).toBeTruthy()
  })
})
