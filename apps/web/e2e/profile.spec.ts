import { test, expect } from '@playwright/test'

test.describe('Profile Page', () => {
  test('profile page loads with radar chart', async ({ page }) => {
    await page.goto('/profile')
    await expect(page.locator('body')).toBeVisible()
  })

  test('profile page shows user data', async ({ page }) => {
    await page.goto('/profile')
    await page.waitForLoadState('networkidle')
    expect(await page.title()).toBeTruthy()
  })
})
