import { test, expect } from '@playwright/test'

test.describe('Navigation', () => {
  test('home page loads and shows title', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible()
    await expect(page).toHaveTitle(/智图|EduMap/)
  })

  test('navigate to chat page', async ({ page }) => {
    await page.goto('/chat')
    await expect(page).toHaveURL(/\/chat/)
  })

  test('navigate to generate page', async ({ page }) => {
    await page.goto('/generate')
    await expect(page).toHaveURL(/\/generate/)
  })

  test('navigate to profile page', async ({ page }) => {
    await page.goto('/profile')
    await expect(page).toHaveURL(/\/profile/)
  })
})

test.describe('Mobile Navigation', () => {
  test.use({ viewport: { width: 375, height: 812 } })

  test('mobile tab bar is visible on small screens', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('navigation').first()).toBeVisible()
  })
})
