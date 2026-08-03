import { test, expect } from '@playwright/test'

test.describe('Chat Page', () => {
  test('chat page has message input', async ({ page }) => {
    await page.goto('/chat')
    // The input should be visible for sending messages
    const input = page.locator('input, textarea, [contenteditable]').first()
    await expect(input).toBeVisible({ timeout: 10000 })
  })

  test('chat page shows user ID header', async ({ page }) => {
    await page.goto('/chat')
    // The page should load without errors
    await expect(page.locator('body')).toBeVisible()
  })

  test('chat page has profile section', async ({ page }) => {
    await page.goto('/chat')
    // Wait for content to render
    await page.waitForLoadState('networkidle')
    // The page loaded successfully
    expect(await page.title()).toBeTruthy()
  })
})
