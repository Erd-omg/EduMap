import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import SettingsPage from '../page'

// Mock Next.js link
vi.mock('next/link', () => ({
  default: 'a',
}))

// Mock getUserId
vi.mock('@/lib/user-id', () => ({
  getUserId: () => 'test-user',
}))

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default: profile fetch succeeds
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        profile: {
          display_name: '测试用户',
          interaction_style: { textual: 0.8, visual: 0.3, interactive: 0.2 },
          focus_characteristics: { recommended_session_length: 30 },
        },
      }),
    })
  })

  it('renders the page title after loading', async () => {
    render(<SettingsPage />)
    await waitFor(() => {
      expect(screen.getByText('⚙️ 用户设置')).toBeTruthy()
    })
  })

  it('loads and displays profile data', async () => {
    render(<SettingsPage />)

    await waitFor(() => {
      const nameInput = screen.getByPlaceholderText('输入你的名称') as HTMLInputElement
      expect(nameInput.value).toBe('测试用户')
    })
  })

  it('renders all preference sections', async () => {
    render(<SettingsPage />)

    await waitFor(() => {
      expect(screen.getByText('个人资料')).toBeTruthy()
      expect(screen.getByText('学习偏好')).toBeTruthy()
      expect(screen.getByText('通知偏好')).toBeTruthy()
      expect(screen.getByText('数据管理')).toBeTruthy()
    })
  })

  it('renders learning style options', async () => {
    render(<SettingsPage />)

    await waitFor(() => {
      expect(screen.getByText('视觉型')).toBeTruthy()
      expect(screen.getByText('文字型')).toBeTruthy()
      expect(screen.getByText('交互型')).toBeTruthy()
    })
  })

  it('renders notification toggle', async () => {
    render(<SettingsPage />)

    await waitFor(() => {
      expect(screen.getByText('学习提醒')).toBeTruthy()
    })
  })

  it('renders data management section', async () => {
    render(<SettingsPage />)

    await waitFor(() => {
      expect(screen.getByText('导出学习数据')).toBeTruthy()
      expect(screen.getByText('清除所有数据')).toBeTruthy()
      expect(screen.getByText('导出')).toBeTruthy()
      expect(screen.getByText('清除')).toBeTruthy()
    })
  })

  it('renders save button', async () => {
    render(<SettingsPage />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '保存设置' })).toBeTruthy()
    })
  })

  it('renders return to dashboard link', async () => {
    render(<SettingsPage />)

    await waitFor(() => {
      const link = screen.getByText('返回仪表盘')
      expect(link.closest('a')?.getAttribute('href')).toBe('/dashboard')
    })
  })
})
