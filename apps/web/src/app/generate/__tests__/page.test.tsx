import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import GeneratePage from '../page'

// Mock Next.js link
vi.mock('next/link', () => ({
  default: 'a',
}))

// Mock getUserId
vi.mock('@/lib/user-id', () => ({
  getUserId: () => 'test-user',
}))

// Mock child components
vi.mock('@/components/resources/resource-library', () => ({
  ResourceLibrary: ({ refreshKey }: { refreshKey: number }) => (
    <div data-testid="resource-library">资源库 (refreshKey: {refreshKey})</div>
  ),
}))

vi.mock('@/components/resources/generation-progress', () => ({
  GenerationProgress: ({ sessionId }: { sessionId: string | null }) => (
    <div data-testid="generation-progress">生成进度: {sessionId || 'none'}</div>
  ),
}))

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

const mockSessionStorage: Record<string, string> = {}
const mockSessionStorageGet = vi.fn((key: string) => mockSessionStorage[key] ?? null)
const mockSessionStorageSet = vi.fn((key: string, value: string) => { mockSessionStorage[key] = value })
const mockSessionStorageRemove = vi.fn((key: string) => { delete mockSessionStorage[key] })

vi.stubGlobal('sessionStorage', {
  getItem: mockSessionStorageGet,
  setItem: mockSessionStorageSet,
  removeItem: mockSessionStorageRemove,
})

describe('GeneratePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default fetch responses
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/v1/kg/courses')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            courses: [{ id: 'cs201', name: '数据结构' }, { id: 'cs101', name: '计算机基础' }],
          }),
        })
      }
      if (url.includes('/api/v1/kg/courses/')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            nodes: [
              { id: 'kp-array', name: '数组', description: '数组概念', difficulty: 2 },
              { id: 'kp-linkedlist', name: '链表', description: '链表概念', difficulty: 3 },
            ],
          }),
        })
      }
      if (url.includes('/api/v1/orchestrator/status/')) {
        return Promise.resolve({ ok: false })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    })
  })

  it('renders the page title', async () => {
    render(<GeneratePage />)
    await waitFor(() => {
      expect(screen.getByText('多智能体资源生成')).toBeTruthy()
    })
  })

  it('renders the page description', async () => {
    render(<GeneratePage />)
    await waitFor(() => {
      expect(screen.getByText(/系统通过 6 个 AI 智能体协作完成/)).toBeTruthy()
    })
  })

  it('renders course selector', async () => {
    render(<GeneratePage />)
    await waitFor(() => {
      expect(screen.getByText(/数据结构/)).toBeTruthy()
    })
  })

  it('renders KP selector', async () => {
    render(<GeneratePage />)
    await waitFor(() => {
      const selects = screen.getAllByRole('combobox')
      // There should be at least one select (the KP selector or course selector)
      expect(selects.length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders the generate button', async () => {
    render(<GeneratePage />)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /开始生成/ })).toBeTruthy()
    })
  })

  it('renders child components after restore completes', async () => {
    render(<GeneratePage />)
    await waitFor(() => {
      expect(screen.getByTestId('resource-library')).toBeTruthy()
      expect(screen.getByTestId('generation-progress')).toBeTruthy()
    })
  })

  it('disables generate button when no KP selected', async () => {
    render(<GeneratePage />)
    await waitFor(() => {
      const btn = screen.getByRole('button', { name: /开始生成/ })
      expect(btn.hasAttribute('disabled')).toBeTruthy()
    })
  })
})
