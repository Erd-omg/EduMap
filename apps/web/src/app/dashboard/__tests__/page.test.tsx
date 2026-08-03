import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import DashboardPage from '../page'

// Mock Next.js link
vi.mock('next/link', () => ({
  default: 'a',
}))

// Mock getUserId
vi.mock('@/lib/user-id', () => ({
  getUserId: () => 'test-user',
}))

// Mock dashboard child components
vi.mock('@/components/dashboard/forgetting-curve-list', () => ({
  ForgettingCurveList: ({ isLoading }: { isLoading: boolean }) => (
    <div data-testid="forgetting-curve-list">
      {isLoading ? '加载中...' : '遗忘曲线列表'}
    </div>
  ),
}))

vi.mock('@/components/dashboard/review-plan', () => ({
  ReviewPlan: ({ isLoading }: { isLoading: boolean }) => (
    <div data-testid="review-plan">
      {isLoading ? '加载中...' : '复习计划'}
    </div>
  ),
}))

vi.mock('@/components/dashboard/mastery-distribution', () => ({
  MasteryDistribution: ({ isLoading }: { isLoading: boolean }) => (
    <div data-testid="mastery-distribution">
      {isLoading ? '加载中...' : '掌握分布'}
    </div>
  ),
}))

vi.mock('@/components/dashboard/activity-heatmap', () => ({
  ActivityHeatmap: ({ isLoading }: { isLoading: boolean }) => (
    <div data-testid="activity-heatmap">
      {isLoading ? '加载中...' : '活动热力图'}
    </div>
  ),
}))

vi.mock('@/components/dashboard/learning-trend', () => ({
  LearningTrend: ({ isLoading }: { isLoading: boolean }) => (
    <div data-testid="learning-trend">
      {isLoading ? '加载中...' : '学习趋势'}
    </div>
  ),
}))

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

describe('DashboardPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default: return empty dashboard data
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        alerts: [],
        alert_count: 0,
        mastery_distribution: { urgent: 0, warning: 0, ok: 0, unknown: 0 },
        total_kps_tracked: 0,
        average_recall: 0,
        all_states: [],
      }),
    })
  })

  it('renders the dashboard title', () => {
    render(<DashboardPage />)
    expect(screen.getByText('📊 学习仪表盘')).toBeTruthy()
  })

  it('renders child components after data loads', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        alerts: [{ kp_id: 'kp1', kp_name: '数组', alert_level: 'urgent', recall_probability: 0.3 }],
        alert_count: 1,
        mastery_distribution: { urgent: 1, warning: 2, ok: 5, unknown: 0 },
        total_kps_tracked: 8,
        average_recall: 0.65,
        all_states: [],
      }),
    })

    render(<DashboardPage />)

    await waitFor(() => {
      expect(screen.getByText('8 个知识点追踪中 · 平均回忆率 65%')).toBeTruthy()
    })

    // Child components rendered
    expect(screen.getByTestId('forgetting-curve-list')).toBeTruthy()
    expect(screen.getByTestId('review-plan')).toBeTruthy()
    expect(screen.getByTestId('mastery-distribution')).toBeTruthy()
    expect(screen.getByTestId('activity-heatmap')).toBeTruthy()
    expect(screen.getByTestId('learning-trend')).toBeTruthy()
  })

  it('renders summary cards with correct data', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        alerts: [],
        alert_count: 3,
        mastery_distribution: { urgent: 1, warning: 2, ok: 5, unknown: 0 },
        total_kps_tracked: 8,
        average_recall: 0.65,
        all_states: [],
      }),
    })

    render(<DashboardPage />)

    await waitFor(() => {
      expect(screen.getByText('3')).toBeTruthy()  // 待复习
      expect(screen.getByText('5')).toBeTruthy()  // 已掌握
    })
  })

  it('renders error state when API fails', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'))

    render(<DashboardPage />)

    await waitFor(() => {
      expect(screen.getByText(/加载失败/)).toBeTruthy()
    })
  })

  it('renders "继续学习" and "刷新" buttons', () => {
    render(<DashboardPage />)
    expect(screen.getByText('继续学习')).toBeTruthy()
    expect(screen.getByText('刷新')).toBeTruthy()
  })
})
