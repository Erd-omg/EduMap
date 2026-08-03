import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import LearnPage from '../page'

// Mock Next.js hooks
vi.mock('next/navigation', () => ({
  useParams: () => ({ courseId: 'cs201' }),
}))

vi.mock('next/link', () => ({
  default: 'a',
}))

// Mock getUserId
vi.mock('@/lib/user-id', () => ({
  getUserId: () => 'test-user',
}))

// Hoisted: store mock fn that can be referenced by vi.mock factory
const { mockLearningPathStore } = vi.hoisted(() => ({
  mockLearningPathStore: vi.fn(),
}))

// Mock learning path store — must provide getState() as a static method
vi.mock('@/stores/learning-path-store', () => ({
  useLearningPathStore: Object.assign(
    () => mockLearningPathStore(),
    { getState: () => mockLearningPathStore() }
  ),
}))

// Mock child components
vi.mock('@/components/knowledge-graph/skill-tree-canvas', () => ({
  SkillTreeCanvas: () => <div data-testid="skill-tree-canvas">技能树</div>,
}))

vi.mock('@/components/knowledge-graph/progress-slider', () => ({
  ProgressSlider: () => <div data-testid="progress-slider">进度滑块</div>,
}))

vi.mock('@/components/knowledge-graph/progress-panel', () => ({
  ProgressPanel: ({ masteredCount, learningCount, nextKpName }: any) => (
    <div data-testid="progress-panel">
      进度面板 (已掌握: {masteredCount}, 学习中: {learningCount})
      {nextKpName && <span>下一个: {nextKpName}</span>}
    </div>
  ),
}))

vi.mock('@/components/resources/resource-viewer', () => ({
  ResourceViewer: () => <div data-testid="resource-viewer">资源查看器</div>,
}))

vi.mock('@/components/quiz/quiz-viewer', () => ({
  QuizViewer: () => <div data-testid="quiz-viewer">测验</div>,
}))

vi.mock('@/components/quiz/quiz-result', () => ({
  QuizResult: () => <div data-testid="quiz-result">测验结果</div>,
}))

vi.mock('@/components/knowledge-graph/demo-console', () => ({
  DemoConsole: () => <div data-testid="demo-console">演示控制台</div>,
}))

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

describe('LearnPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()

    // Default store mock: path loaded
    mockLearningPathStore.mockReturnValue({
      path: {
        nodes: [
          { kp_id: 'kp1', name: '数组', status: 'completed', description: '数组概念', difficulty: 2 },
          { kp_id: 'kp2', name: '链表', status: 'in_progress', description: '链表概念', difficulty: 3 },
          { kp_id: 'kp3', name: '栈', status: 'ready', description: '栈概念', difficulty: 2 },
          { kp_id: 'kp4', name: '队列', status: 'locked', description: '队列概念', difficulty: 3 },
        ],
        total_count: 4,
        course_id: 'cs201',
        user_id: 'test-user',
        mastered_count: 1,
        completed_count: 1,
        progress_percent: 25,
        created_at: new Date().toISOString(),
      },
      recommendation: { next_kp_id: 'kp2', next_kp_name: '链表', reason: '前置完成' },
      isLoading: false,
      error: null,
      fetchPath: vi.fn(),
      recordProgress: vi.fn(),
      lastFetched: Date.now(),
    })

    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ resources: [] }),
    })
  })

  it('renders the course ID as title', () => {
    render(<LearnPage />)
    expect(screen.getByText('cs201')).toBeTruthy()
  })

  it('renders navigation links', () => {
    render(<LearnPage />)
    const generateLink = screen.getByText('生成资源')
    expect(generateLink.closest('a')?.getAttribute('href')).toBe('/generate?courseId=cs201')

    const chatLink = screen.getByText('对话画像')
    expect(chatLink.closest('a')?.getAttribute('href')).toBe('/chat')
  })

  it('renders all main child components when path is loaded', () => {
    render(<LearnPage />)
    expect(screen.getByTestId('progress-slider')).toBeTruthy()
    expect(screen.getByTestId('progress-panel')).toBeTruthy()
    expect(screen.getByTestId('skill-tree-canvas')).toBeTruthy()
    expect(screen.getByTestId('resource-viewer')).toBeTruthy()
  })

  it('shows loading state when isLoading is true', () => {
    mockLearningPathStore.mockReturnValue({
      path: null,
      recommendation: null,
      isLoading: true,
      error: null,
      fetchPath: vi.fn(),
      recordProgress: vi.fn(),
      lastFetched: 0,
    })

    render(<LearnPage />)
    expect(screen.getByText('加载学习路径...')).toBeTruthy()
  })

  it('shows error state when error is present', () => {
    mockLearningPathStore.mockReturnValue({
      path: null,
      recommendation: null,
      isLoading: false,
      error: 'Failed to fetch',
      fetchPath: vi.fn(),
      recordProgress: vi.fn(),
      lastFetched: 0,
    })

    render(<LearnPage />)
    expect(screen.getByText(/加载失败: Failed to fetch/)).toBeTruthy()
  })

  it('renders demo mode toggle button', () => {
    render(<LearnPage />)
    expect(screen.getByText('🎮 演示')).toBeTruthy()
  })
})
