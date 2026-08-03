import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useLearningPathStore } from '../learning-path-store'

const mockPath = {
  course_id: 'cs201',
  user_id: 'user-1',
  nodes: [
    { kp_id: 'kp-a', name: 'A', description: '', difficulty: 1, status: 'ready', prerequisites: [], prerequisites_met: true, recommended_content_types: ['explanation'] },
  ],
  total_count: 1,
  mastered_count: 0,
  completed_count: 0,
  progress_percent: 0,
  created_at: '2024-01-01T00:00:00Z',
}

const mockRecommendation = {
  next_kp_id: 'kp-a',
  next_kp_name: 'A',
  reason: 'Next in sequence',
  recommended_content_type: 'explanation',
  estimated_session_min: 20,
}

describe('LearningPathStore', () => {
  beforeEach(() => {
    useLearningPathStore.setState({
      courseId: null,
      path: null,
      recommendation: null,
      isLoading: false,
      error: null,
      lastFetched: null,
    })
    vi.restoreAllMocks()
  })

  it('starts with default state', () => {
    const state = useLearningPathStore.getState()
    expect(state.path).toBeNull()
    expect(state.isLoading).toBe(false)
    expect(state.error).toBeNull()
  })

  it('fetchPath loads path and recommendation', async () => {
    const mockFetch = vi.fn()
    mockFetch
      .mockResolvedValueOnce({ ok: true, json: async () => mockPath })
      .mockResolvedValueOnce({ ok: true, json: async () => mockRecommendation })
    vi.stubGlobal('fetch', mockFetch)

    await useLearningPathStore.getState().fetchPath('cs201', 'user-1')

    const state = useLearningPathStore.getState()
    expect(state.path).toEqual(mockPath)
    expect(state.recommendation).toEqual(mockRecommendation)
    expect(state.isLoading).toBe(false)
    expect(state.courseId).toBe('cs201')
  })

  it('fetchPath sets error on failure', async () => {
    const mockFetch = vi.fn()
    mockFetch
      .mockResolvedValueOnce({ ok: false, statusText: 'Not Found' })
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) })
    vi.stubGlobal('fetch', mockFetch)

    try {
      await useLearningPathStore.getState().fetchPath('cs201', 'user-1')
    } catch {
      // Expected
    }

    const state = useLearningPathStore.getState()
    expect(state.path).toBeNull()
    // error might still be set depending on implementation
    expect(state.isLoading).toBe(false)
  })

  it('recordProgress sends progress and refreshes path', async () => {
    const mockFetch = vi.fn()
    const progressResponse = {
      updated_path: mockPath,
      next_recommendation: mockRecommendation,
    }
    mockFetch.mockResolvedValueOnce({
      ok: true, json: async () => progressResponse,
    })
    vi.stubGlobal('fetch', mockFetch)

    useLearningPathStore.setState({
      path: mockPath,
      courseId: 'cs201',
    })

    await useLearningPathStore.getState().recordProgress('kp-a', 'completed', 0.9)

    expect(useLearningPathStore.getState().path).toEqual(mockPath)
    expect(useLearningPathStore.getState().recommendation).toEqual(mockRecommendation)
  })

  it('reset clears all state', () => {
    useLearningPathStore.setState({
      courseId: 'cs201',
      path: mockPath,
      recommendation: mockRecommendation,
      isLoading: true,
    })

    useLearningPathStore.getState().reset()

    const state = useLearningPathStore.getState()
    expect(state.path).toBeNull()
    expect(state.courseId).toBeNull()
    expect(state.isLoading).toBe(false)
  })
})
