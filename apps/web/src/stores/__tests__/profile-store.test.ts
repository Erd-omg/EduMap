import { describe, it, expect, beforeEach } from 'vitest'
import { useProfileStore } from '../profile-store'
import type { UserProfile } from '@edumap/shared-types'

const mockProfile = {
  learning_ability: { comprehension: 0.8 },
  learning_motivation: { intrinsic: 0.7 },
} as unknown as UserProfile

describe('ProfileStore', () => {
  beforeEach(() => {
    useProfileStore.setState({
      profile: null,
      confidenceScores: {},
      conversationHistory: [],
      isAnalyzing: false,
    })
  })

  it('starts with default state', () => {
    const state = useProfileStore.getState()
    expect(state.profile).toBeNull()
    expect(state.confidenceScores).toEqual({})
    expect(state.conversationHistory).toEqual([])
    expect(state.isAnalyzing).toBe(false)
  })

  it('updateProfile sets profile and optional confidence', () => {
    useProfileStore.getState().updateProfile(mockProfile, { learning_ability: 0.9 })
    const state = useProfileStore.getState()
    expect(state.profile).toEqual(mockProfile)
    expect(state.confidenceScores.learning_ability).toBe(0.9)
  })

  it('updateProfile preserves existing confidence when not provided', () => {
    useProfileStore.getState().updateProfile(mockProfile, { learning_ability: 0.9 })
    useProfileStore.getState().updateProfile(mockProfile)
    expect(useProfileStore.getState().confidenceScores.learning_ability).toBe(0.9)
  })

  it('addMessage appends to conversation history', () => {
    useProfileStore.getState().addMessage('user', 'Hello')
    useProfileStore.getState().addMessage('assistant', 'Hi there')

    const history = useProfileStore.getState().conversationHistory
    expect(history).toHaveLength(2)
    expect(history[0].role).toBe('user')
    expect(history[0].content).toBe('Hello')
    expect(history[1].role).toBe('assistant')
  })

  it('setAnalyzing toggles the flag', () => {
    useProfileStore.getState().setAnalyzing(true)
    expect(useProfileStore.getState().isAnalyzing).toBe(true)

    useProfileStore.getState().setAnalyzing(false)
    expect(useProfileStore.getState().isAnalyzing).toBe(false)
  })

  it('reset clears all state', () => {
    useProfileStore.getState().updateProfile(mockProfile)
    useProfileStore.getState().addMessage('user', 'Hello')
    useProfileStore.getState().setAnalyzing(true)

    useProfileStore.getState().reset()

    const state = useProfileStore.getState()
    expect(state.profile).toBeNull()
    expect(state.conversationHistory).toEqual([])
    expect(state.isAnalyzing).toBe(false)
  })

  it('updateConfidence replaces scores', () => {
    useProfileStore.getState().updateConfidence({ dim1: 0.7 })
    // updateConfidence replaces, not merges
    useProfileStore.getState().updateConfidence({ dim2: 0.8 })

    const scores = useProfileStore.getState().confidenceScores
    expect(scores.dim2).toBe(0.8)
    // dim1 was replaced
    expect(scores.dim1).toBeUndefined()
  })

  describe('syncToBackend', () => {
    beforeEach(() => {
      // reset fetch mock between tests
      ;(globalThis as unknown as { fetch: unknown }).fetch = vi.fn()
    })

    it('does nothing when profile is null', async () => {
      const fetchMock = vi.fn()
      ;(globalThis as unknown as { fetch: unknown }).fetch = fetchMock

      await useProfileStore.getState().syncToBackend()
      expect(fetchMock).not.toHaveBeenCalled()
    })

    it('PUTs profile + confidence_scores to profile-service', async () => {
      const fetchMock = vi.fn().mockResolvedValue({ ok: true })
      ;(globalThis as unknown as { fetch: unknown }).fetch = fetchMock

      useProfileStore.getState().updateProfile(mockProfile, {
        learning_ability: 0.85,
      })

      await useProfileStore.getState().syncToBackend()

      expect(fetchMock).toHaveBeenCalledTimes(1)
      const [url, init] = fetchMock.mock.calls[0]
      expect(url).toContain('/api/v1/profiles/')
      expect(init.method).toBe('PUT')

      const body = JSON.parse(init.body as string)
      expect(body.profile_data.learning_ability).toEqual({ comprehension: 0.8 })
      expect(body.confidence_scores).toEqual({ learning_ability: 0.85 })
      // 缺省 notifications_enabled 视为 true（首次同步时 settings 还没动过）
      expect(body.profile_data.notifications_enabled).toBe(true)
    })

    it('preserves notifications_enabled set elsewhere in the dict', async () => {
      const fetchMock = vi.fn().mockResolvedValue({ ok: true })
      ;(globalThis as unknown as { fetch: unknown }).fetch = fetchMock

      // 模拟 SSE 推过来的 profile 中已经带了关闭开关
      const profileWithToggle = {
        ...mockProfile,
        notifications_enabled: false,
      } as unknown as UserProfile
      useProfileStore.getState().updateProfile(profileWithToggle)

      await useProfileStore.getState().syncToBackend()

      const body = JSON.parse(fetchMock.mock.calls[0][1].body as string)
      expect(body.profile_data.notifications_enabled).toBe(false)
    })

    it('warns on non-2xx responses without throwing', async () => {
      const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 502 })
      ;(globalThis as unknown as { fetch: unknown }).fetch = fetchMock
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

      useProfileStore.getState().updateProfile(mockProfile)

      // 同步是 best-effort：非 2xx 不抛出，但必须留下可观测的 warn
      await expect(useProfileStore.getState().syncToBackend()).resolves.toBeUndefined()
      expect(warnSpy).toHaveBeenCalledWith(
        '[profile-store] syncToBackend non-2xx:',
        502,
      )
      warnSpy.mockRestore()
    })

    it('does not throw when fetch fails (offline best-effort)', async () => {
      const fetchMock = vi.fn().mockRejectedValue(new Error('offline'))
      ;(globalThis as unknown as { fetch: unknown }).fetch = fetchMock
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

      useProfileStore.getState().updateProfile(mockProfile)

      await expect(useProfileStore.getState().syncToBackend()).resolves.toBeUndefined()
      expect(warnSpy).toHaveBeenCalled()
      warnSpy.mockRestore()
    })
  })
})
