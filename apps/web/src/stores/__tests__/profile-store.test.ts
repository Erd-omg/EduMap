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
})
