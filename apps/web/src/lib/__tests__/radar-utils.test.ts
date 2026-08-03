import { describe, it, expect } from 'vitest'
import { buildRadarData, type RadarDataPoint } from '../radar-utils'
import type { UserProfile } from '@edumap/shared-types'

const emptyProfile: UserProfile = {} as UserProfile

const fullProfile: UserProfile = {
  learning_ability: { comprehension: 0.8, problem_solving: 0.7 },
  learning_motivation: { intrinsic: 0.9, extrinsic: 0.6 },
  knowledge_coverage: { mastered: ['a', 'b'], learning: ['c'], not_started: ['d'] },
  interaction_style: { visual: 0.9, textual: 0.3 },
  focus_characteristics: { attention_span: 0.6, concentration: 0.5 },
  knowledge_base: { algebra: 0.8, calculus: 0.7, geometry: 0.6 },
} as unknown as UserProfile

describe('buildRadarData', () => {
  it('returns 6 dimensions', () => {
    const result = buildRadarData(fullProfile, {})
    expect(result).toHaveLength(6)
  })

  it('has correct structure for each data point', () => {
    const result = buildRadarData(fullProfile, {})
    for (const point of result) {
      expect(point).toHaveProperty('dimension')
      expect(point).toHaveProperty('key')
      expect(point).toHaveProperty('score')
      expect(point).toHaveProperty('confidence')
    }
  })

  it('computes knowledge_coverage as mastered / total', () => {
    const result = buildRadarData(fullProfile, {})
    const kc = result.find((r) => r.key === 'knowledge_coverage')
    expect(kc?.score).toBeCloseTo(0.5) // 2 / 4
  })

  it('computes knowledge_base as average of numeric values', () => {
    const result = buildRadarData(fullProfile, {})
    const kb = result.find((r) => r.key === 'knowledge_base')
    expect(kb?.score).toBeCloseTo((0.8 + 0.7 + 0.6) / 3)
  })

  it('uses confidence from the confidence map', () => {
    const confidence = { learning_ability: 0.9, knowledge_coverage: 0.7 }
    const result = buildRadarData(fullProfile, confidence)
    expect(result.find((r) => r.key === 'learning_ability')?.confidence).toBe(0.9)
    expect(result.find((r) => r.key === 'knowledge_coverage')?.confidence).toBe(0.7)
  })

  it('defaults confidence to 0.5 when not provided', () => {
    const result = buildRadarData(fullProfile, {})
    for (const point of result) {
      expect(point.confidence).toBe(0.5)
    }
  })

  it('clamps scores to [0, 1]', () => {
    const overRangeProfile: UserProfile = {
      ...fullProfile,
      learning_ability: { comprehension: 2.5, problem_solving: -1 },
    } as unknown as UserProfile
    const result = buildRadarData(overRangeProfile, {})
    const la = result.find((r) => r.key === 'learning_ability')
    expect(la?.score).toBeGreaterThanOrEqual(0)
    expect(la?.score).toBeLessThanOrEqual(1)
  })

  it('returns zeros for empty profile', () => {
    const result = buildRadarData(emptyProfile, {})
    for (const point of result) {
      expect(point.score).toBe(0)
    }
  })
})
