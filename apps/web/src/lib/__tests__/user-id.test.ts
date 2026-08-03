import { describe, it, expect, beforeEach } from 'vitest'
import { getUserId, clearUserId } from '../user-id'

describe('getUserId', () => {
  beforeEach(() => {
    localStorage.clear()
    clearUserId()
  })

  it('generates an ID on first call', () => {
    const id = getUserId()
    expect(id).toBeTruthy()
    expect(typeof id).toBe('string')
  })

  it('returns the same ID on subsequent calls', () => {
    const first = getUserId()
    const second = getUserId()
    expect(first).toBe(second)
  })

  it('stores the ID in localStorage', () => {
    const id = getUserId()
    const stored = localStorage.getItem('edumap-user-id')
    expect(stored).toBe(id)
  })

  it('clearUserId removes the stored ID', () => {
    getUserId()
    clearUserId()
    const stored = localStorage.getItem('edumap-user-id')
    expect(stored).toBeNull()
  })

  it('generates a different ID after clearUserId', () => {
    const first = getUserId()
    clearUserId()
    localStorage.clear() // clear storage too
    const second = getUserId()
    expect(first).not.toBe(second)
  })

  it('returns string of reasonable length', () => {
    const id = getUserId()
    expect(id.length).toBeGreaterThan(5)
  })
})
