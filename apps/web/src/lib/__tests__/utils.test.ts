import { describe, it, expect } from 'vitest'
import { cn } from '../utils'

describe('cn', () => {
  it('merges class names', () => {
    expect(cn('foo', 'bar')).toBe('foo bar')
  })

  it('handles conditional classes', () => {
    expect(cn('base', false && 'hidden', 'visible')).toBe('base visible')
  })

  it('handles undefined and null', () => {
    expect(cn('a', undefined, null, 'b')).toBe('a b')
  })

  it('returns empty string for no inputs', () => {
    expect(cn()).toBe('')
  })

  it('handles tailwind conflicts via twMerge', () => {
    expect(cn('px-2', 'px-4')).toBe('px-4')
  })
})
