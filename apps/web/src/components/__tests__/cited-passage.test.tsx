import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CitedPassage, splitBySpan } from '../provenance/cited-passage'

/**
 * These tests focus on the failure modes that make provenance *look* like it
 * works while pointing at the wrong text — the reason this logic is a pure,
 * separately-tested function rather than inline JSX.
 */

describe('splitBySpan', () => {
  it('splits text around a valid span', () => {
    const parts = splitBySpan('hello world', 6, 11)
    expect(parts).toEqual({
      before: 'hello ',
      match: 'world',
      after: '',
      highlighted: true,
    })
  })

  it('keeps surrounding text when the span is in the middle', () => {
    const parts = splitBySpan('abcdefgh', 2, 5)
    expect(parts.before).toBe('ab')
    expect(parts.match).toBe('cde')
    expect(parts.after).toBe('fgh')
  })

  it('does not highlight when the span is null', () => {
    const parts = splitBySpan('hello', null, null)
    expect(parts.highlighted).toBe(false)
    expect(parts.before).toBe('hello')
    expect(parts.match).toBe('')
  })

  it('does not highlight when the span is undefined', () => {
    const parts = splitBySpan('hello', undefined, undefined)
    expect(parts.highlighted).toBe(false)
  })

  it('does NOT treat a null span as offset 0', () => {
    // The critical case: 0 would highlight the start of the document, which
    // looks like working provenance but points at the wrong passage.
    const parts = splitBySpan('hello world', null, null)
    expect(parts.match).not.toBe('hello')
    expect(parts.highlighted).toBe(false)
  })

  it('rejects an inverted span instead of clamping', () => {
    const parts = splitBySpan('hello', 4, 2)
    expect(parts.highlighted).toBe(false)
  })

  it('rejects a span past the end of the text', () => {
    const parts = splitBySpan('short', 0, 999)
    expect(parts.highlighted).toBe(false)
  })

  it('rejects a negative start', () => {
    const parts = splitBySpan('hello', -1, 3)
    expect(parts.highlighted).toBe(false)
  })

  it('rejects an empty span', () => {
    const parts = splitBySpan('hello', 2, 2)
    expect(parts.highlighted).toBe(false)
  })

  it('rejects a whitespace-only span', () => {
    const parts = splitBySpan('hello   world', 5, 8)
    expect(parts.highlighted).toBe(false)
  })

  it('handles a span covering the whole text', () => {
    const parts = splitBySpan('all of it', 0, 9)
    expect(parts.highlighted).toBe(true)
    expect(parts.match).toBe('all of it')
    expect(parts.before).toBe('')
    expect(parts.after).toBe('')
  })

  it('handles CJK text offsets by character', () => {
    // Offsets are character-based on the backend (Python str indexing), which
    // matches JS string indexing for these ranges in practice.
    const text = '数据结构与算法'
    const parts = splitBySpan(text, 0, 4)
    expect(parts.match).toBe('数据结构')
    expect(parts.after).toBe('与算法')
  })
})

describe('CitedPassage', () => {
  it('renders a mark element for a valid span', () => {
    render(<CitedPassage text="hello world" charStart={6} charEnd={11} />)
    const mark = screen.getByTestId('cited-passage')
    expect(mark.textContent).toBe('world')
  })

  it('renders plain text without a mark when the span is null', () => {
    render(<CitedPassage text="hello world" charStart={null} charEnd={null} />)
    expect(screen.queryByTestId('cited-passage')).toBeNull()
    expect(screen.getByText('hello world')).toBeTruthy()
  })

  it('renders plain text for an out-of-range span', () => {
    render(<CitedPassage text="short" charStart={0} charEnd={999} />)
    expect(screen.queryByTestId('cited-passage')).toBeNull()
  })

  it('preserves the full text when highlighting', () => {
    const { container } = render(<CitedPassage text="abcdef" charStart={2} charEnd={4} />)
    // The highlight splits the text across nodes, so assert on the container's
    // combined text rather than querying a single element.
    expect(container.textContent).toBe('abcdef')
    expect(screen.getByTestId('cited-passage').textContent).toBe('cd')
  })

  it('applies the className to the wrapper', () => {
    const { container } = render(
      <CitedPassage text="abc" charStart={0} charEnd={1} className="custom-class" />,
    )
    expect(container.querySelector('.custom-class')).toBeTruthy()
  })
})
