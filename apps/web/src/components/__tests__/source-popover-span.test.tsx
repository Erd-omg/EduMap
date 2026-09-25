import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SourcePopover } from '../provenance/source-popover'
import type { MentorSource } from '@/stores/chat-store'

/**
 * Span-level provenance in the citation popover.
 *
 * The behaviour under test: clicking a source that carries a character span
 * must open the original material **positioned at the cited passage**, not
 * merely list the document from the top. Without that, the offsets the backend
 * now stores buy the user nothing.
 */

const SOURCE_WITH_SPAN: MentorSource = {
  id: 'chunk-1',
  name: '数组详解',
  type: 'chroma',
  score: 0.9,
  summary: '数组插入复杂度O(n)',
  resource_id: 'res-1',
  char_start: 10,
  char_end: 20,
}

const SOURCE_NO_SPAN: MentorSource = {
  id: 'chunk-2',
  name: '链表基础',
  type: 'chroma',
  score: 0.8,
  summary: '链表插入复杂度O(1)',
  resource_id: 'res-1',
  char_start: null,
  char_end: null,
}

function mockFetch(chunks: unknown[]) {
  return vi.fn((url: string) => {
    if (url.includes('/chunks')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ resource_id: 'res-1', total_chunks: chunks.length, chunks }),
      })
    }
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ id: 'res-1', name: '数组详解', type: 'upload' }),
    })
  })
}

describe('SourcePopover span highlighting', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('opens the resource and highlights the cited chunk', async () => {
    const chunks = [
      { index: 0, text_preview: 'aaaa', char_count: 4, char_start: 0, char_end: 4 },
      // the cited chunk: offsets 10..20 in the document
      { index: 1, text_preview: 'bbbbbbbbbb', char_count: 10, char_start: 10, char_end: 20 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    render(<SourcePopover sources={[SOURCE_WITH_SPAN]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByTestId('cited-chunk')).toBeTruthy()
    })
    // the cited chunk is marked, the other is not
    expect(screen.getAllByTestId('cited-chunk')).toHaveLength(1)
  })

  it('renders a highlight mark inside the cited chunk', async () => {
    const chunks = [
      { index: 0, text_preview: 'hello world foo', char_count: 15, char_start: 0, char_end: 15 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    const source = { ...SOURCE_WITH_SPAN, char_start: 6, char_end: 11 }
    render(<SourcePopover sources={[source]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      const mark = screen.getByTestId('cited-passage')
      expect(mark.textContent).toBe('world')
    })
  })

  it('does not mark any chunk when the source has no span', async () => {
    const chunks = [
      { index: 0, text_preview: 'aaaa', char_count: 4, char_start: 0, char_end: 4 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    render(<SourcePopover sources={[SOURCE_NO_SPAN]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('链表基础'))

    await waitFor(() => {
      expect(screen.getByText(/文本段落/)).toBeTruthy()
    })
    expect(screen.queryByTestId('cited-chunk')).toBeNull()
    expect(screen.queryByTestId('cited-passage')).toBeNull()
  })

  it('falls back to plain listing when the span matches no chunk', async () => {
    const chunks = [
      { index: 0, text_preview: 'aaaa', char_count: 4, char_start: 0, char_end: 4 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    // span points outside every chunk
    const source = { ...SOURCE_WITH_SPAN, char_start: 999, char_end: 1000 }
    render(<SourcePopover sources={[source]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByText(/文本段落/)).toBeTruthy()
    })
    expect(screen.queryByTestId('cited-chunk')).toBeNull()
  })

  it('shows the citation hint only when a chunk is cited', async () => {
    const chunks = [
      { index: 0, text_preview: 'bbbb', char_count: 4, char_start: 10, char_end: 14 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    render(<SourcePopover sources={[SOURCE_WITH_SPAN]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByText(/已定位到被引用的段落/)).toBeTruthy()
    })
  })

  it('gives chunks priority over description when a span was clicked', async () => {
    // A description would otherwise hide the highlight entirely — the user
    // asked to see the passage, so the passage must win.
    const chunks = [
      { index: 0, text_preview: 'hello world', char_count: 11, char_start: 0, char_end: 11 },
    ]
    const fetchMock = vi.fn((url: string) => {
      if (url.includes('/chunks')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ resource_id: 'res-1', total_chunks: 1, chunks }),
        })
      }
      return Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            id: 'res-1',
            name: '数组详解',
            type: 'upload',
            description: '这是一段摘要描述，不含被引用原文',
          }),
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    const source = { ...SOURCE_WITH_SPAN, char_start: 0, char_end: 5 }
    render(<SourcePopover sources={[source]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByTestId('cited-chunk')).toBeTruthy()
    })
    expect(screen.queryByText(/这是一段摘要描述/)).toBeNull()
  })

  it('does not highlight when the cited passage is past the 200-char preview', async () => {
    // `text_preview` is only the chunk's first 200 chars, while offsets are
    // document-absolute. A passage at offset 400 of its chunk is simply not in
    // this string — clamping the offset into range (which the first version
    // did) would highlight the tail of the preview, i.e. the wrong text.
    const chunks = [
      { index: 0, text_preview: 'x'.repeat(200), char_count: 512, char_start: 1000, char_end: 1512 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    // cited at document offset 1400 = chunk-relative 400, past the preview
    const source = { ...SOURCE_WITH_SPAN, char_start: 1400, char_end: 1450 }
    render(<SourcePopover sources={[source]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByTestId('cited-chunk')).toBeTruthy()
    })
    // the chunk is marked as the citation, but nothing is highlighted, because
    // the passage is not present in the preview
    expect(screen.queryByTestId('cited-passage')).toBeNull()
    expect(screen.getByText(/超出此处 200 字预览范围/)).toBeTruthy()
  })

  it('highlights when the cited passage is inside the preview', async () => {
    const chunks = [
      { index: 0, text_preview: 'hello world and more', char_count: 20, char_start: 500, char_end: 520 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    // document offsets 506..511 = chunk-relative 6..11 = "world"
    const source = { ...SOURCE_WITH_SPAN, char_start: 506, char_end: 511 }
    render(<SourcePopover sources={[source]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByTestId('cited-passage').textContent).toBe('world')
    })
  })

  it('shows the page number on the cited chunk when known', async () => {
    const chunks = [
      {
        index: 0, text_preview: 'hello world', char_count: 11,
        char_start: 0, char_end: 11, page_number: 7,
      },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    const source = { ...SOURCE_WITH_SPAN, char_start: 0, char_end: 5, page_number: 7 }
    render(<SourcePopover sources={[source]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByText(/第 7 页/)).toBeTruthy()
    })
  })

  it('shows no page label for formats without pages', async () => {
    const chunks = [
      { index: 0, text_preview: 'hello world', char_count: 11, char_start: 0, char_end: 11 },
    ]
    vi.stubGlobal('fetch', mockFetch(chunks))

    const source = { ...SOURCE_WITH_SPAN, char_start: 0, char_end: 5 }
    render(<SourcePopover sources={[source]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('数组详解'))

    await waitFor(() => {
      expect(screen.getByTestId('cited-chunk')).toBeTruthy()
    })
    expect(screen.queryByText(/第 \d+ 页/)).toBeNull()
  })

  it('still shows the description when no span was clicked', async () => {
    const chunks: unknown[] = []
    const fetchMock = vi.fn((url: string) => {
      if (url.includes('/chunks')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ resource_id: 'res-1', total_chunks: 0, chunks }),
        })
      }
      return Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            id: 'res-1',
            name: '数组详解',
            type: 'explanation',
            description: '系统生成的讲解内容',
          }),
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<SourcePopover sources={[SOURCE_NO_SPAN]} />)
    await userEvent.click(screen.getByTitle('查看来源'))
    await userEvent.click(screen.getByText('链表基础'))

    await waitFor(() => {
      expect(screen.getByText('系统生成的讲解内容')).toBeTruthy()
    })
  })
})
