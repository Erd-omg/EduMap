import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { SourcePanel } from '../mentor/source-panel'
import type { MentorSource } from '@/stores/chat-store'

const mockSources: MentorSource[] = [
  { id: '1', name: '数组详解', type: 'chroma', score: 0.92, summary: '数组插入复杂度O(n)' },
  { id: '2', name: '链表基础', type: 'neo4j', score: 0.85, summary: '链表插入复杂度O(1)' },
]

describe('SourcePanel', () => {
  it('renders with sources', () => {
    render(<SourcePanel sources={mockSources} />)
    expect(screen.getByText('数组详解')).toBeTruthy()
    expect(screen.getByText('链表基础')).toBeTruthy()
  })

  it('shows confidence scores', () => {
    render(<SourcePanel sources={mockSources} />)
    expect(screen.getByText('92%')).toBeTruthy()
  })

  it('shows source type labels', () => {
    render(<SourcePanel sources={mockSources} />)
    expect(screen.getByText('向量搜索')).toBeTruthy()
    expect(screen.getByText('知识图谱')).toBeTruthy()
  })

  it('renders source names', () => {
    render(<SourcePanel sources={mockSources} />)
    expect(screen.getByText('数组详解')).toBeTruthy()
  })

  it('renders empty state', () => {
    const { container } = render(<SourcePanel sources={[]} />)
    expect(container.textContent).toBeTruthy()
  })
})
