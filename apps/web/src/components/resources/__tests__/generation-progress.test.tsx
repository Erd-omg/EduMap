import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, act, fireEvent } from '@testing-library/react'
import { GenerationProgress } from '../generation-progress'

/**
 * Minimal EventSource stand-in. No real SSE infra exists in this repo, so we
 * stub the global with a class that records instances and lets tests emit
 * events synchronously.
 */
class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  closed = false
  onerror: ((ev: Event) => void) | null = null
  private listeners: Record<string, Array<(e: MessageEvent) => void>> = {}

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  addEventListener(type: string, fn: (e: MessageEvent) => void) {
    ;(this.listeners[type] ??= []).push(fn)
  }

  close() {
    this.closed = true
  }

  emit(type: string, data: unknown) {
    for (const fn of this.listeners[type] ?? []) {
      fn({ data: JSON.stringify(data) } as MessageEvent)
    }
  }
}

const mockFetch = vi.fn()

beforeEach(() => {
  MockEventSource.instances = []
  mockFetch.mockReset()
  mockFetch.mockResolvedValue({
    ok: true,
    json: async () => ({ overall_status: 'running', agent_results: {} }),
  })
  vi.stubGlobal('EventSource', MockEventSource)
  vi.stubGlobal('fetch', mockFetch)
})

/** Flush the status-fetch promise chain so restore logic has applied. */
async function flushAsync() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
}

describe('GenerationProgress', () => {
  it('shows running status and the cancel button when a session is in progress', async () => {
    render(<GenerationProgress sessionId="s1" onComplete={vi.fn()} />)

    expect(screen.getByText('生成中...')).toBeTruthy()
    expect(screen.getByRole('button', { name: '取消生成' })).toBeTruthy()
  })

  it('restores a cancelled session, closes the EventSource, and does not call onComplete', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        overall_status: 'cancelled',
        agent_results: { planner: { n: 1 } },
      }),
    })
    const onComplete = vi.fn()
    render(<GenerationProgress sessionId="s1" onComplete={onComplete} />)

    await screen.findByText('已取消')
    expect(MockEventSource.instances[0].closed).toBe(true)
    expect(onComplete).not.toHaveBeenCalled()
  })

  it('workflow_complete with status cancelled sets cancelled (not completed) and calls onComplete', async () => {
    const onComplete = vi.fn()
    render(<GenerationProgress sessionId="s1" onComplete={onComplete} />)
    await flushAsync()
    const es = MockEventSource.instances[0]

    act(() => es.emit('workflow_complete', { status: 'cancelled' }))

    await screen.findByText('已取消')
    expect(screen.queryByText('已完成')).toBeNull()
    expect(es.closed).toBe(true)
    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('workflow_complete with status completed calls onComplete (success path regression)', async () => {
    const onComplete = vi.fn()
    render(<GenerationProgress sessionId="s1" onComplete={onComplete} />)
    await flushAsync()
    const es = MockEventSource.instances[0]

    act(() => es.emit('workflow_complete', { status: 'completed' }))

    await waitFor(() => expect(onComplete).toHaveBeenCalledTimes(1))
    // Header '已完成' + all 6 agents marked completed = 7 occurrences.
    expect(screen.getAllByText('已完成').length).toBe(7)
  })

  it('handleCancel fires the cancel endpoint, closes SSE, sets cancelled, and calls onComplete', async () => {
    const onComplete = vi.fn()
    render(<GenerationProgress sessionId="s1" onComplete={onComplete} />)
    await screen.findByText('生成中...')
    const es = MockEventSource.instances[0]

    fireEvent.click(screen.getByRole('button', { name: '取消生成' }))

    await waitFor(() =>
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/orchestrator/cancel/s1'),
        expect.objectContaining({ method: 'POST' }),
      ),
    )
    await screen.findByText('已取消')
    expect(es.closed).toBe(true)
    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('double-click on cancel sends only one request', async () => {
    render(<GenerationProgress sessionId="s1" onComplete={vi.fn()} />)
    await screen.findByText('生成中...')
    const btn = screen.getByRole('button', { name: '取消生成' })

    fireEvent.click(btn)
    fireEvent.click(btn)

    await waitFor(() => {
      const cancelCalls = mockFetch.mock.calls.filter(([url]) =>
        String(url).includes('/cancel/s1'),
      )
      expect(cancelCalls.length).toBe(1)
    })
  })

  it('updates agent statuses on phase_change / agent_complete', async () => {
    render(<GenerationProgress sessionId="s1" onComplete={vi.fn()} />)
    await flushAsync()
    const es = MockEventSource.instances[0]

    act(() => es.emit('phase_change', { agent: 'planner', phase: 'EXTRACT' }))
    expect(screen.getAllByText('进行中').length).toBe(1)

    act(() => es.emit('agent_complete', { agent: 'planner', phase: 'EXTRACT' }))
    expect(screen.getAllByText('已完成').length).toBe(1)
  })

  it('renders per-agent trace (duration + tool calls) from agent_complete', async () => {
    render(<GenerationProgress sessionId="s1" onComplete={vi.fn()} />)
    await flushAsync()
    const es = MockEventSource.instances[0]

    act(() =>
      es.emit('agent_complete', {
        agent: 'planner',
        phase: 'EXTRACT',
        report: {
          duration_ms: 6982.96,
          tool_calls: [
            { tool: 'knowledge_graph_search', success: true, duration_ms: 7.97 },
          ],
        },
      }),
    )

    const trace = screen.getByTestId('agent-trace-planner')
    expect(trace.textContent).toContain('7.0s') // 6982.96ms
    expect(trace.textContent).toContain('knowledge_graph_search')
    expect(trace.textContent).toContain('8ms') // 7.97ms rounded
  })

  it('restores the trace from /status on refresh', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        overall_status: 'completed',
        agent_results: {
          planner: {
            _report: {
              duration_ms: 1234,
              tool_calls: [{ tool: 'resource_search', success: true }],
            },
          },
        },
      }),
    })

    render(<GenerationProgress sessionId="s1" onComplete={vi.fn()} />)
    await flushAsync()

    const trace = screen.getByTestId('agent-trace-planner')
    expect(trace.textContent).toContain('1.2s')
    expect(trace.textContent).toContain('resource_search')
  })

  it('omits the trace when no report is present', async () => {
    render(<GenerationProgress sessionId="s1" onComplete={vi.fn()} />)
    await flushAsync()
    const es = MockEventSource.instances[0]

    act(() => es.emit('agent_complete', { agent: 'planner', phase: 'EXTRACT' }))
    expect(screen.queryByTestId('agent-trace-planner')).toBeNull()
  })
})
