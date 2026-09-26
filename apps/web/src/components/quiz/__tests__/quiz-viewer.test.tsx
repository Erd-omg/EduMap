import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QuizViewer } from '../quiz-viewer'

/**
 * The quiz viewer must not grade. It used to compare each answer against
 * `q.correct_answer` and hand the caller a score — which meant the answer key
 * had to be in the browser, and a learner could read it out of devtools. The
 * component now reports raw answers only; the server grades.
 *
 * These tests pin that separation, because reintroducing a local comparison is
 * exactly how the leak comes back.
 */

const QUESTIONS = [
  {
    id: 'q1',
    type: 'choice' as const,
    content: '数组和链表在插入上哪个更快？',
    options: ['数组', '链表'],
    knowledge_point_id: 'kp-array',
  },
  {
    id: 'q2',
    type: 'choice' as const,
    content: '栈是什么顺序？',
    options: ['LIFO', 'FIFO'],
    knowledge_point_id: 'kp-stack',
  },
]

describe('QuizViewer', () => {
  it('renders without any answer key present', () => {
    render(
      <QuizViewer questions={QUESTIONS} kpName="数组" onSubmit={vi.fn()} onSkip={vi.fn()} />
    )
    // The fixtures carry no `correct_answer` — rendering must still work,
    // proving the component no longer depends on one.
    expect(QUESTIONS.every((q) => !('correct_answer' in q))).toBe(true)
    expect(screen.getByText('数组和链表在插入上哪个更快？')).toBeTruthy()
  })

  it('submits the raw answers and no locally computed score', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(
      <QuizViewer questions={QUESTIONS} kpName="数组" onSubmit={onSubmit} onSkip={vi.fn()} />
    )

    await user.click(screen.getByText('链表'))
    await user.click(screen.getByRole('button', { name: '下一题' }))
    await user.click(screen.getByText('LIFO'))
    await user.click(screen.getByRole('button', { name: '提交答案' }))

    expect(onSubmit).toHaveBeenCalledTimes(1)
    // Exactly one argument: the answers. A second argument would mean the
    // component is computing a score again — the thing this change removed.
    expect(onSubmit.mock.calls[0]).toHaveLength(1)
    expect(onSubmit.mock.calls[0][0]).toEqual({ q1: '链表', q2: 'LIFO' })
  })

  it('does not reveal the answer after submitting', async () => {
    const user = userEvent.setup()
    render(
      <QuizViewer questions={QUESTIONS} kpName="数组" onSubmit={vi.fn()} onSkip={vi.fn()} />
    )
    await user.click(screen.getByText('链表'))
    await user.click(screen.getByRole('button', { name: '下一题' }))
    await user.click(screen.getByText('LIFO'))
    await user.click(screen.getByRole('button', { name: '提交答案' }))

    // Nothing on screen tells the learner which option was right.
    expect(screen.queryByText(/正确答案/)).toBeNull()
  })
})
