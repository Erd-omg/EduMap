import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QuizResult, type GradedQuestionData } from '../quiz-result'
import type { QuizQuestionData } from '../quiz-viewer'

/**
 * The result view must render the *server's* verdict, not a local comparison.
 *
 * It used to recompute correctness by comparing each answer to
 * `q.correct_answer`, which required the key in the browser. Now it reads the
 * `graded` detail returned by `POST /quiz/grade`, where the server has already
 * decided — and where the key is disclosed only because the attempt is scored.
 */

const QUESTIONS: QuizQuestionData[] = [
  { id: 'q1', type: 'choice', content: '数组插入的时间复杂度？', options: ['O(1)', 'O(n)'], knowledge_point_id: 'kp-array' },
  { id: 'q2', type: 'choice', content: '栈的顺序？', options: ['LIFO', 'FIFO'], knowledge_point_id: 'kp-stack' },
]

const GRADED: GradedQuestionData[] = [
  { id: 'q1', correct: true, difficulty: 3, submitted: 'O(n)', correct_answer: 'O(n)' },
  { id: 'q2', correct: false, difficulty: 3, submitted: 'FIFO', correct_answer: 'LIFO' },
]

describe('QuizResult', () => {
  it('renders the server verdict rather than recomputing it', () => {
    render(
      <QuizResult
        questions={QUESTIONS}
        graded={GRADED}
        score={0.5}
        kpName="数组"
        onContinue={vi.fn()}
        onRetry={vi.fn()}
      />
    )
    // Count and percentage both come from the graded payload.
    expect(screen.getByText('1/2 题正确')).toBeTruthy()
    expect(screen.getByText('50%')).toBeTruthy()
  })

  it('shows the correct answer only for questions the server marked wrong', () => {
    render(
      <QuizResult
        questions={QUESTIONS}
        graded={GRADED}
        score={0.5}
        kpName="数组"
        onContinue={vi.fn()}
        onRetry={vi.fn()}
      />
    )
    // q2 was wrong, so its key is shown as review...
    expect(screen.getByText(/正确答案：LIFO/)).toBeTruthy()
    // ...and q1, being correct, shows no key line at all.
    expect(screen.queryByText(/正确答案：O\(n\)/)).toBeNull()
  })

  it('trusts the server even when the submitted answer looks right', () => {
    // The server says q1 is wrong. A component that graded locally would
    // disagree if it had a (stale or forged) key saying otherwise — so this
    // fixture makes the two disagree and asserts the server wins.
    const disagreeing: GradedQuestionData[] = [
      { id: 'q1', correct: false, difficulty: 3, submitted: 'O(1)', correct_answer: 'O(1)' },
    ]
    render(
      <QuizResult
        questions={[QUESTIONS[0]]}
        graded={disagreeing}
        score={0}
        kpName="数组"
        onContinue={vi.fn()}
        onRetry={vi.fn()}
      />
    )
    expect(screen.getByText('0/1 题正确')).toBeTruthy()
    // Wrong per the server ⇒ the key line is rendered.
    expect(screen.getByText(/正确答案：O\(1\)/)).toBeTruthy()
  })
})
