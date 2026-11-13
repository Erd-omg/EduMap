/** Assessment types for quiz and evaluation workflows */

export type QuestionType = 'choice' | 'true_false' | 'fill_blank';

/**
 * A quiz question as delivered to the browser — **no answer key**.
 *
 * `correct_answer` deliberately lives only in `GradedQuestion` below. It used
 * to be on this type, which meant every generate response handed the learner
 * the answers; combined with a grade endpoint that trusted client-supplied
 * questions, it made the score self-certified.
 */
export interface QuizQuestion {
  id: string;
  type: QuestionType;
  content: string;
  options?: string[];
  knowledge_point_id: string;
}

/** Per-question outcome returned by `POST /quiz/grade`, after scoring. */
export interface GradedQuestion {
  id: string;
  correct: boolean;
  difficulty: number;
  /** What the learner submitted; `null`/absent when unanswered. */
  submitted?: string | null;
  /**
   * Disclosed only here, only after grading. Safe because the attempt is
   * already scored and the `quiz_id` is single-use, so it enables review but
   * not a retry.
   */
  correct_answer: string;
}

export interface QuizGradeResponse {
  score: number;
  n_correct: number;
  n_questions: number;
  mastery: number;
  ability: number;
  standard_error: number;
  mastery_delta: Record<string, number>; // knowledge_point_id -> delta
  per_question: GradedQuestion[];
}

export interface QuizResult {
  question_id: string;
  user_answer: string;
  is_correct: boolean;
  time_spent_seconds: number;
}

export interface AssessmentResult {
  quiz_id: string;
  questions: QuizQuestion[];
  results: QuizResult[];
  mastery_delta: Record<string, number>; // knowledge_point_id -> delta
  timestamp: string; // ISO 8601
}
