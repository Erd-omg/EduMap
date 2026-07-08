/** Assessment types for quiz and evaluation workflows */

export type QuestionType = 'choice' | 'true_false' | 'fill_blank';

export interface QuizQuestion {
  id: string;
  type: QuestionType;
  content: string;
  options?: string[];
  correct_answer: string;
  knowledge_point_id: string;
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
