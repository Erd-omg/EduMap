'use client';

import type { QuizQuestionData } from './quiz-viewer';

/** One row of the server's grading result (see `QuizGradeResponse`). */
export interface GradedQuestionData {
  id: string;
  correct: boolean;
  difficulty: number;
  submitted?: string | null;
  correct_answer: string;
}

interface QuizResultProps {
  questions: QuizQuestionData[];
  /** Who said what — from `QuizGradeResponse.per_question`. */
  graded: GradedQuestionData[];
  score: number;
  kpName: string;
  onContinue: () => void;
  onRetry: () => void;
}

export function QuizResult({
  questions,
  graded,
  score,
  kpName,
  onContinue,
  onRetry,
}: QuizResultProps) {
  // Correctness comes from the server's grading, never from comparing here:
  // this component has no answer key (it is only ever disclosed per-question
  // inside `graded`, which is the server's own verdict on the same attempt).
  const correctCount = graded.filter((g) => g.correct).length;
  const gradedById = new Map(graded.map((g) => [g.id, g]));

  const percentage = Math.round(score * 100);

  const getGradeInfo = (pct: number) => {
    if (pct >= 80) return { label: '优秀', color: 'text-green-600', bg: 'bg-green-50 border-green-200', icon: '🌟' };
    if (pct >= 60) return { label: '良好', color: 'text-blue-600', bg: 'bg-blue-50 border-blue-200', icon: '👍' };
    if (pct >= 40) return { label: '需加强', color: 'text-yellow-600', bg: 'bg-yellow-50 border-yellow-200', icon: '📚' };
    return { label: '需复习', color: 'text-red-600', bg: 'bg-red-50 border-red-200', icon: '🔁' };
  };

  const grade = getGradeInfo(percentage);

  return (
    <div className="rounded-xl border border-border bg-bg-card p-6">
      {/* Score circle */}
      <div className="flex flex-col items-center mb-6">
        <div className="text-3xl mb-2">{grade.icon}</div>
        <div
          className={`w-24 h-24 rounded-full flex items-center justify-center text-2xl font-bold border-4 ${grade.bg} ${grade.color}`}
          style={{
            borderColor: percentage >= 80 ? '#22c55e' : percentage >= 60 ? '#3b82f6' : percentage >= 40 ? '#eab308' : '#ef4444',
          }}
        >
          {percentage}%
        </div>
        <p className={`mt-2 text-sm font-semibold ${grade.color}`}>{grade.label}</p>
        <p className="text-xs text-text-light mt-1">
          {correctCount}/{graded.length} 题正确
        </p>
      </div>

      {/* Kp info */}
      <p className="text-xs text-text-light text-center mb-4">
        知识点：{kpName}
      </p>

      {/* Answer review — every value below is the server's, not a local guess */}
      <div className="space-y-3 mb-5 max-h-48 overflow-y-auto">
        {questions.map((q) => {
          const g = gradedById.get(q.id);
          const isCorrect = g?.correct ?? false;
          const submitted = g?.submitted;
          return (
            <div
              key={q.id}
              className={`rounded-lg border p-3 text-sm ${
                isCorrect
                  ? 'border-green-200 bg-green-50'
                  : 'border-red-200 bg-red-50'
              }`}
            >
              <p className="text-xs font-medium text-text-primary mb-1">{q.content}</p>
              <div className="flex items-center gap-2 text-xs">
                <span className={isCorrect ? 'text-green-600' : 'text-red-600'}>
                  你的答案：{submitted || '未作答'}
                </span>
                {!isCorrect && g && (
                  <span className="text-green-600">
                    · 正确答案：{g.correct_answer}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onRetry}
          className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
        >
          重新测验
        </button>
        <button
          onClick={onContinue}
          className="flex-1 rounded-lg bg-brand px-4 py-2.5 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
        >
          {percentage >= 60 ? '继续学习 ✓' : '标记完成'}
        </button>
      </div>
    </div>
  );
}
