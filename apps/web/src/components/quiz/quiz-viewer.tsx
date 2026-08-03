'use client';

import { useState } from 'react';

export interface QuizQuestionData {
  id: string;
  type: 'choice' | 'true_false' | 'fill_blank';
  content: string;
  options: string[] | null;
  correct_answer: string;
  knowledge_point_id: string;
}

interface QuizViewerProps {
  questions: QuizQuestionData[];
  kpName: string;
  onSubmit: (answers: Record<string, string>, score: number) => void;
  onSkip: () => void;
  isLoading?: boolean;
}

export function QuizViewer({
  questions,
  kpName,
  onSubmit,
  onSkip,
  isLoading,
}: QuizViewerProps) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [currentIndex, setCurrentIndex] = useState(0);

  const current = questions[currentIndex];
  const isLast = currentIndex === questions.length - 1;
  const hasAnswered = current && answers[current.id] !== undefined;

  const handleAnswer = (value: string) => {
    setAnswers((prev) => ({ ...prev, [current.id]: value }));
  };

  const handleNext = () => {
    if (isLast) {
      handleSubmit();
    } else {
      setCurrentIndex((i) => i + 1);
    }
  };

  const handleSubmit = () => {
    let correct = 0;
    for (const q of questions) {
      const userAns = answers[q.id] || '';
      if (userAns.trim().toLowerCase() === q.correct_answer.trim().toLowerCase()) {
        correct++;
      }
    }
    const score = questions.length > 0 ? correct / questions.length : 0;
    onSubmit(answers, score);
  };

  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-4 rounded bg-bg-secondary w-3/4" />
          <div className="h-12 rounded bg-bg-secondary w-full" />
          <div className="h-12 rounded bg-bg-secondary w-full" />
          <div className="h-12 rounded bg-bg-secondary w-1/3" />
        </div>
      </div>
    );
  }

  if (!questions.length) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-6 text-center">
        <p className="text-sm text-text-light">暂无测验题目</p>
        <button
          onClick={onSkip}
          className="mt-3 rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
        >
          继续学习
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-bg-card p-6">
      {/* Progress indicator */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">
            📝 小测验
          </h3>
          <p className="text-xs text-text-light mt-0.5">{kpName}</p>
        </div>
        <span className="text-xs text-text-light">
          {currentIndex + 1} / {questions.length}
        </span>
      </div>

      {/* Progress dots */}
      <div className="flex gap-1 mb-4">
        {questions.map((q, i) => (
          <div
            key={q.id}
            className={`h-1.5 flex-1 rounded-full ${
              i < currentIndex
                ? 'bg-brand'
                : i === currentIndex
                  ? 'bg-brand/60'
                  : 'bg-bg-secondary'
            }`}
          />
        ))}
      </div>

      {/* Question */}
      <div key={current.id} className="mb-4">
        <p className="text-sm font-medium text-text-primary mb-3">
          {current.content}
        </p>

        {current.type === 'choice' && current.options && (
          <div className="space-y-2">
            {current.options.map((opt, i) => (
              <button
                key={i}
                onClick={() => handleAnswer(opt)}
                className={`w-full text-left rounded-lg border px-4 py-2.5 text-sm transition-colors ${
                  answers[current.id] === opt
                    ? 'border-brand bg-brand/5 text-brand font-medium'
                    : 'border-border text-text-secondary hover:border-brand/40 hover:bg-bg-secondary'
                }`}
              >
                <span className="mr-2 text-xs text-text-light">{String.fromCharCode(65 + i)}.</span>
                {opt}
              </button>
            ))}
          </div>
        )}

        {current.type === 'true_false' && (
          <div className="flex gap-3">
            {['True', 'False'].map((opt) => (
              <button
                key={opt}
                onClick={() => handleAnswer(opt)}
                className={`flex-1 rounded-lg border px-4 py-3 text-sm font-medium transition-colors ${
                  answers[current.id] === opt
                    ? 'border-brand bg-brand/5 text-brand'
                    : 'border-border text-text-secondary hover:border-brand/40 hover:bg-bg-secondary'
                }`}
              >
                {opt === 'True' ? '✓ 正确' : '✗ 错误'}
              </button>
            ))}
          </div>
        )}

        {current.type === 'fill_blank' && (
          <input
            type="text"
            value={answers[current.id] || ''}
            onChange={(e) => handleAnswer(e.target.value)}
            placeholder="输入你的答案..."
            className="w-full rounded-lg border border-border px-4 py-2.5 text-sm text-text-primary
                       focus:border-brand focus:ring-1 focus:ring-brand outline-none"
          />
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between pt-2 border-t border-border">
        <button
          onClick={onSkip}
          className="text-xs text-text-light hover:text-text-primary transition-colors"
        >
          跳过测验
        </button>
        <button
          onClick={handleNext}
          disabled={!hasAnswered}
          className={`rounded-lg px-5 py-2 text-sm font-medium transition-colors ${
            hasAnswered
              ? 'bg-brand text-white hover:bg-brand-hover'
              : 'bg-bg-secondary text-text-light cursor-not-allowed'
          }`}
        >
          {isLast ? '提交答案' : '下一题'}
        </button>
      </div>
    </div>
  );
}
