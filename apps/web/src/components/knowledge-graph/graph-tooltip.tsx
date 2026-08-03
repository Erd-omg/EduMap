'use client';

import { type ReactNode } from 'react';

export interface TooltipData {
  name: string;
  description: string;
  difficulty: number;
  mastery: string;
  progress: number;
  prerequisites: string[];
}

interface GraphTooltipProps {
  data: TooltipData | null;
  position: { x: number; y: number } | null;
}

const MASTERY_LABELS: Record<string, string> = {
  mastered: '已掌握',
  learning: '学习中',
  not_started: '未开始',
  locked: '未解锁',
};

const DIFFICULTY_LABELS: Record<string, string> = {
  '1': '入门',
  '2': '简单',
  '3': '中等',
  '4': '较难',
  '5': '困难',
};

export function GraphTooltip({ data, position }: GraphTooltipProps) {
  if (!data || !position) return null;

  return (
    <div
      className="pointer-events-none fixed z-50 rounded-xl border border-border bg-bg-card px-4 py-3 shadow-lg animate-[fade-in_0.1s_ease-out]"
      style={{
        left: position.x + 16,
        top: position.y - 10,
        maxWidth: 260,
      }}
    >
      {/* Node name */}
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-sm font-semibold text-text-primary">{data.name}</span>
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
            data.mastery === 'mastered'
              ? 'bg-success-bg text-success'
              : data.mastery === 'learning'
                ? 'bg-warning-bg text-warning'
                : 'bg-danger-bg text-danger'
          }`}
        >
          {MASTERY_LABELS[data.mastery] || data.mastery}
        </span>
      </div>

      {/* Description */}
      {data.description && (
        <p className="text-xs text-text-secondary mb-2 line-clamp-3">
          {data.description}
        </p>
      )}

      {/* Details */}
      <div className="space-y-1 text-xs text-text-light">
        <div className="flex justify-between">
          <span>难度</span>
          <span className="text-text-secondary">
            {DIFFICULTY_LABELS[String(data.difficulty)] || data.difficulty} ({data.difficulty}/5)
          </span>
        </div>
        {data.progress > 0 && (
          <div className="flex justify-between">
            <span>学习进度</span>
            <span className="text-text-secondary">{Math.round(data.progress)}%</span>
          </div>
        )}
        {data.prerequisites.length > 0 && (
          <div>
            <span className="block mb-0.5">前置知识</span>
            <div className="flex flex-wrap gap-1">
              {data.prerequisites.map((p) => (
                <span key={p} className="rounded bg-bg-secondary px-1.5 py-0.5 text-[10px]">
                  {p}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
