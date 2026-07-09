'use client';

import { useState } from 'react';
import { useProfileStore } from '../../stores/profile-store';
import type {
  UserProfile,
  LearningAbility,
  LearningMotivation,
  InteractionStyle,
  FocusCharacteristics,
  KnowledgeCoverage,
} from '@edumap/shared-types';

const DIMENSION_CONFIG: Record<
  string,
  { label: string; color: string; icon: string }
> = {
  learning_ability: {
    label: '学习能力',
    color: 'bg-blue-500',
    icon: '🧠',
  },
  learning_motivation: {
    label: '学习动机',
    color: 'bg-green-500',
    icon: '🎯',
  },
  knowledge_coverage: {
    label: '知识覆盖',
    color: 'bg-purple-500',
    icon: '📚',
  },
  interaction_style: {
    label: '交互偏好',
    color: 'bg-amber-500',
    icon: '💬',
  },
  focus_characteristics: {
    label: '专注力',
    color: 'bg-rose-500',
    icon: '⚡',
  },
  knowledge_base: {
    label: '知识基础',
    color: 'bg-teal-500',
    icon: '📖',
  },
};

const SUB_LABELS: Record<string, Record<string, string>> = {
  learning_ability: {
    understanding_speed: '理解速度',
    problem_solving: '问题解决',
    critical_thinking: '批判思维',
    memory_retention: '记忆保持',
    analytical_ability: '分析能力',
  },
  learning_motivation: {
    intrinsic_interest: '内在兴趣',
    goal_oriented: '目标导向',
    extrinsic_motivation: '外部激励',
  },
  interaction_style: {
    visual: '视觉型',
    textual: '文本型',
    interactive: '交互型',
    auditory: '听觉型',
  },
  focus_characteristics: {
    avg_focus_duration_min: '平均专注(分钟)',
    distraction_frequency: '分心频率',
    recommended_session_length: '建议时长(分钟)',
  },
};

function getScoreColor(score: number): string {
  if (score >= 0.7) return 'text-green-600';
  if (score >= 0.4) return 'text-amber-600';
  return 'text-red-500';
}

function getScoreBg(score: number): string {
  if (score >= 0.7) return 'bg-green-100';
  if (score >= 0.4) return 'bg-amber-100';
  return 'bg-red-50';
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.7) return 'bg-green-500';
  if (confidence >= 0.4) return 'bg-amber-500';
  return 'bg-red-400';
}

function computeDimensionScore(
  profile: UserProfile,
  key: string,
): number {
  const value = (profile as unknown as Record<string, unknown>)[key];
  if (!value || typeof value !== 'object') return 0;

  const obj = value as Record<string, unknown>;
  const numericValues = Object.values(obj).filter(
    (v): v is number => typeof v === 'number' && v >= 0 && v <= 1,
  );
  if (numericValues.length > 0) {
    return numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
  }
  return 0;
}

function computeKnowledgeCoverageScore(coverage: KnowledgeCoverage): number {
  const total =
    coverage.mastered.length +
    coverage.learning.length +
    coverage.not_started.length;
  if (total === 0) return 0;
  return coverage.mastered.length / total;
}

function getSubValues(
  profile: UserProfile,
  key: string,
): [string, number | string][] {
  const value = (profile as unknown as Record<string, unknown>)[key];
  if (!value || typeof value !== 'object') return [];

  const obj = value as Record<string, unknown>;
  const labels = SUB_LABELS[key];

  if (key === 'knowledge_coverage') {
    const cov = value as KnowledgeCoverage;
    return [
      ['已掌握', `${cov.mastered.length} 个`],
      ['学习中', `${cov.learning.length} 个`],
      ['未开始', `${cov.not_started.length} 个`],
    ];
  }

  return Object.entries(obj).map(([k, v]) => {
    const label = labels?.[k] || k;
    if (typeof v === 'number') {
      return [label, v] as [string, number];
    }
    return [label, String(v ?? '')] as [string, string];
  });
}

export function ProfileCard() {
  const { profile, confidenceScores } = useProfileStore();
  const [expandedDim, setExpandedDim] = useState<string | null>(null);

  if (!profile) {
    return (
      <div className="rounded-xl border border-dashed border-gray-200 bg-white p-8 text-center">
        <div className="mx-auto mb-3 text-4xl">📊</div>
        <h3 className="mb-2 text-lg font-semibold text-gray-700">
          6 维画像概览
        </h3>
        <p className="text-sm text-gray-400">
          尚无法显示画像数据。请先进行对话分析以生成您的个性化学习画像。
        </p>
      </div>
    );
  }

  const dims = [
    'learning_ability',
    'learning_motivation',
    'knowledge_coverage',
    'interaction_style',
    'focus_characteristics',
    'knowledge_base',
  ];

  return (
    <div className="rounded-xl border bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          6 维画像概览
        </h3>
        <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs text-blue-600">
          实时更新
        </span>
      </div>

      <div className="space-y-3">
        {dims.map((key) => {
          let score = computeDimensionScore(profile, key);
          let isKnowledgeCoverage = false;

          if (key === 'knowledge_coverage') {
            score = computeKnowledgeCoverageScore(profile.knowledge_coverage);
            isKnowledgeCoverage = true;
          }

          if (key === 'knowledge_base') {
            const kb = profile.knowledge_base;
            const vals = Object.values(kb).filter(
              (v) => typeof v === 'number',
            );
            score =
              vals.length > 0
                ? vals.reduce((a, b) => a + b, 0) / vals.length
                : 0;
          }

          const config = DIMENSION_CONFIG[key];
          const confidence = confidenceScores[key] ?? 0.5;
          const isExpanded = expandedDim === key;
          const subValues = getSubValues(profile, key);

          return (
            <div key={key} className="overflow-hidden rounded-lg border">
              {/* Main dimension row */}
              <button
                onClick={() =>
                  setExpandedDim(isExpanded ? null : key)
                }
                className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-gray-50"
              >
                <span className="text-lg">{config?.icon || '📊'}</span>
                <span className="flex-1 text-sm font-medium text-gray-800">
                  {config?.label || key}
                </span>

                {/* Score badge */}
                <span
                  className={`rounded-md px-2 py-0.5 text-xs font-semibold ${getScoreColor(score)} ${getScoreBg(score)}`}
                >
                  {(score * 100).toFixed(0)}%
                </span>

                {/* Expand indicator */}
                <svg
                  className={`h-4 w-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>

              {/* Confidence bar */}
              <div className="px-3 pb-1.5">
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <span>置信度</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-gray-100">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${getConfidenceColor(confidence)}`}
                      style={{ width: `${confidence * 100}%` }}
                    />
                  </div>
                  <span className="w-8 text-right font-medium text-gray-500">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Expanded sub-details */}
              {isExpanded && subValues.length > 0 && (
                <div className="border-t bg-gray-50/50 px-3 py-2">
                  {subValues.map(([label, value]) => (
                    <div
                      key={label}
                      className="flex items-center justify-between py-1 text-xs"
                    >
                      <span className="text-gray-500">{label}</span>
                      {typeof value === 'number' ? (
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-gray-200">
                            <div
                              className="h-full rounded-full bg-blue-400"
                              style={{
                                width: `${Math.max(0, Math.min(100, value * 100))}%`,
                              }}
                            />
                          </div>
                          <span className="w-8 text-right font-medium text-gray-700">
                            {(value * 100).toFixed(0)}%
                          </span>
                        </div>
                      ) : (
                        <span className="font-medium text-gray-700">
                          {value}
                        </span>
                      )}
                    </div>
                  ))}

                  {/* Knowledge coverage specific detail */}
                  {isKnowledgeCoverage && (
                    <div className="mt-1.5 flex flex-wrap gap-1.5 border-t pt-1.5">
                      {profile.knowledge_coverage.mastered.length > 0 && (
                        <div className="rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700">
                          已掌握: {profile.knowledge_coverage.mastered.join(', ')}
                        </div>
                      )}
                      {profile.knowledge_coverage.learning.length > 0 && (
                        <div className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700">
                          学习中: {profile.knowledge_coverage.learning.join(', ')}
                        </div>
                      )}
                      {profile.knowledge_coverage.not_started.length > 0 && (
                        <div className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
                          未开始: {profile.knowledge_coverage.not_started.join(', ')}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
