'use client';

import { useProfileStore } from '@/stores/profile-store';

interface ProgressPanelProps {
  masteredCount: number;
  learningCount: number;
  notStartedCount: number;
  totalCount: number;
  nextKpName?: string;
  nextKpReason?: string;
  recommendedContentType?: string;
  estimatedSessionMin?: number;
  onStartNext?: () => void;
}

export function ProgressPanel({
  masteredCount,
  learningCount,
  notStartedCount,
  totalCount,
  nextKpName,
  nextKpReason,
  recommendedContentType,
  estimatedSessionMin,
  onStartNext,
}: ProgressPanelProps) {
  const profile = useProfileStore((s) => s.profile);
  const completed = masteredCount + learningCount;

  // Dominant interaction style
  const style = profile?.interaction_style;
  const topStyle = style
    ? Object.entries(style).sort(([, a], [, b]) => b - a)[0]?.[0]
    : null;

  const styleLabels: Record<string, string> = {
    visual: '视觉型',
    textual: '文字型',
    interactive: '交互型',
    auditory: '听觉型',
  };

  return (
    <div className="space-y-5">
      {/* Progress overview */}
      <div>
        <h3 className="mb-2 text-sm font-medium text-text-primary">学习进度</h3>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-border">
          <div
            className="h-full rounded-full bg-brand transition-all duration-500"
            style={{ width: `${totalCount > 0 ? (completed / totalCount) * 100 : 0}%` }}
          />
        </div>
        <div className="mt-1.5 flex justify-between text-xs text-gray-500">
          <span>{completed}/{totalCount} 已完成</span>
          <span>{Math.round(totalCount > 0 ? (completed / totalCount) * 100 : 0)}%</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg border border-success/20 bg-success-bg p-2 text-center">
          <div className="text-lg font-bold text-success">{masteredCount}</div>
          <div className="text-xs text-success">已掌握</div>
        </div>
        <div className="rounded-lg border border-warning/20 bg-warning-bg p-2 text-center">
          <div className="text-lg font-bold text-warning">{learningCount}</div>
          <div className="text-xs text-warning">学习中</div>
        </div>
        <div className="rounded-lg border border-border bg-bg-secondary p-2 text-center">
          <div className="text-lg font-bold text-text-light">{notStartedCount}</div>
          <div className="text-xs text-text-light">未开始</div>
        </div>
      </div>

      {/* Next recommendation */}
      {nextKpName && (
        <div className="rounded-lg border border-accent/20 bg-warning-bg p-3">
          <h4 className="text-xs font-medium text-accent mb-1">
            下一个推荐
          </h4>
          <p className="text-sm font-medium text-text-primary">{nextKpName}</p>
          {nextKpReason && (
            <p className="mt-1 text-xs text-text-secondary">{nextKpReason}</p>
          )}
          {onStartNext && (
            <button
              onClick={onStartNext}
              className="mt-2 rounded bg-accent px-3 py-1 text-xs font-medium text-white hover:opacity-90 transition-colors"
            >
              开始学习
            </button>
          )}
        </div>
      )}

      {/* Adaptive content suggestion */}
      {topStyle && (
        <div className="rounded-lg border border-brand/20 bg-brand/5 p-3">
          <h4 className="text-xs font-medium text-brand mb-1">
            自适应内容推荐
          </h4>
          <p className="text-xs text-text-secondary">
            根据您的{styleLabels[topStyle] || topStyle}学习偏好
          </p>
          {recommendedContentType && (
            <p className="mt-1 text-xs text-text-secondary">
              推荐内容类型：{recommendedContentType}
            </p>
          )}
          {estimatedSessionMin && (
            <p className="mt-1 text-xs text-text-secondary">
              建议学习时长：{estimatedSessionMin} 分钟
            </p>
          )}
        </div>
      )}

      {/* Learning style info */}
      {style && (
        <details className="rounded-lg border border-border bg-bg-secondary p-3">
          <summary className="cursor-pointer text-xs font-medium text-text-secondary">
            学习风格详情
          </summary>
          <div className="mt-2 space-y-1">
            {Object.entries(style).map(([key, val]) => (
              <div key={key} className="flex items-center gap-2">
                <span className="w-16 text-xs text-text-secondary">
                  {styleLabels[key] || key}
                </span>
                <div className="flex-1 h-1.5 rounded-full bg-border">
                  <div
                    className="h-full rounded-full bg-brand"
                    style={{ width: `${(val as number) * 100}%` }}
                  />
                </div>
                <span className="w-8 text-right text-xs text-text-light">
                  {Math.round((val as number) * 100)}%
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
