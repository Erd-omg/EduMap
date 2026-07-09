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
        <h3 className="mb-2 text-sm font-medium text-gray-700">学习进度</h3>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-gray-100">
          <div
            className="h-full rounded-full bg-blue-500 transition-all duration-500"
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
        <div className="rounded-lg border border-green-200 bg-green-50 p-2 text-center">
          <div className="text-lg font-bold text-green-600">{masteredCount}</div>
          <div className="text-xs text-green-500">已掌握</div>
        </div>
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-2 text-center">
          <div className="text-lg font-bold text-blue-600">{learningCount}</div>
          <div className="text-xs text-blue-500">学习中</div>
        </div>
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-2 text-center">
          <div className="text-lg font-bold text-gray-500">{notStartedCount}</div>
          <div className="text-xs text-gray-400">未开始</div>
        </div>
      </div>

      {/* Next recommendation */}
      {nextKpName && (
        <div className="rounded-lg border border-orange-200 bg-orange-50 p-3">
          <h4 className="text-xs font-medium text-orange-700 mb-1">
            下一个推荐
          </h4>
          <p className="text-sm font-medium text-gray-900">{nextKpName}</p>
          {nextKpReason && (
            <p className="mt-1 text-xs text-gray-500">{nextKpReason}</p>
          )}
          {onStartNext && (
            <button
              onClick={onStartNext}
              className="mt-2 rounded bg-orange-500 px-3 py-1 text-xs font-medium text-white hover:bg-orange-600 transition-colors"
            >
              开始学习
            </button>
          )}
        </div>
      )}

      {/* Adaptive content suggestion */}
      {topStyle && (
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
          <h4 className="text-xs font-medium text-blue-700 mb-1">
            自适应内容推荐
          </h4>
          <p className="text-xs text-blue-600">
            根据您的{styleLabels[topStyle] || topStyle}学习偏好
          </p>
          {recommendedContentType && (
            <p className="mt-1 text-xs text-blue-500">
              推荐内容类型：{recommendedContentType}
            </p>
          )}
          {estimatedSessionMin && (
            <p className="mt-1 text-xs text-blue-500">
              建议学习时长：{estimatedSessionMin} 分钟
            </p>
          )}
        </div>
      )}

      {/* Learning style info */}
      {style && (
        <details className="rounded-lg border border-gray-200 bg-gray-50 p-3">
          <summary className="cursor-pointer text-xs font-medium text-gray-600">
            学习风格详情
          </summary>
          <div className="mt-2 space-y-1">
            {Object.entries(style).map(([key, val]) => (
              <div key={key} className="flex items-center gap-2">
                <span className="w-16 text-xs text-gray-500">
                  {styleLabels[key] || key}
                </span>
                <div className="flex-1 h-1.5 rounded-full bg-gray-200">
                  <div
                    className="h-full rounded-full bg-blue-400"
                    style={{ width: `${(val as number) * 100}%` }}
                  />
                </div>
                <span className="w-8 text-right text-xs text-gray-400">
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
