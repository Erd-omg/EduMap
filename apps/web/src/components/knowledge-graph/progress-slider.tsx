'use client';

interface ProgressSliderProps {
  masteredCount: number;
  learningCount: number;
  notStartedCount: number;
  totalCount: number;
}

export function ProgressSlider({
  masteredCount,
  learningCount,
  notStartedCount,
  totalCount,
}: ProgressSliderProps) {
  const masteredPct = totalCount > 0 ? (masteredCount / totalCount) * 100 : 0;
  const learningPct = totalCount > 0 ? (learningCount / totalCount) * 100 : 0;

  return (
    <div className="space-y-3">
      {/* Progress bar */}
      <div className="h-3 w-full overflow-hidden rounded-full bg-gray-100">
        <div className="flex h-full">
          <div
            className="bg-green-500 transition-all duration-500"
            style={{ width: `${masteredPct}%` }}
          />
          <div
            className="bg-blue-500 transition-all duration-500"
            style={{ width: `${learningPct}%` }}
          />
        </div>
      </div>

      {/* Labels */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>{Math.round(masteredPct)}% 已完成</span>
        <span>{totalCount} 个知识点</span>
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-green-500" />
          <span className="text-gray-600">已掌握 ({masteredCount})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-blue-500" />
          <span className="text-gray-600">学习中 ({learningCount})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-gray-300" />
          <span className="text-gray-600">未开始 ({notStartedCount})</span>
        </div>
      </div>
    </div>
  );
}
