'use client';

interface TrendPoint {
  date: string;
  mastered: number;
  in_progress: number;
}

interface LearningTrendProps {
  data: TrendPoint[];
  isLoading?: boolean;
}

export function LearningTrend({ data, isLoading }: LearningTrendProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-3">📈 学习趋势</h3>
        <div className="animate-pulse">
          <div className="h-24 rounded-lg bg-bg-secondary" />
        </div>
      </div>
    );
  }

  // Show last 7 data points or empty state
  const points = data.slice(-7);
  const totalProgress = points.length > 0 ? points[points.length - 1].mastered : 0;

  return (
    <div className="rounded-xl border border-border bg-bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-text-primary">📈 学习趋势</h3>
        <span className="text-xs text-text-light">
          已掌握 {totalProgress} 个知识点
        </span>
      </div>

      {points.length === 0 ? (
        <p className="text-xs text-text-light text-center py-6">
          暂无足够数据 — 开始学习以生成趋势
        </p>
      ) : (
        <div className="space-y-2">
          {points.map((pt) => (
            <div key={pt.date} className="flex items-center gap-3">
              <span className="shrink-0 w-16 text-[10px] text-text-light">
                {pt.date.slice(5)} {/* MM-DD */}
              </span>
              <div className="flex-1 flex gap-0.5 h-5">
                {/* Mastered bar */}
                <div
                  className="h-full rounded-l-sm bg-green-400 transition-all"
                  style={{
                    width: `${Math.min(100, pt.mastered * 5)}%`,
                    minWidth: pt.mastered > 0 ? '4px' : 0,
                  }}
                />
                {/* In progress bar */}
                <div
                  className="h-full rounded-r-sm bg-yellow-400 transition-all"
                  style={{
                    width: `${Math.min(100, pt.in_progress * 3)}%`,
                    minWidth: pt.in_progress > 0 ? '4px' : 0,
                  }}
                />
              </div>
              <span className="shrink-0 text-[10px] text-text-light w-8 text-right">
                {pt.mastered}
              </span>
            </div>
          ))}

          {/* Legend */}
          <div className="flex items-center gap-4 pt-1 border-t border-border mt-2">
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-green-400" />
              <span className="text-[10px] text-text-light">已掌握</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-yellow-400" />
              <span className="text-[10px] text-text-light">学习中</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
