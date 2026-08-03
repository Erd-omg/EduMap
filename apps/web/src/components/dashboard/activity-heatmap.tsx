'use client';

interface ActivityData {
  date: string;
  count: number;
}

interface ActivityHeatmapProps {
  data: ActivityData[];
  isLoading?: boolean;
}

export function ActivityHeatmap({ data, isLoading }: ActivityHeatmapProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-3">🔥 学习活跃度</h3>
        <div className="animate-pulse">
          <div className="h-24 rounded-lg bg-bg-secondary" />
        </div>
      </div>
    );
  }

  // Generate last 7 days data
  const today = new Date();
  const days: Array<{ date: Date; label: string; count: number }> = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const dateStr = d.toISOString().split('T')[0];
    const dayData = data.find((dd) => dd.date === dateStr);
    days.push({
      date: d,
      label: ['日', '一', '二', '三', '四', '五', '六'][d.getDay()],
      count: dayData?.count ?? 0,
    });
  }

  const maxCount = Math.max(1, ...days.map((d) => d.count));

  return (
    <div className="rounded-xl border border-border bg-bg-card p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-3">🔥 学习活跃度</h3>

      {/* Bar chart */}
      <div className="flex items-end justify-between gap-2 h-28">
        {days.map((day) => {
          const height = maxCount > 0 ? (day.count / maxCount) * 100 : 0;
          return (
            <div key={day.date.toISOString()} className="flex flex-col items-center gap-1 flex-1">
              <span className="text-[10px] text-text-light font-medium">{day.count}</span>
              <div className="w-full rounded-md bg-bg-secondary overflow-hidden" style={{ height: '80px' }}>
                <div
                  className="w-full rounded-md bg-brand transition-all duration-500"
                  style={{
                    height: `${Math.max(4, height)}%`,
                    marginTop: `${100 - Math.max(4, height)}%`,
                  }}
                />
              </div>
              <span className="text-[10px] text-text-light">{day.label}</span>
            </div>
          );
        })}
      </div>

      {days.every((d) => d.count === 0) && (
        <p className="text-xs text-text-light text-center mt-2">暂无学习记录</p>
      )}
    </div>
  );
}
