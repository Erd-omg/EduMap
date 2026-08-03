'use client';

interface ReviewPlanItem {
  kp_id: string;
  kp_name: string;
  reason: string;
  priority: 'high' | 'medium' | 'low';
  estimated_minutes: number;
}

interface ReviewPlanProps {
  items: ReviewPlanItem[];
  isLoading?: boolean;
  onStartReview?: (kpId: string) => void;
}

export function ReviewPlan({ items, isLoading, onStartReview }: ReviewPlanProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-3">📋 推荐复习计划</h3>
        <div className="animate-pulse space-y-2">
          {[1, 2].map((i) => (
            <div key={i} className="h-12 rounded-lg bg-bg-secondary" />
          ))}
        </div>
      </div>
    );
  }

  const priorityConfig = {
    high: { label: '高优先级', color: 'text-red-600', dot: 'bg-red-400' },
    medium: { label: '中优先级', color: 'text-yellow-600', dot: 'bg-yellow-400' },
    low: { label: '低优先级', color: 'text-green-600', dot: 'bg-green-400' },
  };

  const sorted = [...items].sort((a, b) => {
    const rank = { high: 0, medium: 1, low: 2 };
    return (rank[a.priority] ?? 3) - (rank[b.priority] ?? 3);
  });

  return (
    <div className="rounded-xl border border-border bg-bg-card p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-3">📋 推荐复习计划</h3>

      {items.length === 0 ? (
        <p className="text-xs text-text-light text-center py-6">
          暂无复习计划 — 开始学习新知识点吧
        </p>
      ) : (
        <div className="space-y-2">
          {sorted.map((item) => {
            const config = priorityConfig[item.priority] || priorityConfig.medium;
            return (
              <button
                key={item.kp_id}
                onClick={() => onStartReview?.(item.kp_id)}
                className="w-full text-left rounded-lg border border-border p-3 hover:border-brand/40 hover:bg-bg-secondary transition-colors"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${config.dot}`} />
                    <span className="text-sm font-medium text-text-primary truncate">
                      {item.kp_name || item.kp_id}
                    </span>
                  </div>
                  <span className={`shrink-0 text-[10px] font-medium ${config.color}`}>
                    {config.label}
                  </span>
                </div>
                <p className="mt-1 text-xs text-text-light">{item.reason}</p>
                {item.estimated_minutes > 0 && (
                  <p className="mt-0.5 text-[10px] text-text-light">
                    预计 {item.estimated_minutes} 分钟
                  </p>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
