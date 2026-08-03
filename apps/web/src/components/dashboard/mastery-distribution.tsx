'use client';

interface MasteryDistributionProps {
  distribution: {
    urgent: number;
    warning: number;
    ok: number;
    unknown: number;
  };
  total: number;
  isLoading?: boolean;
}

const COLORS = {
  urgent: { label: '需紧急复习', color: '#ef4444', bg: 'bg-red-100 text-red-700' },
  warning: { label: '建议复习', color: '#eab308', bg: 'bg-yellow-100 text-yellow-700' },
  ok: { label: '掌握良好', color: '#22c55e', bg: 'bg-green-100 text-green-700' },
  unknown: { label: '未开始', color: '#9ca3af', bg: 'bg-gray-100 text-gray-600' },
};

export function MasteryDistribution({ distribution, total, isLoading }: MasteryDistributionProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-3">🎯 掌握度分布</h3>
        <div className="animate-pulse flex items-center gap-4">
          <div className="w-24 h-24 rounded-full bg-bg-secondary" />
          <div className="space-y-2 flex-1">
            {[1, 2, 3].map((i) => <div key={i} className="h-4 rounded bg-bg-secondary w-3/4" />)}
          </div>
        </div>
      </div>
    );
  }

  const items = [
    { key: 'urgent', value: distribution.urgent, ...COLORS.urgent },
    { key: 'warning', value: distribution.warning, ...COLORS.warning },
    { key: 'ok', value: distribution.ok, ...COLORS.ok },
    { key: 'unknown', value: distribution.unknown, ...COLORS.unknown },
  ] as const;

  const totalTracked = total || items.reduce((sum, item) => sum + item.value, 0);

  // Calculate pie chart segments (conic-gradient)
  const pieSegments = items
    .filter((item) => item.value > 0)
    .map((item) => ({ color: item.color, value: item.value }));
  const totalPieValue = pieSegments.reduce((sum, seg) => sum + seg.value, 0);
  const conicGradient = pieSegments
    .reduce((acc, seg, i) => {
      const startPct = acc.reduce((s, p) => s + p.pct, 0);
      const pct = totalPieValue > 0 ? (seg.value / totalPieValue) * 100 : 0;
      const endPct = startPct + pct;
      acc.push({ color: seg.color, pct, startPct, endPct });
      return acc;
    }, [] as Array<{ color: string; pct: number; startPct: number; endPct: number }>);

  const gradientStops = conicGradient
    .map((seg) => `${seg.color} ${seg.startPct}% ${seg.endPct}%`)
    .join(', ');

  return (
    <div className="rounded-xl border border-border bg-bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-text-primary">🎯 掌握度分布</h3>
        <span className="text-xs text-text-light">{totalTracked} 个知识点</span>
      </div>

      <div className="flex items-center gap-6">
        {/* Donut chart */}
        <div className="relative shrink-0">
          <div
            className="w-24 h-24 rounded-full"
            style={{
              background: conicGradient.length > 0
                ? `conic-gradient(${gradientStops})`
                : '#f3f4f6',
            }}
          />
          <div className="absolute inset-3 rounded-full bg-white flex items-center justify-center">
            <span className="text-lg font-bold text-text-primary">
              {Math.round(
                totalPieValue > 0
                  ? (distribution.ok / totalPieValue) * 100
                  : 0,
              )}%
            </span>
          </div>
        </div>

        {/* Legend */}
        <div className="flex-1 space-y-1.5">
          {items.map((item) => (
            <div key={item.key} className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-text-secondary">{item.label}</span>
              </div>
              <span className={`font-medium px-1.5 py-0.5 rounded ${item.bg}`}>
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
