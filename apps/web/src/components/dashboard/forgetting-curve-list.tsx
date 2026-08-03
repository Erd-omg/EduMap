'use client';

interface Alert {
  kp_id: string;
  kp_name: string;
  recall_probability: number;
  strength: number;
  review_count: number;
  last_review: number;
  alert_level: 'urgent' | 'warning' | 'ok';
}

interface ForgettingCurveListProps {
  alerts: Alert[];
  isLoading?: boolean;
  onKpClick?: (kpId: string) => void;
}

export function ForgettingCurveList({ alerts, isLoading, onKpClick }: ForgettingCurveListProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-3">📊 遗忘曲线预警</h3>
        <div className="animate-pulse space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-14 rounded-lg bg-bg-secondary" />
          ))}
        </div>
      </div>
    );
  }

  const urgentAlerts = alerts.filter((a) => a.alert_level === 'urgent');
  const warningAlerts = alerts.filter((a) => a.alert_level === 'warning');
  const okAlerts = alerts.filter((a) => a.alert_level === 'ok');

  const renderAlertItem = (alert: Alert) => {
    const recallPct = Math.round(alert.recall_probability * 100);
    return (
      <button
        key={alert.kp_id}
        onClick={() => onKpClick?.(alert.kp_id)}
        className="w-full text-left rounded-lg border border-border p-3 hover:border-brand/40 hover:bg-bg-secondary transition-colors"
      >
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-medium text-text-primary truncate">
            {alert.kp_name || alert.kp_id}
          </span>
          <span className={`shrink-0 text-xs font-medium px-2 py-0.5 rounded-full ${
            alert.alert_level === 'urgent'
              ? 'bg-red-100 text-red-700'
              : alert.alert_level === 'warning'
                ? 'bg-yellow-100 text-yellow-700'
                : 'bg-green-100 text-green-700'
          }`}>
            {recallPct}%
          </span>
        </div>
        {/* Recall bar */}
        <div className="mt-1.5 h-1.5 w-full rounded-full bg-bg-secondary overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              alert.alert_level === 'urgent'
                ? 'bg-red-400'
                : alert.alert_level === 'warning'
                  ? 'bg-yellow-400'
                  : 'bg-green-400'
            }`}
            style={{ width: `${recallPct}%` }}
          />
        </div>
        <div className="flex items-center gap-3 mt-1 text-[10px] text-text-light">
          <span>已复习 {alert.review_count} 次</span>
          <span>记忆强度 {alert.strength}h</span>
        </div>
      </button>
    );
  };

  return (
    <div className="rounded-xl border border-border bg-bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-text-primary">📊 遗忘曲线预警</h3>
        <span className="text-xs text-text-light">{alerts.length} 项</span>
      </div>

      {alerts.length === 0 ? (
        <p className="text-xs text-text-light text-center py-6">
          暂无预警 — 继续学习更多知识点吧
        </p>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {urgentAlerts.length > 0 && (
            <>
              <p className="text-[10px] font-medium text-red-500 uppercase tracking-wider">紧急复习</p>
              {urgentAlerts.map(renderAlertItem)}
            </>
          )}
          {warningAlerts.length > 0 && (
            <>
              <p className="text-[10px] font-medium text-yellow-500 uppercase tracking-wider mt-2">建议复习</p>
              {warningAlerts.map(renderAlertItem)}
            </>
          )}
        </div>
      )}
    </div>
  );
}
