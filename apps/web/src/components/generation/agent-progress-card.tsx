'use client';

import { cn } from '@/lib/utils';

export type AgentPhaseStatus = 'pending' | 'running' | 'passed' | 'failed' | 'degraded';

interface AgentProgressCardProps {
  agentType: string;
  status: AgentPhaseStatus;
  agentName: string;
  retryCount?: number;
  className?: string;
}

const STATUS_CONFIG: Record<AgentPhaseStatus, { icon: string; color: string; label: string }> = {
  pending: { icon: '○', color: 'text-gray-400', label: '待开始' },
  running: { icon: '◌', color: 'text-blue-500 animate-pulse', label: '运行中' },
  passed: { icon: '✓', color: 'text-green-500', label: '通过' },
  failed: { icon: '✕', color: 'text-red-500', label: '失败' },
  degraded: { icon: '⚠', color: 'text-amber-500', label: '降级' },
};

const AGENT_ICONS: Record<string, string> = {
  planner: '📋',
  guardian: '🛡️',
  designer: '🎨',
  coder: '💻',
  content_auditor: '🔍',
  assessment: '📝',
};

export function AgentProgressCard({
  agentType,
  status,
  agentName,
  retryCount,
  className,
}: AgentProgressCardProps) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  const icon = AGENT_ICONS[agentType] ?? '🤖';

  return (
    <div
      className={cn(
        'flex items-center gap-3 rounded-lg border bg-white p-3 shadow-sm transition-all',
        status === 'running' && 'border-blue-300 shadow-blue-100',
        status === 'failed' && 'border-red-300',
        status === 'degraded' && 'border-amber-300',
        className,
      )}
    >
      <span className="text-xl">{icon}</span>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-900 truncate">{agentName}</span>
          {retryCount && retryCount > 0 ? (
            <span className="text-xs text-gray-400">(重试 {retryCount})</span>
          ) : null}
        </div>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className={cn('text-sm', cfg.color)}>{cfg.icon}</span>
          <span className={cn('text-xs', cfg.color)}>{cfg.label}</span>
        </div>
      </div>
    </div>
  );
}
