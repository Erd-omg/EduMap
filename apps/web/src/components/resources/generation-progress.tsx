'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

/* ── Types ───────────────────────────────────────────────────────── */

type AgentStatus = 'pending' | 'running' | 'completed' | 'failed';
type OverallStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

interface AgentPhase {
  agent: string;
  phase: string;
  label: string;
  status: AgentStatus;
}

const AGENTS: AgentPhase[] = [
  { agent: 'planner', phase: 'EXTRACT', label: '知识提取', status: 'pending' },
  { agent: 'guardian', phase: 'VALIDATE', label: '结构验证', status: 'pending' },
  { agent: 'designer', phase: 'GENERATE', label: '内容设计', status: 'pending' },
  { agent: 'coder', phase: 'GENERATE', label: '代码生成', status: 'pending' },
  { agent: 'content_auditor', phase: 'REVIEW', label: '质量审核', status: 'pending' },
  { agent: 'assessment', phase: 'ASSESS', label: '评估生成', status: 'pending' },
];

const AGENT_ICONS: Record<string, string> = {
  planner: '📋',
  guardian: '🛡️',
  designer: '🎨',
  coder: '💻',
  content_auditor: '🔍',
  assessment: '📝',
};

interface GenerationProgressProps {
  sessionId: string | null;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

/* ── Component ───────────────────────────────────────────────────── */

export function GenerationProgress({
  sessionId,
  onComplete,
  onError,
}: GenerationProgressProps) {
  const [agents, setAgents] = useState<AgentPhase[]>(AGENTS);
  const [overallStatus, setOverallStatus] = useState<OverallStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const restoringRef = useRef(false); // true when restoring session on page refresh

  const completedCount = agents.filter((a) => a.status === 'completed').length;
  const progress = agents.length > 0 ? Math.round((completedCount / agents.length) * 100) : 0;

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  // Connect to SSE when sessionId changes
  useEffect(() => {
    // Close previous connection
    eventSourceRef.current?.close();
    eventSourceRef.current = null;

    if (!sessionId) {
      setOverallStatus('idle');
      setAgents(AGENTS);
      setErrorMessage(null);
      return;
    }

    const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

    setOverallStatus('running');
    setErrorMessage(null);

    // Fetch current status first, then connect SSE based on result
    restoringRef.current = true;
    fetch(`${API_BASE}/api/v1/orchestrator/status/${sessionId}`)
      .then((r) => r.json())
      .then((data) => {
        // Handle already-completed or failed sessions (no SSE needed)
        if (data.overall_status === 'completed') {
          setOverallStatus('completed');
          setAgents(AGENTS.map((a) => ({ ...a, status: 'completed' as const })));
          // NOT calling onComplete — this is a restore from refresh, not a
          // real-time completion. Parent would clear sessionId, hiding progress.
          return;
        }
        if (data.overall_status === 'failed') {
          setOverallStatus('failed');
          setAgents(AGENTS);
          setErrorMessage(data.errors?.[0]?.error || '生成过程出错');
          // NOT calling onError — same restore reason as above.
          return;
        }

        // Restore agent state from agent_results for in-progress sessions
        const agentResults = data.agent_results || {};
        const completedNames = Object.keys(agentResults);
        setAgents(
          AGENTS.map((a) => ({
            ...a,
            status: completedNames.includes(a.agent)
              ? ('completed' as const)
              : ('pending' as const),
          })),
        );
        // Session is still in-progress — clear restoring flag so subsequent
        // SSE events can fire callbacks normally.
        restoringRef.current = false;
      })
      .catch(() => {
        // Status fetch failed — keep initial pending state, SSE will update
        setAgents(AGENTS);
        restoringRef.current = false;
      });

    // Connect SSE for real-time progress (always, for new or in-progress sessions)
    const url = `${API_BASE}/api/v1/orchestrator/stream/${sessionId}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.addEventListener('phase_change', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        // Mark this agent as running
        setAgents((prev) =>
          prev.map((a) =>
            a.agent === data.agent ? { ...a, status: 'running' as const } : a,
          ),
        );
      } catch {
        // Ignore parse errors
      }
    });

    es.addEventListener('agent_complete', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        setAgents((prev) =>
          prev.map((a) =>
            a.agent === data.agent ? { ...a, status: 'completed' as const } : a,
          ),
        );
      } catch {
        // Ignore parse errors
      }
    });

    es.addEventListener('workflow_complete', (event: MessageEvent) => {
      let succeeded = true;
      let errMsg = '';
      try {
        const data = JSON.parse(event.data);
        succeeded = data.status !== 'failed';
        errMsg = data.error || '';
      } catch {
        // Ignore parse errors
      }
      // Mark all remaining pending/running agents as completed
      setAgents((prev) =>
        prev.map((a) =>
          a.status === 'pending' || a.status === 'running'
            ? { ...a, status: succeeded ? 'completed' as const : 'failed' as const }
            : a,
        ),
      );
      if (succeeded) {
        setOverallStatus('completed');
      } else {
        setOverallStatus('failed');
        if (errMsg) setErrorMessage(errMsg);
      }
      es.close();
      eventSourceRef.current = null;
      if (succeeded) onComplete?.();
      else onError?.(errMsg || '生成过程出错');
    });

    es.addEventListener('workflow_error', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const errMsg = data.error || data.message || '生成过程出错';
        setErrorMessage(errMsg);
        onError?.(errMsg);
      } catch {
        setErrorMessage('生成过程出现未知错误');
      }
      // Mark the agent that failed
      setAgents((prev) => {
        const runningIdx = prev.findIndex((a) => a.status === 'running');
        if (runningIdx >= 0) {
          const updated = [...prev];
          updated[runningIdx] = { ...updated[runningIdx], status: 'failed' as const };
          return updated;
        }
        return prev;
      });
      setOverallStatus('failed');
      es.close();
      eventSourceRef.current = null;
    });

    es.onerror = () => {
      // Only show error if we're still in running state (not already completed)
      setOverallStatus((prev) => {
        if (prev === 'running') {
          setErrorMessage('连接中断，无法获取生成进度');
          return 'failed';
        }
        return prev;
      });
      es.close();
      eventSourceRef.current = null;
    };
  }, [sessionId, onComplete, onError]);

  const handleRetry = useCallback(() => {
    setOverallStatus('idle');
    setAgents(AGENTS);
    setErrorMessage(null);
  }, []);

  const handleCancel = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setOverallStatus('cancelled');
    setErrorMessage(null);
    onComplete?.();
  }, [onComplete]);

  if (overallStatus === 'idle') return null;

  return (
    <div className="rounded-xl border border-border bg-bg-card p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text-primary">
          🤖 AI 资源生成
        </h3>
        <div className="flex items-center gap-2">
          <span className={`inline-block h-2 w-2 rounded-full ${
            overallStatus === 'running' ? 'bg-brand animate-pulse' :
            overallStatus === 'completed' ? 'bg-success' :
            overallStatus === 'cancelled' ? 'bg-text-light' :
            'bg-danger'
          }`} />
          <span className="text-xs text-text-secondary">
            {overallStatus === 'running' ? '生成中...' :
             overallStatus === 'completed' ? '已完成' :
             overallStatus === 'cancelled' ? '已取消' :
             '生成失败'}
          </span>
          {overallStatus === 'running' && (
            <button
              onClick={handleCancel}
              className="rounded-lg border border-danger/30 px-2.5 py-1 text-[10px] font-medium text-danger hover:bg-danger-bg transition-colors"
            >
              取消生成
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="space-y-1">
        <div className="h-2 w-full overflow-hidden rounded-full bg-border">
          <div
            className="h-full rounded-full bg-brand transition-all duration-700 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-right text-xs text-text-light">
          {completedCount}/{agents.length} 阶段完成
        </p>
      </div>

      {/* Agent grid */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {agents.map((agent) => (
          <div
            key={agent.agent}
            className={`flex flex-col items-center gap-1.5 rounded-lg border p-3 text-center transition-colors ${
              agent.status === 'completed'
                ? 'border-success/30 bg-success-bg'
                : agent.status === 'running'
                  ? 'border-brand/30 bg-brand/5'
                  : agent.status === 'failed'
                    ? 'border-danger/30 bg-danger-bg'
                    : 'border-border bg-bg-secondary'
            }`}
          >
            <span className="text-xl">
              {agent.status === 'completed' ? '✅' :
               agent.status === 'running' ? '⏳' :
               agent.status === 'failed' ? '❌' :
               AGENT_ICONS[agent.agent] || '⏸️'}
            </span>
            <span className="text-xs font-medium text-text-primary">
              {agent.label}
            </span>
            <span className="text-[10px] text-text-light">
              {agent.status === 'completed' ? '已完成' :
               agent.status === 'running' ? '进行中' :
               agent.status === 'failed' ? '失败' :
               '等待中'}
            </span>
            {/* Individual progress bar */}
            <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-border">
              <div
                className={`h-full rounded-full transition-all duration-700 ${
                  agent.status === 'completed'
                    ? 'w-full bg-success'
                    : agent.status === 'running'
                      ? 'w-3/4 bg-brand animate-pulse'
                      : agent.status === 'failed'
                        ? 'w-1/2 bg-danger'
                        : 'w-0'
                }`}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Error message */}
      {errorMessage && (
        <div className="rounded-lg border border-danger/20 bg-danger-bg p-3">
          <p className="text-xs text-danger">{errorMessage}</p>
        </div>
      )}

      {/* Retry button on failure */}
      {overallStatus === 'failed' && (
        <div className="flex justify-center">
          <button
            onClick={handleRetry}
            className="rounded-lg bg-brand px-4 py-1.5 text-xs font-medium text-white hover:bg-brand-hover transition-colors"
          >
            重新生成
          </button>
        </div>
      )}
    </div>
  );
}
