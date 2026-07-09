'use client';

import { useState, useEffect, useCallback } from 'react';
import { AgentProgressCard, AgentPhaseStatus } from './agent-progress-card';
import { ResourcePreviewList } from './resource-display';
import {
  useGenerationStore,
  type AgentStatus,
  type GenerationInput,
} from '@/stores/generation-store';
import { useSSE } from '@/hooks/use-sse';

interface SSENamedEvent {
  event: string;
  data: string;
}

const AGENT_ORDER = [
  { key: 'planner', name: '知识点提取 (Planner)' },
  { key: 'guardian', name: '结构验证 (Guardian)' },
  { key: 'designer', name: '内容生成 (Designer)' },
  { key: 'coder', name: '代码生成 (Coder)' },
  { key: 'content_auditor', name: '质量审核 (Auditor)' },
  { key: 'assessment', name: '测评生成 (Assessment)' },
];

const AGENT_PHASE_MAP: Record<string, string> = {
  planner: 'EXTRACT',
  guardian: 'VALIDATE',
  designer: 'GENERATE',
  coder: 'GENERATE',
  content_auditor: 'REVIEW',
  assessment: 'ASSESS',
};

export function GenerationWorkflow() {
  const store = useGenerationStore();
  const [input, setInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [sseUrl, setSseUrl] = useState<string | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

  // SSE handler
  const onEvent = useCallback(
    (event: SSENamedEvent) => {
      try {
        const data = JSON.parse(event.data);

        switch (event.event) {
          case 'phase_change':
            store.updatePhase(data.phase);
            break;

          case 'agent_complete':
            if (data.agent) {
              store.updateAgentStatus(data.agent, {
                phase: AGENT_PHASE_MAP[data.agent] || 'UNKNOWN',
                status: 'passed',
                retryCount: 0,
              });
            }
            break;

          case 'agent_error':
            if (data.agent) {
              store.updateAgentStatus(data.agent, {
                phase: AGENT_PHASE_MAP[data.agent] || 'UNKNOWN',
                status: 'failed',
                retryCount: 0,
              });
            }
            break;

          case 'workflow_complete':
            store.setCompleted(data.status || 'completed');
            break;

          case 'workflow_error':
            store.addError(data.error || 'Unknown error');
            store.setCompleted('failed');
            break;
        }
      } catch {
        // Ignore parse errors
      }
    },
    [store],
  );

  // Connect SSE when URL changes
  useSSE(sseUrl || '', {
    onEvent,
    autoConnect: !!sseUrl,
  });

  const handleSubmit = async () => {
    if (!input.trim() || isSubmitting) return;

    setIsSubmitting(true);
    store.reset();

    const sessionId = await store.startGeneration({
      task_input: input.trim(),
    } as GenerationInput);

    if (sessionId) {
      setSseUrl(`${API_BASE}/api/v1/orchestrator/stream/${sessionId}`);
    }

    setIsSubmitting(false);
  };

  // Get agent status from store
  const getAgentStatus = (key: string): AgentPhaseStatus => {
    if (store.overallStatus === 'idle') return 'pending';

    const agent = store.agentStatuses[key];
    if (!agent) {
      // Not started yet
      return 'pending';
    }
    return agent.status as AgentPhaseStatus;
  };

  const getAgentRetryCount = (key: string): number => {
    return store.agentStatuses[key]?.retryCount ?? 0;
  };

  return (
    <div className="space-y-6">
      {/* Input form */}
      <div className="rounded-xl border bg-white p-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          你想学习什么？
        </label>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={'例如：教我Python中的循环结构，包括for循环和while循环'}
          className="w-full rounded-lg border border-gray-300 p-3 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none"
          rows={3}
          disabled={isSubmitting || store.isStreaming}
        />
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || isSubmitting || store.isStreaming}
          className="mt-3 rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {store.isStreaming ? '生成中...' : '开始生成'}
        </button>
      </div>

      {/* Status bar */}
      {store.overallStatus !== 'idle' && (
        <div className="flex items-center gap-2 text-sm">
          <span className="font-medium text-gray-700">状态：</span>
          <span
            className={`px-2 py-0.5 rounded-full text-xs font-medium ${
              store.overallStatus === 'processing'
                ? 'bg-blue-100 text-blue-700'
                : store.overallStatus === 'completed'
                  ? 'bg-green-100 text-green-700'
                  : store.overallStatus === 'failed'
                    ? 'bg-red-100 text-red-700'
                    : store.overallStatus === 'degraded'
                      ? 'bg-amber-100 text-amber-700'
                      : 'bg-gray-100 text-gray-700'
            }`}
          >
            {store.overallStatus === 'processing'
              ? '处理中'
              : store.overallStatus === 'completed'
                ? '已完成'
                : store.overallStatus === 'failed'
                  ? '失败'
                  : store.overallStatus === 'degraded'
                    ? '降级完成'
                    : '未知'}
          </span>
          {store.currentPhase && (
            <span className="text-gray-500">
              当前阶段：{store.currentPhase}
            </span>
          )}
        </div>
      )}

      {/* Agent pipeline */}
      {store.overallStatus !== 'idle' && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {AGENT_ORDER.map((agent) => (
            <AgentProgressCard
              key={agent.key}
              agentType={agent.key}
              status={getAgentStatus(agent.key)}
              agentName={agent.name}
              retryCount={getAgentRetryCount(agent.key)}
            />
          ))}
        </div>
      )}

      {/* Results */}
      {store.generatedResources.length > 0 && (
        <ResourcePreviewList resources={store.generatedResources} />
      )}

      {/* Errors */}
      {store.errors.length > 0 && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <h3 className="text-sm font-medium text-red-800 mb-2">错误</h3>
          <ul className="list-disc list-inside space-y-1">
            {store.errors.map((err, i) => (
              <li key={i} className="text-sm text-red-600">
                {err}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
