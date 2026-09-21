'use client';

import { useState } from 'react';
import type { IntentInfo } from '@/stores/chat-store';

const INTENT_LABELS: Record<IntentInfo['intent'], string> = {
  profile: '画像更新',
  question: '知识问答',
  mixed: '画像+问答',
};

const INTENT_STYLES: Record<IntentInfo['intent'], string> = {
  profile: 'bg-agent-guardian/10 text-agent-guardian border-agent-guardian/30',
  question: 'bg-agent-coder/10 text-agent-coder border-agent-coder/30',
  mixed: 'bg-agent-orchestrator/10 text-agent-orchestrator border-agent-orchestrator/30',
};

const PATH_LABELS: Record<string, string> = {
  cache: '缓存命中',
  rule: '规则快路径',
  fused: '多路融合',
  fallback: '降级默认',
};

interface IntentBadgeProps {
  intent: IntentInfo;
  /** compact = header pill; full = message meta row with tooltip on click */
  variant?: 'compact' | 'full';
}

/**
 * Intent classification badge — surfaces the multi-path intent
 * recognition result (intent · confidence · decision path) emitted
 * by the backend `intent` SSE event.
 */
export function IntentBadge({ intent, variant = 'full' }: IntentBadgeProps) {
  const [open, setOpen] = useState(false);

  const label = INTENT_LABELS[intent.intent] ?? intent.intent;
  const conf = Math.round((intent.confidence ?? 0) * 100);
  const pathLabel = PATH_LABELS[intent.path] ?? intent.path;

  if (variant === 'compact') {
    return (
      <span
        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${INTENT_STYLES[intent.intent] ?? INTENT_STYLES.question}`}
        title={`意图识别: ${label} · 置信度 ${conf}% · ${pathLabel}`}
      >
        🎯 {label} {conf}%
      </span>
    );
  }

  return (
    <span className="relative inline-flex items-center">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium transition-opacity hover:opacity-80 ${INTENT_STYLES[intent.intent] ?? INTENT_STYLES.question}`}
        title="点击查看意图识别详情"
      >
        🎯 {label}
      </button>

      {open && (
        <span className="absolute bottom-full right-0 z-20 mb-1 w-56 rounded-lg border border-border bg-bg-card p-2.5 text-left shadow-lg">
          <span className="block text-[11px] font-semibold text-text-primary">
            意图识别详情
          </span>
          <span className="mt-1 block text-[10px] text-text-secondary">
            意图: {label} · 置信度 {conf}%
          </span>
          <span className="block text-[10px] text-text-secondary">
            决策路径: {pathLabel}
          </span>

          {intent.votes &&
            Object.entries(intent.votes).map(([path, vote]) => (
              <span key={path} className="mt-1 block text-[10px] text-text-light">
                {vote.cached_from
                  ? `缓存自: ${String(vote.cached_from)}`
                  : `${path}: ${String(vote.intent ?? '?')} × ${String(vote.weight ?? 1)} (${Math.round(Number(vote.confidence ?? 0) * 100)}%)`}
              </span>
            ))}
        </span>
      )}
    </span>
  );
}
