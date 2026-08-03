'use client';

import { useColdStart, type ColdStartResult } from '@/hooks/use-cold-start';
import Link from 'next/link';

function BannerContent({ result }: { result: ColdStartResult }) {
  if (result.verdict === 'sufficient') {
    return (
      <div className="rounded-lg border border-success/20 bg-success-bg px-4 py-3 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-lg">✅</span>
          <span className="font-medium text-success">画像就绪</span>
          <span className="text-text-secondary">
            您的学习画像已完善，可以开始使用全部功能
          </span>
        </div>
      </div>
    );
  }

  if (result.verdict === 'moderate') {
    return (
      <div className="rounded-lg border border-warning/20 bg-warning-bg px-4 py-3 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-lg">📊</span>
          <span className="font-medium text-warning">数据有限</span>
          <span className="text-text-secondary">
            上传参考文档或继续与 AI 对话以完善画像，生成质量将更佳
          </span>
        </div>
        <div className="mt-2 flex gap-2">
          <Link
            href="/generate"
            className="rounded bg-warning px-3 py-1 text-xs font-medium text-white hover:opacity-90"
          >
            上传资料
          </Link>
          <Link
            href="/chat"
            className="rounded border border-warning/30 px-3 py-1 text-xs font-medium text-warning hover:bg-warning/10"
          >
            去对话
          </Link>
        </div>
      </div>
    );
  }

  // insufficient
  return (
    <div className="rounded-lg border border-danger/20 bg-danger-bg px-4 py-3 text-sm">
      <div className="flex items-center gap-2">
        <span className="text-lg">⚠️</span>
        <span className="font-medium text-danger">语料不足</span>
        <span className="text-text-secondary">
          请先上传至少一份参考文档，或与 AI 对话以提供学习背景
        </span>
      </div>
      <div className="mt-2 flex gap-2">
        <Link
          href="/generate"
          className="rounded bg-danger px-3 py-1 text-xs font-medium text-white hover:opacity-90"
        >
          上传资料
        </Link>
        <Link
          href="/chat"
          className="rounded border border-danger/30 px-3 py-1 text-xs font-medium text-danger hover:opacity-90"
        >
          对话完善
        </Link>
      </div>
    </div>
  );
}

export function ColdStartBanner({ userId }: { userId?: string }) {
  const { loading, result, error, refresh } = useColdStart(userId);

  if (loading) {
    return (
      <div className="rounded-lg border border-border bg-bg-card px-4 py-3 text-sm text-text-light">
        正在评估学习画像...
      </div>
    );
  }

  if (error || !result) {
    return null; // Silently fail — non-blocking
  }

  return (
    <div className="relative">
      <button
        onClick={refresh}
        className="absolute right-1 top-1 rounded p-1 text-text-light hover:text-text-primary hover:bg-bg-secondary transition-colors"
        title="重新评估"
      >
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
      <BannerContent result={result} />
    </div>
  );
}
