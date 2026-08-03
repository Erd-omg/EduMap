'use client';

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Page error:', error);
  }, [error]);

  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center gap-4 px-4 text-center">
      <div className="text-5xl">😵</div>
      <h2 className="text-lg font-semibold text-text-primary">出错了</h2>
      <p className="max-w-md text-sm text-text-secondary">
        页面遇到了一个意外错误。请尝试刷新页面，或返回首页。
      </p>
      <div className="flex gap-3">
        <button
          onClick={reset}
          className="rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
        >
          重试
        </button>
        <a
          href="/"
          className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
        >
          返回首页
        </a>
      </div>
      {process.env.NODE_ENV === 'development' && (
        <details className="mt-4 max-w-lg text-left">
          <summary className="cursor-pointer text-xs text-text-light">错误详情</summary>
          <pre className="mt-2 overflow-auto rounded bg-bg-secondary p-3 text-xs text-text-secondary">
            {error.message}
            {error.stack && `\n\n${error.stack}`}
          </pre>
        </details>
      )}
    </div>
  );
}
