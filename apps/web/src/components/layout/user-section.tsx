'use client';

import { cn } from '@/lib/utils';

interface UserSectionProps {
  collapsed: boolean;
}

export function UserSection({ collapsed }: UserSectionProps) {
  return (
    <div
      className={cn(
        'flex items-center border-t border-border px-4',
        collapsed ? 'justify-center' : 'gap-3',
      )}
      style={{ height: 72 }}
    >
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand/20 text-sm font-semibold text-brand">
        U
      </div>
      {!collapsed && (
        <div className="flex flex-1 flex-col overflow-hidden">
          <span className="text-sm font-medium text-text-primary truncate">
            用户
          </span>
          <span className="text-xs text-text-light">六维画像</span>
        </div>
      )}
      {!collapsed && (
        <button
          className="rounded-lg p-1.5 text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
          title="退出登录"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
        </button>
      )}
    </div>
  );
}
