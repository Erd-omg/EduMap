'use client';

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui';

interface CourseContextProps {
  collapsed: boolean;
}

export function CourseContext({ collapsed }: CourseContextProps) {
  return (
    <div
      className={cn(
        'border-b border-border',
        collapsed ? 'px-3 py-4' : 'px-4 py-3',
      )}
    >
      {collapsed ? (
        <button
          className="flex w-full items-center justify-center rounded-lg p-2 text-text-secondary hover:bg-bg-secondary transition-colors"
          title="选择课程"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
          </svg>
        </button>
      ) : (
        <>
          {/* Course selector */}
          <button className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left hover:bg-bg-secondary transition-colors">
            <span className="text-sm font-medium text-text-primary flex-1 truncate">
              人工智能导论
            </span>
            <svg className="h-4 w-4 shrink-0 text-text-light" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Progress bar */}
          <div className="mt-2 px-3">
            <div className="flex items-center justify-between text-xs text-text-secondary mb-1">
              <span>学习进度</span>
              <span>68%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-border">
              <div
                className="h-full rounded-full bg-brand transition-all duration-500"
                style={{ width: '68%' }}
              />
            </div>
          </div>

          {/* Upload button — moved to /generate (资源库) */}
        </>
      )}
    </div>
  );
}
