'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import { Logo } from './logo';
import { CourseContext } from './course-context';
import { SidebarNav } from './sidebar-nav';
import { UserSection } from './user-section';

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        'flex shrink-0 flex-col bg-bg-secondary border-r border-border transition-all duration-300 sticky top-0 h-screen',
        collapsed ? 'w-14' : 'w-60',
        className,
      )}
    >
      {/* Toggle button — top-right corner */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className={cn(
          'absolute -right-3 top-16 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-bg-card text-text-light shadow-sm hover:text-text-primary transition-colors',
          collapsed && 'rotate-180',
        )}
        aria-label={collapsed ? '展开侧边栏' : '折叠侧边栏'}
      >
        <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      {/* Logo + Brand */}
      <Logo collapsed={collapsed} />

      {/* Course context */}
      <CourseContext collapsed={collapsed} />

      {/* Navigation */}
      <SidebarNav collapsed={collapsed} />

      {/* User section */}
      <UserSection collapsed={collapsed} />
    </aside>
  );
}
