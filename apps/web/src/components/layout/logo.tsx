'use client';

import Link from 'next/link';

interface LogoProps {
  collapsed: boolean;
}

export function Logo({ collapsed }: LogoProps) {
  return (
    <Link
      href="/"
      className="flex items-center gap-3 border-b border-border px-4"
      style={{ height: 64 }}
    >
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand text-white text-sm font-bold">
        E
      </div>
      {!collapsed && (
        <div className="flex items-center gap-2 overflow-hidden">
          <span className="text-base font-bold text-text-primary whitespace-nowrap">
            智图 EduMap
          </span>
          <span className="h-2 w-2 rounded-full bg-success" title="在线" />
        </div>
      )}
    </Link>
  );
}
