'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { href: '/chat', label: '对话', icon: '💬' },
  { href: '/learn/cs201', label: '图谱', icon: '🗺️' },
  { href: '/generate', label: '资源库', icon: '📚' },
  { href: '/profile', label: '仪表盘', icon: '📊' },
] as const;

interface SidebarNavProps {
  collapsed: boolean;
}

export function SidebarNav({ collapsed }: SidebarNavProps) {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  return (
    <nav className="flex-1 space-y-1 px-3 py-4">
      {NAV_ITEMS.map((item) => {
        const active = isActive(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
              active
                ? 'bg-brand/10 text-brand'
                : 'text-text-secondary hover:bg-bg-secondary hover:text-text-primary',
            )}
            title={collapsed ? item.label : undefined}
          >
            <span className="flex h-5 w-5 items-center justify-center text-base shrink-0">
              {item.icon}
            </span>
            {!collapsed && (
              <span className="truncate">{item.label}</span>
            )}
          </Link>
        );
      })}
    </nav>
  );
}
