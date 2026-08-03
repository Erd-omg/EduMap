'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';

interface TabItem {
  label: string;
  href: string;
  icon: string;
}

const TABS: TabItem[] = [
  { label: '对话', href: '/chat', icon: '💬' },
  { label: '图谱', href: '/learn/cs201', icon: '🌐' },
  { label: '资源', href: '/generate', icon: '📚' },
  { label: '更多', href: '/dashboard', icon: '⚙️' },
];

export function MobileTabBar() {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === '/chat') return pathname.startsWith('/chat') || pathname.startsWith('/mentor');
    if (href === '/learn/cs201') return pathname.startsWith('/learn');
    if (href === '/generate') return pathname.startsWith('/generate');
    if (href === '/dashboard') return pathname.startsWith('/dashboard') || pathname.startsWith('/settings') || pathname.startsWith('/profile');
    return pathname === href;
  };

  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 block border-t border-border bg-bg-card md:hidden">
      <div className="flex items-center justify-around h-14">
        {TABS.map((tab) => {
          const active = isActive(tab.href);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`flex flex-col items-center justify-center gap-0.5 px-3 py-1.5 text-xs transition-colors ${
                active
                  ? 'text-brand'
                  : 'text-text-light hover:text-text-secondary'
              }`}
              aria-current={active ? 'page' : undefined}
            >
              <span className="text-lg leading-none">{tab.icon}</span>
              <span>{tab.label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
