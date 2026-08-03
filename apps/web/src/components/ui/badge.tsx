'use client';

import { cn } from '@/lib/utils';

interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'brand';
  size?: 'sm' | 'md';
  children: React.ReactNode;
  className?: string;
}

const VARIANT_STYLES = {
  default: 'bg-bg-secondary text-text-secondary',
  success: 'bg-success-bg text-success',
  warning: 'bg-warning-bg text-warning',
  danger: 'bg-danger-bg text-danger',
  info: 'bg-info/10 text-info',
  brand: 'bg-brand/10 text-brand',
};

export function Badge({
  variant = 'default',
  size = 'sm',
  children,
  className,
}: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full font-medium',
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-sm',
        VARIANT_STYLES[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
