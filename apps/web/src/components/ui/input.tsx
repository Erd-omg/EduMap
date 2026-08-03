'use client';

import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  icon?: React.ReactNode;
  error?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, icon, error, ...props }, ref) => {
    return (
      <div className="space-y-1.5">
        {label && (
          <label className="block text-sm font-medium text-text-secondary">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-text-light">
              {icon}
            </div>
          )}
          <input
            ref={ref}
            className={cn(
              'w-full rounded-lg border border-border bg-bg-card px-4 py-2.5 text-sm text-text-primary',
              'placeholder:text-text-light',
              'transition-colors duration-150',
              'focus:border-brand focus:ring-1 focus:ring-brand/30 focus:outline-none',
              'disabled:cursor-not-allowed disabled:bg-bg-secondary disabled:text-text-light',
              icon && 'pl-10',
              error && 'border-danger focus:border-danger focus:ring-danger/30',
              className,
            )}
            {...props}
          />
        </div>
        {error && <p className="text-xs text-danger">{error}</p>}
      </div>
    );
  },
);
Input.displayName = 'Input';

export { Input };
