'use client';

import { forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-brand/40 disabled:cursor-not-allowed disabled:opacity-50',
  {
    variants: {
      variant: {
        primary:
          'bg-brand text-white hover:bg-brand-hover active:bg-brand-hover shadow-sm',
        secondary:
          'bg-bg-secondary text-text-primary hover:bg-border active:bg-border',
        ghost:
          'bg-transparent text-text-secondary hover:bg-bg-secondary active:bg-border',
        danger:
          'bg-danger text-white hover:opacity-90 active:opacity-80 shadow-sm',
        accent:
          'bg-accent text-white hover:opacity-90 active:opacity-80 shadow-sm',
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-4 text-sm',
        lg: 'h-12 px-6 text-base',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(buttonVariants({ variant, size, className }))}
        {...props}
      />
    );
  },
);
Button.displayName = 'Button';

export { Button, buttonVariants };
