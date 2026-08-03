'use client';

import { useEffect, useCallback, useRef, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  className?: string;
  /** Whether to close when clicking the backdrop */
  closeOnBackdrop?: boolean;
}

export function Modal({
  open,
  onClose,
  title,
  children,
  className,
  closeOnBackdrop = true,
}: ModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (open) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [open, handleKeyDown]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm animate-[fade-in_0.2s_ease-out]"
      onClick={(e) => {
        if (closeOnBackdrop && e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={cn(
          'relative w-[90%] max-h-[90%] overflow-auto rounded-2xl bg-bg-card shadow-xl',
          'animate-[fade-in_0.25s_ease-out]',
          className || 'max-w-2xl',
        )}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        {title && (
          <div className="flex items-center justify-between border-b border-border px-6 py-4">
            <h2 className="text-lg font-semibold text-text-primary">{title}</h2>
            <button
              onClick={onClose}
              className="rounded-lg p-1 text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
              aria-label="关闭"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
}
