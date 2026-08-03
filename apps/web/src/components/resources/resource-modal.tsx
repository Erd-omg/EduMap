'use client';

import { useEffect, useCallback } from 'react';

interface ResourceModalProps {
  isOpen: boolean;
  title: string;
  content: string;
  contentType?: string;
  onClose: () => void;
}

export function ResourceModal({ isOpen, title, content, contentType, onClose }: ResourceModalProps) {
  const handleEsc = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleEsc);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEsc);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleEsc]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="mx-4 w-full max-w-3xl max-h-[85vh] rounded-2xl border border-border bg-white shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-3 min-w-0">
            <h2 className="text-base font-semibold text-text-primary truncate">{title}</h2>
            {contentType && (
              <span className="shrink-0 rounded bg-brand/10 px-2 py-0.5 text-[10px] font-medium text-brand">
                {contentType}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="shrink-0 rounded-lg p-1.5 text-text-light hover:text-text-primary hover:bg-bg-secondary transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <div className="prose prose-sm max-w-none text-text-primary whitespace-pre-wrap">
            {content}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-border flex justify-end">
          <button
            onClick={onClose}
            className="rounded-lg bg-brand px-5 py-2 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}
