'use client';

import { useState, useRef, useEffect } from 'react';
import type { MentorSource } from '@/stores/chat-store';

interface SourcePopoverProps {
  sources: MentorSource[];
}

export function SourcePopover({ sources }: SourcePopoverProps) {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [isOpen]);

  if (!sources.length) return null;

  return (
    <div ref={ref} className="relative inline-block">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center gap-0.5 text-xs text-text-light hover:text-brand transition-colors"
        title="查看来源"
      >
        📖 <span className="text-[10px]">{sources.length}</span>
      </button>

      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 z-50 w-72 rounded-xl border border-border bg-white shadow-lg">
          {/* Arrow */}
          <div className="absolute -bottom-1 left-4 w-2 h-2 rotate-45 border-r border-b border-border bg-white" />

          <div className="p-3">
            <p className="text-xs font-semibold text-text-primary mb-2">
              引用来源 ({sources.length})
            </p>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {sources.map((source) => (
                <div
                  key={source.id}
                  className="rounded-lg bg-bg-secondary p-2 text-xs"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-text-primary truncate">
                      {source.name}
                    </span>
                    <span className="shrink-0 text-[10px] text-text-light">
                      {Math.round(source.score * 100)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className={`rounded px-1 py-0.5 text-[9px] font-medium ${
                      source.type === 'chroma'
                        ? 'bg-purple-100 text-purple-700'
                        : 'bg-blue-100 text-blue-700'
                    }`}>
                      {source.type === 'chroma' ? '向量搜索' : '知识图谱'}
                    </span>
                    <div className="flex-1 h-1 rounded-full bg-border overflow-hidden">
                      <div
                        className="h-full rounded-full bg-brand"
                        style={{ width: `${Math.round(source.score * 100)}%` }}
                      />
                    </div>
                  </div>
                  {source.summary && (
                    <p className="mt-1 text-[10px] text-text-light line-clamp-2">
                      {source.summary}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
