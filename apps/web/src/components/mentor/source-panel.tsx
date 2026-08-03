'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import { MentorSource } from '@/stores/chat-store';

interface SourcePanelProps {
  sources: MentorSource[];
  isLoading?: boolean;
}

const TYPE_LABELS: Record<string, string> = {
  chroma: '向量搜索',
  neo4j: '知识图谱',
};

const TYPE_COLORS: Record<string, string> = {
  chroma: 'bg-agent-assessment text-white',
  neo4j: 'bg-agent-planner text-white',
};

export function SourcePanel({ sources, isLoading }: SourcePanelProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (isLoading) {
    return (
      <div className="space-y-2">
        <h3 className="text-xs font-medium text-text-light uppercase tracking-wider">搜索资料中...</h3>
        <div className="animate-pulse space-y-2">
          <div className="h-16 rounded-lg bg-bg-secondary" />
          <div className="h-16 rounded-lg bg-bg-secondary" />
        </div>
      </div>
    );
  }

  if (!sources.length) {
    return (
      <div className="text-center py-8">
        <p className="text-xs text-text-light">暂无来源引用</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-medium text-text-light uppercase tracking-wider">
        来源引用 ({sources.length})
      </h3>
      {sources.map((source) => (
        <div
          key={source.id}
          className="rounded-lg border border-border bg-bg-card transition-colors hover:border-brand cursor-pointer"
          onClick={() => setExpandedId(expandedId === source.id ? null : source.id)}
        >
          <div className="p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-text-primary truncate">
                  {source.name}
                </p>
                <span
                  className={`inline-block mt-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    TYPE_COLORS[source.type] || 'bg-bg-secondary text-text-secondary'
                  }`}
                >
                  {TYPE_LABELS[source.type] || source.type}
                </span>
              </div>
              <div className="flex flex-col items-end shrink-0">
                <div className="flex items-center gap-1">
                  <div
                    className="h-1.5 w-12 rounded-full bg-border overflow-hidden"
                  >
                    <div
                      className="h-full rounded-full bg-brand"
                      style={{ width: `${Math.round(source.score * 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-text-light">
                    {Math.round(source.score * 100)}%
                  </span>
                </div>
              </div>
            </div>

            {expandedId === source.id && source.summary && (
              <p className="mt-2 text-xs text-text-secondary border-t border-border pt-2">
                {source.summary}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
