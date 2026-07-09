'use client';

import { useState } from 'react';
import { MentorSource } from '@/stores/mentor-store';

interface SourcePanelProps {
  sources: MentorSource[];
  isLoading?: boolean;
}

const TYPE_LABELS: Record<string, string> = {
  chroma: '向量搜索',
  neo4j: '知识图谱',
};

const TYPE_COLORS: Record<string, string> = {
  chroma: 'bg-purple-100 text-purple-600',
  neo4j: 'bg-blue-100 text-blue-600',
};

export function SourcePanel({ sources, isLoading }: SourcePanelProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (isLoading) {
    return (
      <div className="space-y-2">
        <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider">搜索资料中...</h3>
        <div className="animate-pulse space-y-2">
          <div className="h-16 rounded-lg bg-gray-100" />
          <div className="h-16 rounded-lg bg-gray-100" />
        </div>
      </div>
    );
  }

  if (!sources.length) {
    return (
      <div className="text-center py-8">
        <p className="text-xs text-gray-400">暂无来源引用</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider">
        来源引用 ({sources.length})
      </h3>
      {sources.map((source) => (
        <div
          key={source.id}
          className="rounded-lg border border-gray-200 bg-white transition-colors hover:border-blue-200 cursor-pointer"
          onClick={() => setExpandedId(expandedId === source.id ? null : source.id)}
        >
          <div className="p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {source.name}
                </p>
                <span
                  className={`inline-block mt-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    TYPE_COLORS[source.type] || 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {TYPE_LABELS[source.type] || source.type}
                </span>
              </div>
              <div className="flex flex-col items-end shrink-0">
                <div className="flex items-center gap-1">
                  <div
                    className="h-1.5 w-12 rounded-full bg-gray-200 overflow-hidden"
                  >
                    <div
                      className="h-full rounded-full bg-blue-500"
                      style={{ width: `${Math.round(source.score * 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-gray-400">
                    {Math.round(source.score * 100)}%
                  </span>
                </div>
              </div>
            </div>

            {expandedId === source.id && source.summary && (
              <p className="mt-2 text-xs text-gray-500 border-t border-gray-100 pt-2">
                {source.summary}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
