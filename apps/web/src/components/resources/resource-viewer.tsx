'use client';

import { useState } from 'react';

interface ResourceItem {
  id: string;
  name: string;
  type: string;
  content?: string;
  kp_id?: string;
  description?: string;
}

interface ResourceViewerProps {
  kpId?: string;
  kpName?: string;
  contentType?: string;
  content?: string;
  resources?: ResourceItem[];
  isLoading?: boolean;
  onClose?: () => void;
  onResourceClick?: (resource: ResourceItem) => void;
}

const TYPE_LABELS: Record<string, string> = {
  explanation: '讲解',
  exercise: '练习',
  visualization: '可视化',
  code: '代码',
  upload: '上传文件',
  reading: '阅读',
};

const TYPE_COLORS: Record<string, string> = {
  explanation: 'bg-blue-100 text-blue-700',
  exercise: 'bg-green-100 text-green-700',
  visualization: 'bg-purple-100 text-purple-700',
  code: 'bg-orange-100 text-orange-700',
  upload: 'bg-gray-100 text-gray-700',
};

export function ResourceViewer({
  kpId,
  kpName,
  contentType,
  content,
  resources,
  isLoading,
  onClose,
  onResourceClick,
}: ResourceViewerProps) {
  const [selectedResource, setSelectedResource] = useState<ResourceItem | null>(null);

  if (!kpId) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-6">
        <p className="text-center text-sm text-text-light">
          <span className="text-2xl block mb-2">📖</span>
          选择技能树中的知识点查看学习资源
        </p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-bg-card p-6">
        <h3 className="text-sm font-medium text-text-primary mb-3">{kpName || kpId}</h3>
        <div className="animate-pulse space-y-2">
          <div className="h-4 rounded bg-bg-secondary w-3/4" />
          <div className="h-4 rounded bg-bg-secondary w-1/2" />
          <div className="h-4 rounded bg-bg-secondary w-2/3" />
        </div>
      </div>
    );
  }

  const activeResource = selectedResource;
  const displayResources = resources && resources.length > 0;
  const hasContent = content || activeResource?.description;

  return (
    <div className="rounded-xl border border-border bg-bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <h3 className="text-sm font-medium text-text-primary truncate">
            {activeResource ? activeResource.name : (kpName || kpId)}
          </h3>
          {(contentType || activeResource?.type) && (
            <span
              className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                TYPE_COLORS[contentType || activeResource?.type || ''] || 'bg-bg-secondary text-text-secondary'
              }`}
            >
              {TYPE_LABELS[contentType || activeResource?.type || ''] || contentType || activeResource?.type}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {activeResource && (
            <button
              onClick={() => setSelectedResource(null)}
              className="text-xs text-text-light hover:text-text-primary transition-colors"
            >
              返回列表
            </button>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="text-xs text-text-light hover:text-text-primary transition-colors ml-2"
            >
              关闭
            </button>
          )}
        </div>
      </div>

      {/* Resource list */}
      {displayResources && !activeResource && (
        <div className="space-y-1 mb-3 max-h-48 overflow-y-auto">
          {resources!.map((r) => (
            <button
              key={r.id}
              onClick={() => {
                setSelectedResource(r);
                onResourceClick?.(r);
              }}
              className="w-full text-left flex items-center gap-2 rounded-lg px-3 py-2 text-sm
                         hover:bg-bg-secondary transition-colors"
            >
              <span
                className={`shrink-0 rounded px-1 py-0.5 text-[10px] font-medium ${
                  TYPE_COLORS[r.type] || 'bg-bg-secondary text-text-secondary'
                }`}
              >
                {TYPE_LABELS[r.type] || r.type}
              </span>
              <span className="truncate text-text-primary">{r.name}</span>
            </button>
          ))}
        </div>
      )}

      {/* Content */}
      {hasContent ? (
        <div className="prose prose-sm max-w-none text-text-primary whitespace-pre-wrap text-sm max-h-64 overflow-y-auto">
          {activeResource?.description || content}
        </div>
      ) : displayResources && !activeResource ? (
        <p className="text-xs text-text-light text-center py-2">
          点击上方资源查看详情
        </p>
      ) : (
        <p className="text-xs text-text-light text-center py-4">
          该知识点暂无生成资源，请使用"生成资源"功能创建
        </p>
      )}
    </div>
  );
}
