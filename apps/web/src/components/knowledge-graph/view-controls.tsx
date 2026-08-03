'use client';

import { cn } from '@/lib/utils';

export type GraphLayout = 'hierarchical' | 'force';

interface ViewControlsProps {
  layout: GraphLayout;
  onLayoutChange: (layout: GraphLayout) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  isFocusMode: boolean;
  onExitFocus: () => void;
  focusedNodeName?: string;
  relatedCount?: number;
}

export function ViewControls({
  layout,
  onLayoutChange,
  zoomIn,
  zoomOut,
  resetZoom,
  isFocusMode,
  onExitFocus,
  focusedNodeName,
  relatedCount,
}: ViewControlsProps) {
  return (
    <>
      {/* Focus mode back button */}
      {isFocusMode && (
        <div className="absolute left-4 top-4 z-10">
          <button
            onClick={onExitFocus}
            className="flex items-center gap-2 rounded-xl border border-border bg-bg-card/80 px-4 py-2 text-sm font-medium text-text-primary shadow-sm backdrop-blur-sm hover:bg-bg-card transition-colors"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            返回全局图谱
          </button>
          {focusedNodeName && (
            <span className="ml-3 text-xs text-text-light">
              聚焦: {focusedNodeName}
              {relatedCount !== undefined && ` · ${relatedCount} 个关联节点`}
            </span>
          )}
        </div>
      )}

      {/* Toolbar — bottom-right */}
      <div className="absolute bottom-4 right-4 z-10 flex items-center gap-1 rounded-xl border border-border bg-bg-card shadow-sm">
        {/* Layout toggle */}
        <button
          onClick={() => onLayoutChange(layout === 'hierarchical' ? 'force' : 'hierarchical')}
          className="rounded-lg p-2 text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
          title={`切换布局 (当前: ${layout === 'hierarchical' ? '层次' : '力导向'})`}
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
          </svg>
        </button>

        <div className="h-5 w-px bg-border" />

        {/* Zoom controls */}
        <button
          onClick={zoomIn}
          className="rounded-lg p-2 text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
          title="放大"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
        </button>
        <button
          onClick={zoomOut}
          className="rounded-lg p-2 text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
          title="缩小"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M20 12H4" />
          </svg>
        </button>
        <button
          onClick={resetZoom}
          className="rounded-lg p-2 text-xs text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
          title="重置缩放"
        >
          1:1
        </button>
      </div>
    </>
  );
}
