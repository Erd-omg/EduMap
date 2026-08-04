'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import type { MentorSource } from '@/stores/chat-store';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

const PANEL_WIDTH = 288; // w-72

interface ResourceDetail {
  id: string;
  name: string;
  type: string;
  description?: string | null;
  kp_name?: string | null;
}

interface ChunkPreview {
  index: number;
  text_preview: string;
  char_count: number;
}

interface SourcePopoverProps {
  sources: MentorSource[];
}

/**
 * Citation popover rendered through a portal with fixed positioning.
 *
 * The old implementation used `absolute bottom-full` inside the message
 * scroll container (`overflow-y-auto`), which clipped the panel whenever the
 * trigger sat near the top of the visible area.  Fixed + portal escapes that
 * ancestor entirely; placement flips above/below based on viewport space and
 * is clamped to stay fully on-screen.
 *
 * Sources that carry a `resource_id` (uploaded-material chunks) are clickable
 * and open a modal showing the original material.
 */
export function SourcePopover({ sources }: SourcePopoverProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [placement, setPlacement] = useState<'above' | 'below'>('below');
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // Resource detail modal state
  const [detail, setDetail] = useState<ResourceDetail | null>(null);
  const [chunks, setChunks] = useState<ChunkPreview[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);

  const reposition = useCallback(() => {
    const trigger = triggerRef.current;
    const panel = panelRef.current;
    if (!trigger || !panel) return;
    const rect = trigger.getBoundingClientRect();
    const panelHeight = panel.offsetHeight || 220;
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    // Keep the panel fully on-screen — this is what used to make it look
    // "cut off" inside the scroll container.
    let top = placement === 'above' ? rect.top - panelHeight - 8 : rect.bottom + 8;
    top = Math.max(8, Math.min(top, vh - panelHeight - 8));
    const left = Math.min(Math.max(8, rect.left), vw - PANEL_WIDTH - 8);
    setPosition({ top, left });
  }, [placement]);

  const toggle = useCallback(() => {
    if (isOpen) {
      setIsOpen(false);
      return;
    }
    const trigger = triggerRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    // Trigger in the upper half of the viewport → open downward (and vice
    // versa), so the panel always has room and is never clipped.
    setPlacement(rect.top > window.innerHeight / 2 ? 'above' : 'below');
    setPosition({
      top: rect.bottom + 8,
      left: Math.min(Math.max(8, rect.left), window.innerWidth - PANEL_WIDTH - 8),
    });
    setIsOpen(true);
  }, [isOpen]);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handleClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        panelRef.current && !panelRef.current.contains(target) &&
        triggerRef.current && !triggerRef.current.contains(target)
      ) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [isOpen]);

  // Reposition on scroll / resize while open
  useEffect(() => {
    if (!isOpen) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => {
      window.removeEventListener('scroll', reposition, true);
      window.removeEventListener('resize', reposition);
    };
  }, [isOpen, reposition]);

  const openResource = useCallback(async (source: MentorSource) => {
    if (!source.resource_id) return;
    setDetail({ id: source.resource_id, name: source.name, type: '' });
    setChunks([]);
    setDetailLoading(true);
    try {
      const [metaRes, chunkRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/resources/${source.resource_id}`),
        fetch(`${API_BASE}/api/v1/resources/${source.resource_id}/chunks?limit=10`),
      ]);
      if (metaRes.ok) {
        const meta = await metaRes.json();
        setDetail({
          id: meta.id,
          name: meta.name,
          type: meta.type,
          description: meta.description,
          kp_name: meta.kp_name,
        });
      }
      if (chunkRes.ok) {
        const data = await chunkRes.json();
        setChunks(data.chunks || []);
      }
    } catch {
      // Keep the modal open with whatever we have; ignore fetch errors.
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const closeDetail = useCallback(() => {
    setDetail(null);
    setChunks([]);
  }, []);

  if (!sources.length) return null;

  const panel = isOpen && position
    ? createPortal(
        <div
          ref={panelRef}
          style={{ position: 'fixed', top: position.top, left: position.left, width: PANEL_WIDTH }}
          className="z-50 rounded-xl border border-border bg-white shadow-lg"
        >
          {/* Arrow */}
          <div
            className={`absolute left-4 w-2 h-2 rotate-45 bg-white ${
              placement === 'above'
                ? '-bottom-1 border-r border-b border-border'
                : '-top-1 border-l border-t border-border'
            }`}
          />

          <div className="p-3">
            <p className="text-xs font-semibold text-text-primary mb-2">
              引用来源 ({sources.length})
              <span className="ml-1 font-normal text-text-light">点击可查看原资料</span>
            </p>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {sources.map((source) => {
                const clickable = Boolean(source.resource_id);
                const inner = (
                  <>
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-text-primary truncate">{source.name}</span>
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
                      <p className="mt-1 text-[10px] text-text-light line-clamp-2">{source.summary}</p>
                    )}
                    {clickable && (
                      <span className="mt-1 inline-block text-[10px] text-brand">点击打开原资料 →</span>
                    )}
                  </>
                );
                return clickable ? (
                  <button
                    key={source.id}
                    onClick={() => openResource(source)}
                    className="w-full text-left rounded-lg bg-bg-secondary p-2 text-xs hover:bg-brand/10 transition-colors"
                  >
                    {inner}
                  </button>
                ) : (
                  <div key={source.id} className="rounded-lg bg-bg-secondary p-2 text-xs">
                    {inner}
                  </div>
                );
              })}
            </div>
          </div>
        </div>,
        document.body,
      )
    : null;

  const detailModal = detail
    ? createPortal(
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4"
          onClick={closeDetail}
        >
          <div
            className="max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="min-w-0">
                <h3 className="text-base font-semibold text-text-primary truncate">{detail.name}</h3>
                {(detail.type || detail.kp_name) && (
                  <p className="mt-0.5 text-xs text-text-light">
                    {[detail.kp_name && `知识点: ${detail.kp_name}`, detail.type && `类型: ${detail.type}`]
                      .filter(Boolean)
                      .join(' · ')}
                  </p>
                )}
              </div>
              <button
                onClick={closeDetail}
                className="rounded-lg p-1.5 text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
                aria-label="关闭"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {detailLoading ? (
              <div className="flex items-center justify-center py-8 text-sm text-text-light">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-brand border-t-transparent mr-2" />
                加载内容...
              </div>
            ) : detail.description ? (
              <div className="rounded-lg border border-border bg-bg-secondary p-3 text-xs text-text-primary whitespace-pre-wrap leading-relaxed max-h-72 overflow-y-auto">
                {detail.description}
              </div>
            ) : chunks.length > 0 ? (
              <div className="space-y-3">
                <p className="text-xs text-text-secondary">
                  共 {chunks.length} 个文本段落（每段预览前 200 字符）
                </p>
                {chunks.map((chunk, i) => (
                  <div key={i} className="rounded-lg border border-border bg-bg-secondary p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-text-secondary">段落 #{i + 1}</span>
                      <span className="text-[10px] text-text-light">{chunk.char_count} 字符</span>
                    </div>
                    <p className="text-xs text-text-primary whitespace-pre-wrap leading-relaxed">
                      {chunk.text_preview}
                      {chunk.char_count > 200 && '...'}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-center py-8 text-sm text-text-light">暂无内容</p>
            )}
          </div>
        </div>,
        document.body,
      )
    : null;

  return (
    <div className="relative inline-block">
      <button
        ref={triggerRef}
        onClick={toggle}
        className="inline-flex items-center gap-0.5 text-xs text-text-light hover:text-brand transition-colors"
        title="查看来源"
      >
        📖 <span className="text-[10px]">{sources.length}</span>
      </button>
      {panel}
      {detailModal}
    </div>
  );
}
