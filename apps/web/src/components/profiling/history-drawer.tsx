'use client';

import { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { useChatStore, type ConversationSession } from '@/stores/chat-store';

interface HistoryDrawerProps {
  open: boolean;
  onClose: () => void;
}

export function HistoryDrawer({ open, onClose }: HistoryDrawerProps) {
  const [search, setSearch] = useState('');
  const [isManaging, setIsManaging] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const drawerRef = useRef<HTMLDivElement>(null);

  const { sessions, activeSessionId, switchSession, deleteSession, createSession } =
    useChatStore();

  // Close on ESC
  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  // Close on backdrop click
  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (drawerRef.current && !drawerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    // Delay to prevent the open trigger from immediately closing
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClick);
    }, 100);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClick);
    };
  }, [open, onClose]);

  const filtered = search
    ? sessions.filter((s) => s.title.includes(search))
    : sessions;
  // Exclude the active session from the history list
  const historySessions = filtered.filter((s) => s.id !== activeSessionId);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleBatchDelete = () => {
    if (selectedIds.size === 0) return;
    if (!confirm(`确定删除选中的 ${selectedIds.size} 个会话？`)) return;
    selectedIds.forEach((id) => deleteSession(id));
    setSelectedIds(new Set());
    setIsManaging(false);
  };

  const formatDate = (ts: number) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 86400000) return '今天';
    if (diff < 172800000) return '昨天';
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}天前`;
    return `${d.getMonth() + 1}/${d.getDate()}`;
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/20 backdrop-blur-sm animate-[fade-in_0.15s_ease-out]">
      <div
        ref={drawerRef}
        className="w-80 bg-bg-card border-l border-border h-full shadow-xl animate-[slide-in-right_0.2s_ease-out] flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h3 className="text-sm font-semibold text-text-primary">📜 对话历史</h3>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-text-light hover:text-text-primary hover:bg-bg-secondary transition-colors"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Search */}
        <div className="px-3 py-2">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="🔍 搜索会话..."
            className="w-full rounded-lg border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary placeholder:text-text-light outline-none focus:border-brand transition-colors"
          />
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto px-3">
          {/* Current session */}
          {activeSessionId && (
            <div className="mb-3">
              <div className="text-xs font-medium text-text-light mb-1 px-2">
                📌 当前会话
              </div>
              <CurrentSessionCard
                session={sessions.find((s) => s.id === activeSessionId)}
              />
            </div>
          )}

          {/* History sessions */}
          {historySessions.length > 0 && (
            <div>
              <div className="text-xs font-medium text-text-light mb-1 px-2">
                📂 历史会话
              </div>
              <div className="space-y-1">
                {historySessions.map((s) => (
                  <SessionCard
                    key={s.id}
                    session={s}
                    isActive={false}
                    isManaging={isManaging}
                    isSelected={selectedIds.has(s.id)}
                    onSelect={() => {
                      if (isManaging) {
                        toggleSelect(s.id);
                      } else {
                        switchSession(s.id);
                        onClose();
                      }
                    }}
                    onDelete={() => deleteSession(s.id)}
                  />
                ))}
              </div>
            </div>
          )}

          {!activeSessionId && historySessions.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <p className="text-sm text-text-light">暂无对话记录</p>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="border-t border-border px-3 py-3 flex gap-2">
          {isManaging ? (
            <>
              <button
                onClick={() => {
                  setIsManaging(false);
                  setSelectedIds(new Set());
                }}
                className="flex-1 rounded-lg border border-border px-3 py-2 text-xs text-text-secondary hover:bg-bg-secondary transition-colors"
              >
                取消
              </button>
              <button
                onClick={handleBatchDelete}
                disabled={selectedIds.size === 0}
                className="flex-1 rounded-lg bg-danger px-3 py-2 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                删除所选 ({selectedIds.size})
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => {
                  createSession();
                  onClose();
                }}
                className="flex-1 rounded-lg bg-brand px-3 py-2 text-xs font-medium text-white hover:bg-brand-hover transition-colors"
              >
                ＋ 新建会话
              </button>
              <button
                onClick={() => setIsManaging(true)}
                className="rounded-lg border border-border px-3 py-2 text-xs text-text-secondary hover:bg-bg-secondary transition-colors"
              >
                🗑️ 管理
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function CurrentSessionCard({ session }: { session?: ConversationSession }) {
  if (!session) {
    return (
      <div className="rounded-lg border border-border bg-bg-secondary/50 p-3">
        <p className="text-xs text-text-light">当前无活跃会话</p>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-brand/20 bg-brand/5 p-3">
      <p className="text-sm font-medium text-text-primary line-clamp-1">
        {session.title}
      </p>
      <p className="text-xs text-text-light mt-0.5">{session.messageCount} 条消息</p>
    </div>
  );
}

function SessionCard({
  session,
  isActive,
  isManaging,
  isSelected,
  onSelect,
  onDelete,
}: {
  session: ConversationSession;
  isActive: boolean;
  isManaging?: boolean;
  isSelected?: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const formatDate = (ts: number) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 86400000) return '今天';
    if (diff < 172800000) return '昨天';
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}天前`;
    return `${d.getMonth() + 1}/${d.getDate()}`;
  };

  return (
    <div
      className={cn(
        'group flex items-start gap-2 rounded-lg p-2.5 cursor-pointer transition-colors',
        isManaging
          ? isSelected
            ? 'bg-brand/10'
            : 'hover:bg-bg-secondary'
          : isActive
            ? 'bg-brand/5'
            : 'hover:bg-bg-secondary',
      )}
      onClick={onSelect}
    >
      {isManaging && (
        <div className="shrink-0 pt-0.5" onClick={(e) => e.stopPropagation()}>
          <input
            type="checkbox"
            checked={isSelected || false}
            onChange={() => onSelect()}
            className="h-4 w-4 rounded border-border accent-brand cursor-pointer"
          />
        </div>
      )}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-text-primary truncate">
          {session.title}
        </p>
        <p className="text-xs text-text-light mt-0.5">
          {formatDate(session.createdAt)}
          {session.messageCount > 0 && (
            <> · {session.messageCount} 条消息</>
          )}
        </p>
      </div>
      {!isManaging && !isActive && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="shrink-0 rounded p-1 text-text-light opacity-0 group-hover:opacity-100 hover:text-danger transition-all"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      )}
    </div>
  );
}
