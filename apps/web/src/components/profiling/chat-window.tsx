'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import { getUserId } from '@/lib/user-id';
import { useChatStore, type MentorSource } from '@/stores/chat-store';
import { useProfileStore } from '@/stores/profile-store';
import { ChatMessage } from './chat-message';
import { ChatInput } from './chat-input';
import { HistoryDrawer } from './history-drawer';
import { ProfileCard } from './profile-card';
import { SourcePanel } from '@/components/mentor/source-panel';
import { Badge, Card, CardContent } from '@/components/ui';
import type { UserProfile } from '@edumap/shared-types';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export function ChatWindow() {
  const [historyOpen, setHistoryOpen] = useState(false);
  const [sseError, setSseError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const streamingRef = useRef('');
  const lastSentTextRef = useRef('');
  const completedRef = useRef(false);

  const store = useChatStore();
  const { profile, confidenceScores, updateProfile, syncToBackend } = useProfileStore();

  // Derive active messages from session store
  const messages = store.activeSessionId
    ? (store.sessionMessages[store.activeSessionId] || [])
    : [];

  // Ensure a session exists — wait for zustand persist to hydrate first
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => {
    const tryCreate = () => {
      if (!useChatStore.getState().activeSessionId) {
        useChatStore.getState().createSession();
      }
      setHydrated(true);
    };
    // Subscribe to hydration completion
    const unsub = useChatStore.persist.onFinishHydration(tryCreate);
    // If already hydrated, create immediately
    if (useChatStore.persist.hasHydrated()) {
      tryCreate();
    }
    return () => unsub();
  }, []);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    };
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, store.streamingContent]);

  const disconnectSSE = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

  // ── Send handler ──────────────────────────────────────────────────────

  const handleSend = useCallback(
    (text: string) => {
      if (!text || store.isStreaming) return;

      setSseError(null);
      streamingRef.current = '';
      lastSentTextRef.current = text;
      completedRef.current = false;

      const userMsg = {
        id: `user-${Date.now()}`,
        type: 'user' as const,
        role: 'user' as const,
        content: text,
        timestamp: Date.now(),
      };
      store.addMessage(userMsg);

      store.setStreaming(true);
      disconnectSSE();

      // Unified mode → connect to backend-core analysis SSE
      const url = `${API_BASE}/api/v1/analysis/stream/${getUserId()}?message=${encodeURIComponent(text)}`;
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.addEventListener('token', (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          const content = data.content ?? '';
          if (content) {
            streamingRef.current += content;
            store.setStreamingContent(streamingRef.current);
          }
        } catch {
          if (event.data && typeof event.data === 'string') {
            streamingRef.current += event.data;
            store.setStreamingContent(streamingRef.current);
          }
        }
      });

      es.addEventListener('source', (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          if (data.sources) {
            store.setSources(data.sources);
          }
        } catch {
          // Ignore
        }
      });

      es.addEventListener('profile_update', (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          const profileData = data.profile ?? data;
          const confidence = data.confidence_scores ?? {};
          if (profileData) {
            updateProfile(profileData as UserProfile, confidence);
            // 后端分析出新的 6 维画像后立即同步到 profile-service，
            // 让推荐路由（学习路径/通知）能读到；失败仅 warn 不阻塞 UI
            void syncToBackend();
          }
        } catch {
          // Ignore parse errors
        }
      });

      // Listen for backend error events (if backend sends them)
      es.addEventListener('error', (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          const errContent = data.content ?? data.message ?? '服务返回错误';
          setSseError(errContent);
        } catch {
          // Ignore parse failures for raw error events
        }
        store.setStreaming(false);
        streamingRef.current = '';
        store.setStreamingContent('');
        es.close();
        eventSourceRef.current = null;
      });

      es.addEventListener('complete', (event: MessageEvent) => {
        completedRef.current = true;
        try {
          const data = JSON.parse(event.data);
          store.finalizeMessage(data.sources || []);
        } catch {
          store.finalizeMessage();
        }
        es.close();
        eventSourceRef.current = null;
      });

      es.onerror = () => {
        // Ignore clean closures after a complete event
        if (completedRef.current) return;

        const state = es.readyState;
        if (state === EventSource.CONNECTING) {
          // Transient connection issue — the browser will retry
          setSseError('正在连接服务，请稍候...');
        } else if (state === EventSource.CLOSED) {
          // Connection failed permanently
          const lastContent = streamingRef.current.slice(-100);
          setSseError(
            lastContent
              ? `连接中断: ${lastContent}`
              : '无法连接到服务（http://localhost:8000），请确认后端服务是否运行中',
          );
        } else {
          setSseError('连接异常，请尝试重试');
        }
        store.setStreaming(false);
        streamingRef.current = '';
        store.setStreamingContent('');
        es.close();
        eventSourceRef.current = null;
      };
    },
    [store, disconnectSSE, updateProfile],
  );

  // ── Retry handler ────────────────────────────────────────────────────
  const handleRetry = useCallback(() => {
    const lastText = lastSentTextRef.current;
    if (lastText) {
      handleSend(lastText);
    }
  }, [handleSend]);

  const hasContent = messages.length > 0 || store.streamingContent !== '';

  return (
    <>
      {/* History drawer */}
      <HistoryDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} />

      <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
        {/* Left: Chat area */}
        <Card className="flex flex-1 flex-col min-h-[500px] lg:min-h-[600px] overflow-hidden">
          {/* Chat header */}
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setHistoryOpen(true)}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
              >
                📜 历史
                {store.sessions.length > 0 && (
                  <Badge variant="default" size="sm">{store.sessions.length}</Badge>
                )}
              </button>
              <div
                className={`h-2 w-2 rounded-full ${store.isStreaming ? 'bg-success' : 'bg-text-light'}`}
              />
              <span className="text-xs font-medium text-text-secondary">
                {store.isStreaming
                  ? 'AI 正在回答...'
                  : sseError
                    ? '连接错误'
                    : '就绪'}
              </span>
            </div>
            <button
              onClick={() => {
                // Start a fresh conversation — old one stays in history.
                disconnectSSE();
                store.createSession();
              }}
              className="text-xs text-text-light hover:text-text-primary transition-colors"
            >
              ✨ 新建对话
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-4 p-4">
            {!hasContent ? (
              <div className="flex h-full flex-col items-center justify-center text-center">
                <div className="mb-4 text-5xl">💬</div>
                <p className="text-sm text-text-secondary">
                  输入你的学习背景和目标，AI 将为你构建个性化画像并回答学习问题
                </p>
                <p className="mt-1 text-xs text-text-light">
                  例如：「我目前在学习数据结构和算法，动态规划和贪心算法有什么区别？」
                </p>
              </div>
            ) : (
              <>

                {/* Rendered messages */}
                {messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}

                {/* Streaming message */}
                {store.streamingContent && (
                  <ChatMessage
                    message={{
                      id: 'streaming',
                      type: 'ai_text',
                      content: store.streamingContent,
                      timestamp: Date.now(),
                    }}
                    isStreaming
                  />
                )}

                {/* Error */}
                {sseError && (
                  <div className="flex flex-col items-center gap-2 px-4 py-3">
                    <ChatMessage
                      message={{
                        id: 'error-msg',
                        type: 'error',
                        content: sseError,
                        timestamp: Date.now(),
                        source: 'SSE',
                      }}
                    />
                    {!store.isStreaming && (
                      <button
                        onClick={handleRetry}
                        className="rounded-lg bg-brand px-4 py-1.5 text-xs font-medium text-white hover:bg-brand-hover transition-colors"
                      >
                        重新连接
                      </button>
                    )}
                  </div>
                )}
              </>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <ChatInput onSend={handleSend} isStreaming={store.isStreaming} />
        </Card>

        {/* Right panel — unified: profile + sources */}
        <div className="w-full lg:w-80 xl:w-96 space-y-4">
          <div className="sticky top-4 space-y-4">
            <ProfileCard />
            <SourcePanel sources={store.sources} />
          </div>
        </div>
      </div>
    </>
  );
}
