'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type MessageType = 'user' | 'ai_text' | 'resource_card' | 'system' | 'progress' | 'error';

export interface ResourceReference {
  type: string;
  title: string;
  kp_id?: string;
}

export interface MentorSource {
  id: string;
  name: string;
  type: 'chroma' | 'neo4j';
  score: number;
  summary: string;
  /** Resource id for chroma sources — enables opening the original material. */
  resource_id?: string | null;
}

export interface Message {
  id: string;
  type: MessageType;
  role?: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  /** For resource_card type */
  resources?: ResourceReference[];
  /** For progress type — which agent */
  agent?: string;
  /** For progress/system type */
  confidence?: number;
  /** For mentor mode — source references */
  sources?: MentorSource[];
  /** For error type */
  source?: string;
}

export interface ConversationSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
  resourceCount: number;
  messagePreview: string;
}

interface SessionData {
  messages: Message[];
  streamingContent: string;
  sources: MentorSource[];
}

interface ChatState {
  // Per-session data keyed by session ID
  sessionMessages: Record<string, Message[]>;
  // Active session tracking
  activeSessionId: string | null;
  // Streaming state for active session
  isStreaming: boolean;
  streamingContent: string;
  sources: MentorSource[];

  // Mode (unified — no toggle, always connects to analysis endpoint)
  mode: 'unified';

  // History
  sessions: ConversationSession[];

  // Actions
  addMessage: (msg: Message) => void;
  setStreaming: (v: boolean) => void;
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (token: string) => void;
  setSources: (sources: MentorSource[]) => void;
  clearMessages: () => void;
  finalizeMessage: (sources?: MentorSource[]) => void;

  // Session management
  createSession: () => string;
  switchSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  renameSession: (sessionId: string, title: string) => void;

  // Derived getter
  getMessages: () => Message[];
}

/** Ensure a session has data entries initialized. */
function ensureSessionData(
  state: { sessionMessages: Record<string, Message[]>; activeSessionId: string | null },
) {
  const id = state.activeSessionId;
  if (!id) return;
  if (!state.sessionMessages[id]) {
    state.sessionMessages[id] = [];
  }
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // Per-session message storage
      sessionMessages: {},
      activeSessionId: null,
      isStreaming: false,
      streamingContent: '',
      sources: [],
      mode: 'unified',
      sessions: [],

      // ── Actions operate on the active session ──────────────────────

      addMessage: (msg) =>
        set((s) => {
          ensureSessionData(s);
          const id = s.activeSessionId!;
          const existing = s.sessionMessages[id] || [];
          const updatedMessages = {
            ...s.sessionMessages,
            [id]: [...existing, msg],
          };

          // Auto-title from first user message
          let updatedSessions = s.sessions;
          if (msg.role === 'user' && existing.length === 0) {
            const title = msg.content.length > 20
              ? msg.content.slice(0, 20) + '…'
              : msg.content;
            updatedSessions = s.sessions.map((sess) =>
              sess.id === id
                ? { ...sess, title, messageCount: 1, messagePreview: msg.content.slice(0, 50), updatedAt: Date.now() }
                : sess,
            );
          } else if (msg.role === 'user') {
            // Increment message count for non-first messages
            updatedSessions = s.sessions.map((sess) =>
              sess.id === id
                ? { ...sess, messageCount: (sess.messageCount || 0) + 1, updatedAt: Date.now() }
                : sess,
            );
          }

          return { sessionMessages: updatedMessages, sessions: updatedSessions };
        }),

      setStreaming: (v) => set({ isStreaming: v }),

      setStreamingContent: (content) => set({ streamingContent: content }),

      appendStreamingContent: (token) =>
        set((s) => ({ streamingContent: s.streamingContent + token })),

      setSources: (sources) => set({ sources }),

      clearMessages: () =>
        set((s) => {
          const id = s.activeSessionId;
          if (!id) return { streamingContent: '', sources: [] };
          return {
            sessionMessages: { ...s.sessionMessages, [id]: [] },
            streamingContent: '',
            sources: [],
          };
        }),

      finalizeMessage: (sources) =>
        set((s) => {
          const content = s.streamingContent;
          if (!content) return { streamingContent: '', isStreaming: false };
          const msg: Message = {
            id: `assistant-${Date.now()}`,
            type: 'ai_text',
            role: 'assistant',
            content,
            timestamp: Date.now(),
            ...(sources && sources.length > 0 ? { sources } : {}),
          };
          const id = s.activeSessionId;
          if (!id) return { streamingContent: '', isStreaming: false };

          // Update session metadata (messageCount, updatedAt)
          const updatedSessions = s.sessions.map((sess) =>
            sess.id === id
              ? { ...sess, messageCount: (sess.messageCount || 0) + 1, updatedAt: Date.now() }
              : sess,
          );

          return {
            sessionMessages: {
              ...s.sessionMessages,
              [id]: [...(s.sessionMessages[id] || []), msg],
            },
            sessions: updatedSessions,
            streamingContent: '',
            isStreaming: false,
            ...(sources ? { sources: [] } : {}),
          };
        }),

      // ── Session management ─────────────────────────────────────────

      createSession: () => {
        const id = `session-${Date.now()}`;
        const session: ConversationSession = {
          id,
          title: '新会话',
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messageCount: 0,
          resourceCount: 0,
          messagePreview: '',
        };
        set((s) => ({
          activeSessionId: id,
          sessions: [session, ...s.sessions],
          sessionMessages: { ...s.sessionMessages, [id]: [] },
          streamingContent: '',
          sources: [],
        }));
        return id;
      },

      switchSession: (sessionId) => {
        set((s) => {
          // Save current session's streaming state into its messages
          const prevId = s.activeSessionId;
          const updatedMessages = { ...s.sessionMessages };
          if (prevId && s.streamingContent) {
            // Append unsaved streaming content as a message
            const unsaved: Message = {
              id: `auto-${Date.now()}`,
              type: 'ai_text',
              role: 'assistant',
              content: s.streamingContent,
              timestamp: Date.now(),
            };
            updatedMessages[prevId] = [...(updatedMessages[prevId] || []), unsaved];
          }

          return {
            activeSessionId: sessionId,
            // Restore target session's streaming state
            sessionMessages: updatedMessages,
            streamingContent: '',
            sources: [],
          };
        });
      },

      deleteSession: (sessionId) =>
        set((s) => {
          const { [sessionId]: _, ...remaining } = s.sessionMessages;
          return {
            sessions: s.sessions.filter((sess) => sess.id !== sessionId),
            sessionMessages: remaining,
            ...(s.activeSessionId === sessionId
              ? { activeSessionId: null, sources: [] }
              : {}),
          };
        }),

      renameSession: (sessionId, title) =>
        set((s) => ({
          sessions: s.sessions.map((sess) =>
            sess.id === sessionId ? { ...sess, title } : sess,
          ),
        })),

      // ── Derived getter ─────────────────────────────────────────────

      getMessages: () => {
        const { sessionMessages, activeSessionId } = get();
        return activeSessionId ? sessionMessages[activeSessionId] || [] : [];
      },
    }),
    {
      name: 'edumap-chat-storage',
      partialize: (state) => ({
        sessionMessages: state.sessionMessages,
        activeSessionId: state.activeSessionId,
        sessions: state.sessions,
      }),
      // Handle migration from old flat format
      merge: (persisted: unknown, current: ChatState) => {
        const p = persisted as Record<string, unknown>;
        const result = { ...current };

        // Detect old format: had a top-level `messages` array
        if (p.messages && !p.sessionMessages) {
          const oldMessages = p.messages as Message[];
          const legacyId = p.sessionId as string || 'legacy';
          result.sessionMessages = { [legacyId]: oldMessages };
          result.activeSessionId = legacyId;
          result.sessions = (p.sessions as ConversationSession[]) || [];
          if (legacyId === 'legacy' && result.sessions.length === 0) {
            result.sessions = [{
              id: 'legacy',
              title: '历史会话',
              createdAt: Date.now() - 86400000,
              updatedAt: Date.now(),
              messageCount: oldMessages.length,
              resourceCount: 0,
              messagePreview: oldMessages[oldMessages.length - 1]?.content?.slice(0, 50) || '',
            }];
          }
        } else {
          // New format
          if (p.sessionMessages) result.sessionMessages = p.sessionMessages as Record<string, Message[]>;
          if (p.activeSessionId) result.activeSessionId = p.activeSessionId as string;
          if (p.sessions) result.sessions = p.sessions as ConversationSession[];
        }

        return result;
      },
    },
  ),
);
