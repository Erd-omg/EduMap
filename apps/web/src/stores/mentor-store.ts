'use client';

import { create } from 'zustand';

export interface MentorMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: MentorSource[];
  timestamp: number;
}

export interface MentorSource {
  id: string;
  name: string;
  type: 'chroma' | 'neo4j';
  score: number;
  summary: string;
}

interface MentorState {
  messages: MentorMessage[];
  sources: MentorSource[];
  isStreaming: boolean;
  error: string | null;

  // Actions
  sendQuery: (query: string) => Promise<void>;
  addUserMessage: (content: string) => void;
  addAssistantMessage: (content: string) => void;
  setSources: (sources: MentorSource[]) => void;
  appendToken: (token: string) => void;
  finalizeMessage: (sources: MentorSource[]) => void;
  setStreaming: (v: boolean) => void;
  setError: (err: string | null) => void;
  clearMessages: () => void;
}

const MENTOR_BASE_URL = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export const useMentorStore = create<MentorState>((set, get) => ({
  messages: [],
  sources: [],
  isStreaming: false,
  error: null,

  sendQuery: async (query: string) => {
    const state = get();
    if (!query.trim() || state.isStreaming) return;

    // Clear error and add user message
    set({ error: null, isStreaming: true, sources: [] });
    state.addUserMessage(query);

    // Start SSE connection
    const url = `${MENTOR_BASE_URL}/api/v1/mentor/stream/anonymous?query=${encodeURIComponent(query)}`;
    const es = new EventSource(url);

    es.addEventListener('source', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.sources) {
          set({ sources: data.sources });
        }
      } catch {
        // Ignore parse errors
      }
    });

    es.addEventListener('token', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.content) {
          get().appendToken(data.content);
        }
      } catch {
        if (typeof event.data === 'string') {
          get().appendToken(event.data);
        }
      }
    });

    es.addEventListener('complete', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        get().finalizeMessage(data.sources || []);
      } catch {
        get().finalizeMessage([]);
      }
      es.close();
    });

    es.addEventListener('error', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        set({ error: data.content || 'Mentor 服务返回错误' });
      } catch {
        set({ error: 'Mentor 服务连接失败' });
      }
      set({ isStreaming: false });
      es.close();
    });

    es.onerror = () => {
      set({ error: '无法连接到 Mentor 服务', isStreaming: false });
      es.close();
    };
  },

  addUserMessage: (content: string) =>
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id: `user-${Date.now()}`,
          role: 'user',
          content,
          timestamp: Date.now(),
        },
      ],
    })),

  addAssistantMessage: (content: string) =>
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content,
          timestamp: Date.now(),
        },
      ],
    })),

  setSources: (sources) => set({ sources }),

  appendToken: (token: string) =>
    set((s) => {
      const msgs = [...s.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === 'assistant') {
        msgs[msgs.length - 1] = {
          ...last,
          content: last.content + token,
        };
      } else {
        msgs.push({
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: token,
          timestamp: Date.now(),
        });
      }
      return { messages: msgs };
    }),

  finalizeMessage: (sources) =>
    set((s) => {
      const msgs = [...s.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, sources };
      }
      return { messages: msgs, isStreaming: false, sources };
    }),

  setStreaming: (v) => set({ isStreaming: v }),
  setError: (err) => set({ error: err }),

  clearMessages: () => set({ messages: [], sources: [], error: null }),
}));
