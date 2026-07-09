'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useChatStore } from '../../stores/chat-store';
import { useProfileStore } from '../../stores/profile-store';
import { RadarChart } from './radar-chart';
import { ProfileCard } from './profile-card';
import type { UserProfile } from '@edumap/shared-types';

const ANALYSIS_BASE_URL =
  'http://localhost:8001/api/v1/analysis/stream/test_user';

function formatTime(timestamp: number): string {
  const d = new Date(timestamp);
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
}

function buildRadarData(
  profile: UserProfile,
  confidence: Record<string, number>,
) {
  const dims = [
    'learning_ability',
    'learning_motivation',
    'knowledge_coverage',
    'interaction_style',
    'focus_characteristics',
    'knowledge_base',
  ];

  return dims.map((key) => {
    const value = (profile as unknown as Record<string, unknown>)[key];
    let score = 0;

    if (typeof value === 'object' && value !== null) {
      const obj = value as Record<string, unknown>;
      if (key === 'knowledge_coverage') {
        const cov = value as {
          mastered: string[];
          learning: string[];
          not_started: string[];
        };
        const total =
          cov.mastered.length + cov.learning.length + cov.not_started.length;
        score = total > 0 ? cov.mastered.length / total : 0;
      } else if (key === 'knowledge_base') {
        const nums = Object.values(obj).filter(
          (v): v is number => typeof v === 'number',
        );
        score =
          nums.length > 0
            ? nums.reduce((a, b) => a + b, 0) / nums.length
            : 0;
      } else {
        const nums = Object.values(obj).filter(
          (v): v is number => typeof v === 'number' && v >= 0 && v <= 1,
        );
        score =
          nums.length > 0
            ? nums.reduce((a, b) => a + b, 0) / nums.length
            : 0;
      }
    } else if (typeof value === 'number') {
      score = value;
    }

    return {
      dimension: key,
      key,
      score: Math.max(0, Math.min(1, score)),
      confidence: confidence[key] ?? 0.5,
    };
  });
}

export function ChatWindow() {
  const [input, setInput] = useState('');
  const [streamingContent, setStreamingContent] = useState('');
  const [sseError, setSseError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const streamingRef = useRef('');

  const { messages, addMessage, isStreaming, setStreaming } = useChatStore();
  const {
    profile,
    confidenceScores,
    updateProfile,
    addMessage: addProfileMsg,
    setAnalyzing,
  } = useProfileStore();

  // Build radar data from profile
  const radarData = profile
    ? buildRadarData(profile, confidenceScores)
    : [];

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    };
  }, []);

  // Auto-scroll when messages or streaming content changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Focus input when not streaming
  useEffect(() => {
    if (!isStreaming) {
      inputRef.current?.focus();
    }
  }, [isStreaming]);

  const disconnectSSE = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput('');
    setSseError(null);
    streamingRef.current = '';

    // Add user message to both stores
    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user' as const,
      content: text,
      timestamp: Date.now(),
    };
    addMessage(userMsg);
    addProfileMsg('user', text);

    // Set streaming state
    setStreaming(true);
    setAnalyzing(true);

    // Connect to SSE
    const url = `${ANALYSIS_BASE_URL}?message=${encodeURIComponent(text)}`;

    // Close any existing connection
    disconnectSSE();

    const es = new EventSource(url);
    eventSourceRef.current = es;

    // Handle connection open
    es.onopen = () => {
      // Connection established
    };

    // Handle named SSE events

    // token event — streaming text
    es.addEventListener('token', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const content = data.content ?? '';
        if (content) {
          streamingRef.current += content;
          setStreamingContent(streamingRef.current);
        }
      } catch {
        // If the data is just plain text, append it directly
        if (event.data && typeof event.data === 'string') {
          streamingRef.current += event.data;
          setStreamingContent(streamingRef.current);
        }
      }
    });

    // profile_update event — update the profile store
    es.addEventListener('profile_update', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const profileData = data.profile ?? data;
        const confidence = data.confidence_scores ?? {};
        if (profileData) {
          updateProfile(profileData as UserProfile, confidence);
        }
      } catch {
        // Silently ignore parse errors for profile updates
      }
    });

    // complete event — finalize the assistant message
    es.addEventListener('complete', (_event: MessageEvent) => {
      const finalContent = streamingRef.current;
      if (finalContent) {
        const assistantMsg = {
          id: `assistant-${Date.now()}`,
          role: 'assistant' as const,
          content: finalContent,
          timestamp: Date.now(),
        };
        addMessage(assistantMsg);
        addProfileMsg('assistant', finalContent);
      }
      streamingRef.current = '';
      setStreamingContent('');
      setStreaming(false);
      setAnalyzing(false);
      es.close();
      eventSourceRef.current = null;
    });

    // error event — backend reported error
    es.addEventListener('error', (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        setSseError(data.content ?? '分析服务返回错误');
      } catch {
        setSseError('分析服务返回错误');
      }
      streamingRef.current = '';
      setStreamingContent('');
      setStreaming(false);
      setAnalyzing(false);
      es.close();
      eventSourceRef.current = null;
    });

    // Connection error (network / server down)
    es.onerror = () => {
      setSseError('无法连接到分析服务，请检查服务是否运行在 localhost:8001');
      setStreaming(false);
      setAnalyzing(false);
      streamingRef.current = '';
      setStreamingContent('');
      es.close();
      eventSourceRef.current = null;
    };
  }, [
    input,
    isStreaming,
    addMessage,
    addProfileMsg,
    setStreaming,
    setAnalyzing,
    updateProfile,
    disconnectSSE,
  ]);

  // Handle keyboard enter
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
      {/* Left column: Chat messages + input */}
      <div className="flex min-h-[500px] flex-1 flex-col rounded-xl border bg-white shadow-sm lg:min-h-[600px]">
        {/* Chat header */}
        <div className="flex items-center gap-2 border-b px-4 py-3">
          <div
            className={`h-2.5 w-2.5 rounded-full ${isStreaming ? 'bg-green-500' : 'bg-gray-300'}`}
          />
          <span className="text-xs font-medium text-gray-500">
            {isStreaming ? 'AI 正在分析中...' : '就绪'}
          </span>
          {sseError && (
            <span className="ml-auto text-xs text-red-500">{sseError}</span>
          )}
        </div>

        {/* Messages area */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && streamingContent === '' && (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-3 text-4xl">💬</div>
              <p className="text-sm text-gray-400">
                输入你的学习背景和目标，AI 将为你构建个性化画像
              </p>
              <p className="mt-1 text-xs text-gray-300">
                例如：&ldquo;我目前在学习数据结构和算法&rdquo;
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-2.5 ${
                  msg.role === 'user'
                    ? 'bg-primary text-white'
                    : msg.role === 'system'
                      ? 'bg-red-50 text-red-600'
                      : 'bg-gray-100 text-gray-800'
                }`}
              >
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content}
                </p>
                <p
                  className={`mt-1 text-right text-[10px] ${
                    msg.role === 'user'
                      ? 'text-blue-200'
                      : 'text-gray-400'
                  }`}
                >
                  {formatTime(msg.timestamp)}
                </p>
              </div>
            </div>
          ))}

          {/* Streaming assistant message */}
          {streamingContent && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-2xl bg-gray-100 px-4 py-2.5 text-gray-800">
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {streamingContent}
                  <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-primary" />
                </p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="border-t p-4">
          <div className="flex gap-2">
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="描述你的学习需求..."
              disabled={isStreaming}
              className="flex-1 rounded-lg border border-gray-200 px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-gray-400 focus:border-primary disabled:cursor-not-allowed disabled:bg-gray-50 disabled:text-gray-400"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="flex items-center gap-1.5 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-primary-dark disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {isStreaming ? (
                <>
                  <svg
                    className="h-4 w-4 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  分析中
                </>
              ) : (
                '发送'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Right panel: Radar chart + Profile card */}
      <div className="w-full space-y-4 lg:w-80 xl:w-96">
        <div className="sticky top-4 space-y-4">
          {/* Radar chart */}
          <div className="rounded-xl border bg-white p-4 shadow-sm">
            <h4 className="mb-3 text-sm font-semibold text-gray-700">
              实时画像雷达图
            </h4>
            <RadarChart data={radarData} size={260} />
          </div>

          {/* Profile card */}
          <ProfileCard />

          {/* Connection status */}
          <div className="rounded-lg border bg-white px-4 py-2.5 shadow-sm">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-400">连接状态</span>
              <span
                className={`flex items-center gap-1 ${
                  isStreaming ? 'text-green-600' : 'text-gray-400'
                }`}
              >
                <span
                  className={`inline-block h-2 w-2 rounded-full ${
                    isStreaming ? 'bg-green-500' : 'bg-gray-300'
                  }`}
                />
                {isStreaming ? '分析中' : '等待输入'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
