'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useMentorStore } from '@/stores/mentor-store';
import { SourcePanel } from './source-panel';

export function MentorChat() {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const {
    messages,
    sources,
    isStreaming,
    error,
    sendQuery,
    clearMessages,
  } = useMentorStore();

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus
  useEffect(() => {
    if (!isStreaming) inputRef.current?.focus();
  }, [isStreaming]);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput('');
    sendQuery(text);
  }, [input, isStreaming, sendQuery]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  // Extract sources from last assistant message
  const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
  const displaySources = isStreaming ? sources : (lastAssistantMsg?.sources || []);

  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
      {/* Left: Chat */}
      <div className="flex min-h-[500px] flex-1 flex-col rounded-xl border bg-white shadow-sm lg:min-h-[600px]">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <div
              className={`h-2.5 w-2.5 rounded-full ${isStreaming ? 'bg-green-500' : 'bg-gray-300'}`}
            />
            <span className="text-xs font-medium text-gray-500">
              {isStreaming ? 'Mentor 正在回答...' : '就绪'}
            </span>
          </div>
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="text-xs text-gray-400 hover:text-gray-600"
            >
              清除对话
            </button>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-3 text-4xl">🎓</div>
              <p className="text-sm text-gray-400">
                输入你的学习问题，Mentor 将基于资料回答
              </p>
              <p className="mt-1 text-xs text-gray-300">
                例如：&ldquo;Python 中 for 循环和 while 循环有什么区别？&rdquo;
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
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content}
                  {msg.role === 'assistant' && isStreaming && (
                    <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-primary" />
                  )}
                </p>
                {msg.sources && msg.sources.length > 0 && !isStreaming && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {msg.sources.map((s) => (
                      <span
                        key={s.id}
                        className="inline-block rounded bg-blue-50 px-1.5 py-0.5 text-[10px] text-blue-600"
                      >
                        {s.name}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {error && (
            <div className="rounded-lg bg-red-50 p-3 text-sm text-red-600">
              {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t p-4">
          <div className="flex gap-2">
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的学习问题..."
              disabled={isStreaming}
              className="flex-1 rounded-lg border border-gray-200 px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-gray-400 focus:border-primary disabled:cursor-not-allowed disabled:bg-gray-50"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-primary-dark disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {isStreaming ? '回答中...' : '发送'}
            </button>
          </div>
        </div>
      </div>

      {/* Right: Source panel */}
      <div className="w-full lg:w-72 xl:w-80">
        <div className="sticky top-4 rounded-xl border bg-white p-4 shadow-sm">
          <SourcePanel
            sources={displaySources}
            isLoading={isStreaming && sources.length === 0}
          />
        </div>
      </div>
    </div>
  );
}
