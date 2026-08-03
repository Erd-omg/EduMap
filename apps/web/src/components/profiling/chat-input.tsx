'use client';

import { useState, useRef, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { ToolMenu } from './tool-menu';

interface ChatInputProps {
  onSend: (text: string) => void;
  isStreaming: boolean;
  placeholder?: string;
  disabled?: boolean;
}

export function ChatInput({
  onSend,
  isStreaming,
  placeholder = '输入你的问题或指令...',
  disabled,
}: ChatInputProps) {
  const [input, setInput] = useState('');
  const [toolMenuOpen, setToolMenuOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming || disabled) return;
    setInput('');
    onSend(text);
  }, [input, isStreaming, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleToolSelect = useCallback((toolId: string) => {
    // Prepend @tool to input
    setInput((prev) => {
      const prefix = prev.trim() ? `${prev} ` : '';
      return `${prefix}@${toolId} `;
    });
    inputRef.current?.focus();
  }, []);

  return (
    <div className="relative border-t border-border p-4">
      {/* @Tool menu (floats above) */}
      <ToolMenu
        open={toolMenuOpen}
        onClose={() => setToolMenuOpen(false)}
        onSelect={handleToolSelect}
      />

      <div className="flex items-end gap-2">
        {/* Attachment button */}
        <button
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
          title="上传附件"
          disabled={disabled || isStreaming}
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3 3 0 1119.5 7.372L8.552 18.32m.009-.01l-.01.01m5.699-9.941l-7.81 7.81a1.5 1.5 0 002.112 2.13" />
          </svg>
        </button>

        {/* @Tool button */}
        <button
          onClick={() => setToolMenuOpen(!toolMenuOpen)}
          className={cn(
            'flex h-10 items-center gap-1 rounded-lg px-2.5 text-sm font-medium transition-colors',
            toolMenuOpen
              ? 'bg-brand/10 text-brand'
              : 'text-text-light hover:bg-bg-secondary hover:text-text-primary',
          )}
          title="选择工具"
          disabled={disabled || isStreaming}
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <span className="hidden sm:inline text-xs">@工具</span>
        </button>

        {/* Text input */}
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || isStreaming}
          className="flex-1 rounded-lg border border-border bg-bg-card px-4 py-2.5 text-sm text-text-primary placeholder:text-text-light outline-none transition-colors focus:border-brand focus:ring-1 focus:ring-brand/30 disabled:cursor-not-allowed disabled:opacity-60"
        />

        {/* Voice input placeholder */}
        <button
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
          title="语音输入(即将推出)"
          disabled
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
          </svg>
        </button>

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={!input.trim() || isStreaming || disabled}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-brand text-white transition-colors hover:bg-brand-hover disabled:cursor-not-allowed disabled:opacity-50"
          title="发送"
        >
          {isStreaming ? (
            <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          )}
        </button>
      </div>

      {/* Hint */}
      <p className="mt-1.5 text-xs text-text-light text-right">
        支持 @Designer 生成资源 / @Coder 运行代码
      </p>
    </div>
  );
}
