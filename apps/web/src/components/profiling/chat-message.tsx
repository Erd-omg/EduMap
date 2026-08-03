'use client';

import { useMemo, type ReactNode } from 'react';
import { cn } from '@/lib/utils';
import type { Message } from '@/stores/chat-store';
import { SourcePopover } from '@/components/provenance/source-popover';

function formatTime(timestamp: number): string {
  const d = new Date(timestamp);
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
}

/**
 * Simple markdown-to-JSX renderer (no external deps needed).
 * Handles: headings, bold, italic, inline code, code blocks, lists, links, hr.
 */
function renderMarkdown(text: string): ReactNode {
  if (!text) return null;

  // Split into blocks by double newline
  const blocks = text.split(/\n\n+/);
  const elements: ReactNode[] = [];

  for (let bi = 0; bi < blocks.length; bi++) {
    const block = blocks[bi];

    // ── Code block (```...```) ──
    const codeMatch = block.match(/^```(\w*)\n?([\s\S]*?)```$/);
    if (codeMatch) {
      const lang = codeMatch[1] || '';
      const code = codeMatch[2].replace(/\n$/, '');
      elements.push(
        <pre
          key={`code-${bi}`}
          className="my-2 overflow-x-auto rounded-lg bg-gray-100 p-3 text-xs leading-relaxed"
        >
          <code>{code}</code>
        </pre>,
      );
      continue;
    }

    // ── Horizontal rule ──
    if (/^[-*_]{3,}$/.test(block.trim())) {
      elements.push(<hr key={`hr-${bi}`} className="my-2 border-border" />);
      continue;
    }

    // ── Split block into lines and render ──
    const lines = block.split('\n');
    const lineElements: ReactNode[] = [];

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      const trimmed = line.trim();

      // ── Heading ──
      const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
      if (headingMatch) {
        const level = headingMatch[1].length;
        const content = renderInline(headingMatch[2]);
        const Tag = level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3';
        lineElements.push(
          <Tag key={`l-${li}`} className="font-semibold my-1 text-text-primary">
            {content}
          </Tag>,
        );
        continue;
      }

      // ── Unordered list ──
      const listMatch = trimmed.match(/^[-*+]\s+(.+)$/);
      if (listMatch) {
        lineElements.push(
          <li key={`l-${li}`} className="ml-4 list-disc text-sm leading-relaxed text-text-primary">
            {renderInline(listMatch[1])}
          </li>,
        );
        continue;
      }

      // ── Ordered list ──
      const olMatch = trimmed.match(/^\d+[.)]\s+(.+)$/);
      if (olMatch) {
        lineElements.push(
          <li key={`l-${li}`} className="ml-4 list-decimal text-sm leading-relaxed text-text-primary">
            {renderInline(olMatch[1])}
          </li>,
        );
        continue;
      }

      // ── Regular paragraph line ──
      if (trimmed) {
        lineElements.push(
          <span key={`l-${li}`} className="text-sm leading-relaxed text-text-primary">
            {renderInline(trimmed)}
            {li < lines.length - 1 && <br />}
          </span>,
        );
      }
    }

    elements.push(
      <div key={`block-${bi}`} className="my-1">
        {lineElements}
      </div>,
    );
  }

  return elements;
}

/** Render inline markdown (bold, italic, code, links) into a JSX fragment. */
function renderInline(text: string): ReactNode {
  if (!text) return null;

  // Split by inline code first (highest precedence, never parse inside)
  const segments = text.split(/(`[^`]+`)/g);
  const result: ReactNode[] = [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];

    // Inline code
    const codeMatch = seg.match(/^`([^`]+)`$/);
    if (codeMatch) {
      result.push(
        <code key={`code-${i}`} className="rounded bg-gray-100 px-1 text-xs font-mono text-brand">
          {codeMatch[1]}
        </code>,
      );
      continue;
    }

    // Non-code segment — scan for bold-italic, bold, italic, and links
    // using a global regex that matches anywhere in the string
    // Order matters: *** before ** before * (longest match wins)
    const pattern =
      /(\*\*\*(.+?)\*\*\*|___(.+?)___|\*\*(.+?)\*\*|__(.+?)__|\*(.+?)\*|\[([^\]]+)\]\(([^)]+)\))/g;

    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(seg)) !== null) {
      // Plain text before this match
      if (match.index > lastIndex) {
        result.push(
          <span key={`t-${i}-${lastIndex}`}>{seg.slice(lastIndex, match.index)}</span>,
        );
      }

      const m = match[0];
      if (match[2]) {
        // ***bold+italic***
        result.push(
          <strong key={`m-${i}-${match.index}`} className="font-bold">
            <em>{match[2]}</em>
          </strong>,
        );
      } else if (match[3]) {
        // ___bold+italic___
        result.push(
          <strong key={`m-${i}-${match.index}`} className="font-bold">
            <em>{match[3]}</em>
          </strong>,
        );
      } else if (match[4]) {
        // **bold**
        result.push(
          <strong key={`m-${i}-${match.index}`} className="font-bold">
            {match[4]}
          </strong>,
        );
      } else if (match[5]) {
        // __bold__
        result.push(
          <strong key={`m-${i}-${match.index}`} className="font-bold">
            {match[5]}
          </strong>,
        );
      } else if (match[6]) {
        // *italic*
        result.push(<em key={`m-${i}-${match.index}`}>{match[6]}</em>);
      } else if (match[7] && match[8]) {
        // [text](url)
        result.push(
          <a
            key={`m-${i}-${match.index}`}
            href={match[8]}
            target="_blank"
            rel="noreferrer"
            className="text-brand underline hover:text-brand-hover"
          >
            {match[7]}
          </a>,
        );
      }

      lastIndex = match.index + m.length;
    }

    // Remaining text after the last match
    if (lastIndex < seg.length) {
      result.push(<span key={`t-${i}-end`}>{seg.slice(lastIndex)}</span>);
    }
  }

  return result;
}

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

/**
 * Unified message dispatcher — renders the correct visual
 * based on message.type.
 */
export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  switch (message.type) {
    case 'user':
      return <UserMessage message={message} />;
    case 'ai_text':
      return <AiTextMessage message={message} isStreaming={isStreaming} />;
    case 'resource_card':
      return <ResourceCardMessage message={message} />;
    case 'system':
      return <SystemMessage message={message} />;
    case 'progress':
      return <ProgressMessage message={message} />;
    case 'error':
      return <ErrorMessage message={message} />;
    default:
      return null;
  }
}

/* ======== User message ======== */
function UserMessage({ message }: { message: Message }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-br-md bg-brand px-4 py-2.5 text-white">
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
        <p className="mt-1 text-right text-[10px] text-white/60">{formatTime(message.timestamp)}</p>
      </div>
    </div>
  );
}

/* ======== AI text ======== */
function AiTextMessage({
  message,
  isStreaming,
}: {
  message: Message;
  isStreaming?: boolean;
}) {
  const rendered = useMemo(() => renderMarkdown(message.content), [message.content]);

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl rounded-bl-md bg-bg-card border border-border px-4 py-2.5 text-text-primary shadow-sm">
        <div className="prose prose-sm max-w-none">
          {rendered}
          {isStreaming && (
            <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-brand rounded-sm" />
          )}
        </div>
        <p className="mt-1 text-right text-[10px] text-text-light flex items-center justify-end gap-2">
          {message.sources && message.sources.length > 0 ? (
            <SourcePopover sources={message.sources} />
          ) : message.type === 'ai_text' && !isStreaming ? (
            <span
              className="inline-flex items-center gap-1 text-amber-600"
              title="此回答基于 AI 知识生成，未引用学习资料"
            >
              <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              AI生成
            </span>
          ) : null}
          {isStreaming ? 'AI 生成中...' : formatTime(message.timestamp)}
        </p>
      </div>
    </div>
  );
}

/* ======== Resource card ======== */
function ResourceCardMessage({ message }: { message: Message }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-xl border border-border bg-bg-card shadow-sm overflow-hidden">
        {/* Card header */}
        <div className="flex items-center justify-between bg-brand/5 px-4 py-2">
          <span className="flex items-center gap-1.5 text-sm font-medium text-text-primary">
            📦 {message.content}
          </span>
          <button className="text-xs text-text-light hover:text-text-primary transition-colors">
            [📖 来源溯源]
          </button>
        </div>

        {/* Resource items */}
        {message.resources && message.resources.length > 0 && (
          <div className="divide-y divide-border">
            {message.resources.map((r, i) => (
              <div key={i} className="flex items-center gap-3 px-4 py-2.5 hover:bg-bg-secondary transition-colors cursor-pointer">
                <span className="h-2 w-2 rounded-full bg-brand shrink-0" />
                <span className="text-sm text-text-primary">{r.title}</span>
                <span className="ml-auto text-xs text-text-light">{r.type}</span>
              </div>
            ))}
          </div>
        )}

        <p className="px-4 py-2 text-[10px] text-text-light">
          {formatTime(message.timestamp)}
        </p>
      </div>
    </div>
  );
}

/* ======== System message ======== */
function SystemMessage({ message }: { message: Message }) {
  return (
    <div className="flex justify-center">
      <div className="flex items-center gap-2 rounded-full bg-bg-secondary px-4 py-1.5 text-xs text-text-secondary">
        <span className="shrink-0">🔁</span>
        <span>{message.content}</span>
      </div>
    </div>
  );
}

/* ======== Progress message ======== */
const AGENT_COLORS: Record<string, string> = {
  planner: 'bg-agent-planner',
  guardian: 'bg-agent-guardian',
  designer: 'bg-agent-designer',
  coder: 'bg-agent-coder',
  assessment: 'bg-agent-assessment',
  auditor: 'bg-agent-auditor',
  orchestrator: 'bg-agent-orchestrator',
};

function ProgressMessage({ message }: { message: Message }) {
  const agent = message.agent || 'orchestrator';
  const dotColor = AGENT_COLORS[agent] || 'bg-agent-orchestrator';

  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-2 rounded-lg bg-bg-secondary/50 px-3 py-2">
        <span className={cn('h-2 w-2 rounded-full animate-pulse shrink-0', dotColor)} />
        <span className="text-xs text-text-secondary">{message.content}</span>
      </div>
    </div>
  );
}

/* ======== Error message ======== */
function ErrorMessage({ message }: { message: Message }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-xl border border-danger/30 bg-danger-bg px-4 py-2.5">
        <p className="text-sm text-danger">{message.content}</p>
        {message.source && (
          <p className="mt-1 text-xs text-danger/70">来源: {message.source}</p>
        )}
      </div>
    </div>
  );
}
