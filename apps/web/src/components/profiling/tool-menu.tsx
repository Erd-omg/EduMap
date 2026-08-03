'use client';

import { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface ToolMenuItem {
  id: string;
  label: string;
  icon: string;
  description: string;
}

const TOOLS: ToolMenuItem[] = [
  { id: 'designer', label: '生成资源', icon: '📝', description: 'Designer - 生成讲解文档' },
  { id: 'coder', label: '运行代码', icon: '💻', description: 'Coder - 运行代码示例' },
  { id: 'mindmap', label: '思维导图', icon: '🧠', description: '生成知识思维导图' },
  { id: 'quiz', label: '生成练习题', icon: '📝', description: 'Assessment - 生成练习题' },
  { id: 'video', label: '讲解视频(实验)', icon: '🎬', description: '生成视频讲解(实验性)' },
];

interface ToolMenuProps {
  open: boolean;
  onClose: () => void;
  onSelect: (toolId: string) => void;
}

export function ToolMenu({ open, onClose, onSelect }: ToolMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={menuRef}
      className="absolute bottom-full left-0 mb-2 w-64 rounded-xl border border-border bg-bg-card shadow-lg animate-[fade-in_0.15s_ease-out]"
    >
      <div className="px-3 py-2 text-xs font-medium text-text-light border-b border-border">
        选择工具
      </div>
      <div className="p-1.5 space-y-0.5">
        {TOOLS.map((tool) => (
          <button
            key={tool.id}
            onClick={() => {
              onSelect(tool.id);
              onClose();
            }}
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-bg-secondary transition-colors"
          >
            <span className="text-lg shrink-0">{tool.icon}</span>
            <div>
              <div className="text-sm font-medium text-text-primary">
                {tool.label}
              </div>
              <div className="text-xs text-text-light">{tool.description}</div>
            </div>
          </button>
        ))}
      </div>
      <div className="border-t border-border px-3 py-2">
        <button className="w-full text-left text-xs text-text-light hover:text-text-primary transition-colors">
          + 自定义指令...
        </button>
      </div>
    </div>
  );
}
