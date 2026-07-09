'use client';

import { ChatWindow } from '@/components/profiling/chat-window';

export default function ChatPage() {
  return (
    <div className="container mx-auto max-w-7xl p-4">
      <h1 className="mb-1 text-2xl font-bold text-gray-900">对话画像</h1>
      <p className="mb-6 text-sm text-gray-500">
        通过对话了解你的学习风格，系统将自动生成六维学习画像
      </p>
      <ChatWindow />
    </div>
  );
}
