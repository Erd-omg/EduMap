'use client';
import { ChatWindow } from '@/components/profiling/chat-window';

export default function ChatPage() {
  return (
    <div className="container mx-auto flex h-[calc(100vh-4rem)] flex-col p-4">
      <h1 className="mb-4 text-xl font-bold">对话画像</h1>
      <ChatWindow />
    </div>
  );
}
