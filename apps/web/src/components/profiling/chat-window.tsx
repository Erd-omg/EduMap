'use client';
import { useState } from 'react';

export function ChatWindow() {
  const [input, setInput] = useState('');

  return (
    <div className="flex flex-1 flex-col rounded-xl border bg-white">
      <div className="flex-1 overflow-y-auto p-4">
        <p className="text-center text-gray-400">
          输入你的学习背景和目标，AI 将为你构建个性化画像
        </p>
      </div>
      <div className="border-t p-4">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="描述你的学习需求..."
            className="flex-1 rounded-lg border px-4 py-2 text-sm outline-none focus:border-primary"
          />
          <button className="rounded-lg bg-primary px-6 py-2 text-sm text-white hover:bg-primary-dark">
            发送
          </button>
        </div>
      </div>
    </div>
  );
}
