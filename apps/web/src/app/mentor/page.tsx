'use client';

import { MentorChat } from '@/components/mentor/mentor-chat';

export default function MentorPage() {
  return (
    <div className="container mx-auto max-w-7xl p-4">
      <h1 className="mb-1 text-2xl font-bold text-gray-900">智能学习导师</h1>
      <p className="mb-6 text-sm text-gray-500">
        基于知识图谱和已生成内容的 RAG 问答系统
      </p>
      <MentorChat />
    </div>
  );
}
