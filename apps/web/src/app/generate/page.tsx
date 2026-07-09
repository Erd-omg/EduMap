'use client';

import { GenerationWorkflow } from '@/components/generation/generation-workflow';

export default function GeneratePage() {
  return (
    <div className="container mx-auto max-w-4xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">多智能体资源生成</h1>
        <p className="mt-1 text-sm text-gray-500">
          AI 智能体将从知识提取到内容生成再到测评，全流程自动化
        </p>
      </div>

      <GenerationWorkflow />
    </div>
  );
}
