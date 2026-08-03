'use client';

import { useState, useEffect } from 'react';

const ONBOARDING_KEY = 'edumap-onboarding-completed';

const STEPS = [
  {
    title: '🧠 欢迎使用智图 EduMap',
    description: '你的 AI 个性化学习助手。\n\n智图能够：\n• 分析你的学习风格和知识水平\n• 生成个性化的学习路径\n• 提供智能辅导和即时反馈\n• 追踪遗忘曲线，智能安排复习',
    action: '开始了解',
  },
  {
    title: '🗺️ 学习路径',
    description: '每门课程的知识点按照前置关系组织成技能树。\n\n• 点击节点查看学习资源\n• 完成学习后进行测验\n• 系统会根据你的进度推荐下一步',
    action: '继续',
  },
  {
    title: '💬 对话辅导',
    description: '遇到不懂的概念？随时向 AI 导师提问。\n\n• 基于知识图谱和文档库的精准回答\n• 每个回答都附带来源引用\n• 也可以进行学习画像分析',
    action: '开始使用 ✨',
  },
];

export function WelcomeOverlay() {
  const [show, setShow] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const completed = localStorage.getItem(ONBOARDING_KEY);
    if (!completed) {
      // Small delay so page renders first
      const timer = setTimeout(() => setShow(true), 500);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleComplete = () => {
    localStorage.setItem(ONBOARDING_KEY, 'true');
    setShow(false);
  };

  const handleSkip = () => {
    localStorage.setItem(ONBOARDING_KEY, 'true');
    setShow(false);
  };

  if (!show) return null;

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="mx-4 max-w-md w-full rounded-2xl border border-border bg-white shadow-2xl overflow-hidden">
        {/* Content */}
        <div className="p-8 text-center">
          <div className="text-5xl mb-4">{current.title.split(' ')[0]}</div>
          <h2 className="text-xl font-bold text-text-primary mb-3">{current.title}</h2>
          <p className="text-sm text-text-secondary whitespace-pre-line leading-relaxed">
            {current.description}
          </p>
        </div>

        {/* Progress dots */}
        <div className="flex justify-center gap-1.5 pb-2">
          {STEPS.map((_, i) => (
            <div
              key={i}
              className={`h-1.5 rounded-full transition-all ${
                i === step
                  ? 'w-6 bg-brand'
                  : i < step
                    ? 'w-1.5 bg-brand/40'
                    : 'w-1.5 bg-border'
              }`}
            />
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between p-4 pt-2 border-t border-border">
          <button
            onClick={handleSkip}
            className="text-xs text-text-light hover:text-text-primary transition-colors"
          >
            跳过引导
          </button>
          <button
            onClick={() => {
              if (isLast) {
                handleComplete();
              } else {
                setStep((s) => s + 1);
              }
            }}
            className="rounded-lg bg-brand px-6 py-2 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
          >
            {isLast ? current.action : current.action}
          </button>
        </div>
      </div>
    </div>
  );
}
