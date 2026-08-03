import Link from 'next/link';
import { ColdStartBanner } from '@/components/cold-start/cold-start-banner';

const FEATURES = [
  {
    title: '对话画像 + 智能辅导',
    desc: 'AI 画像分析与学习辅导二合一。对话构建学习画像，随时提问获取精准解答',
    icon: '💬',
    href: '/chat',
    color: 'from-blue-500 to-blue-600',
  },
  {
    title: '资源库',
    desc: '上传学习资料，管理系统生成的讲解、练习、可视化等资源',
    icon: '📚',
    href: '/generate',
    color: 'from-purple-500 to-purple-600',
  },
  {
    title: '学习路径',
    desc: '个性化知识图谱技能树，自适应推荐下一步学习',
    icon: '🗺️',
    href: '/learn/cs201',
    color: 'from-green-500 to-green-600',
  },
  {
    title: '个人画像',
    desc: '查看六维画像雷达图和学习进度',
    icon: '👤',
    href: '/profile',
    color: 'from-rose-500 to-rose-600',
  },
];

export default function Home() {
  return (
    <div className="flex flex-col items-center gap-12 px-4 py-16">
      {/* Cold start banner */}
      <div className="w-full max-w-4xl">
        <ColdStartBanner />
      </div>

      {/* Hero */}
      <div className="text-center max-w-2xl">
        <h1 className="text-5xl font-bold tracking-tight text-text-primary">
          智图 <span className="text-brand">EduMap</span>
        </h1>
        <p className="mt-4 text-lg text-text-secondary">
          多智能体驱动的个性化学习平台
        </p>
        <p className="mt-2 text-sm text-text-light">
          基于知识图谱 · 多智能体协作 · 自适应个性化
        </p>
      </div>

      {/* Feature cards */}
      <div className="grid w-full max-w-4xl gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map((feature) => (
          <Link
            key={feature.href}
            href={feature.href}
            className="group rounded-xl border border-border bg-bg-card p-5 shadow-sm transition-all hover:shadow-md hover:-translate-y-0.5"
          >
            <div
              className={`inline-flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br text-xl text-white ${feature.color}`}
            >
              {feature.icon}
            </div>
            <h3 className="mt-3 text-base font-semibold text-text-primary">
              {feature.title}
            </h3>
            <p className="mt-1 text-sm text-text-secondary leading-relaxed">
              {feature.desc}
            </p>
          </Link>
        ))}
      </div>

      {/* Footer */}
      <p className="text-xs text-text-light">
        智图 EduMap v0.1.0 — Phase 5 MVP
      </p>
    </div>
  );
}
