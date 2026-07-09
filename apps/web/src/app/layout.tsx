import type { Metadata } from 'next';
import { NavBar } from '@/components/layout/nav-bar';
import './globals.css';

export const metadata: Metadata = {
  title: '智图 EduMap — 个性化学习平台',
  description: '基于多智能体与知识图谱的个性化学习系统',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-surface text-gray-900 antialiased">
        <NavBar />
        <main>{children}</main>
      </body>
    </html>
  );
}
