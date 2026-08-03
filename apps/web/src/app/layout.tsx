import type { Metadata } from 'next';
import { Sidebar } from '@/components/layout/sidebar';
import { MobileTabBar } from '@/components/layout/mobile-tab-bar';
import { OnboardingProvider } from '@/components/onboarding/onboarding-provider';
import './globals.css';

export const metadata: Metadata = {
  title: '智图 EduMap — 个性化学习平台',
  description: '基于多智能体与知识图谱的个性化学习系统',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-bg-primary text-text-primary antialiased">
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 min-h-screen transition-all duration-300 pb-14 md:pb-0">
            {children}
          </main>
        </div>
        <MobileTabBar />
        <OnboardingProvider />
      </body>
    </html>
  );
}
