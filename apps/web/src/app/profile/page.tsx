'use client';

import Link from 'next/link';
import { RadarChart } from '@/components/profiling/radar-chart';
import { ProfileCard } from '@/components/profiling/profile-card';
import { useProfileStore } from '@/stores/profile-store';
import { buildRadarData } from '@/lib/radar-utils';

export default function ProfilePage() {
  const { profile, confidenceScores } = useProfileStore();

  const radarData = profile ? buildRadarData(profile, confidenceScores) : [];

  return (
    <div className="p-4">
      {!profile ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-bg-card py-20 text-center">
          <div className="mb-4 text-5xl">📊</div>
          <h2 className="mb-2 text-lg font-semibold text-text-primary">
            尚无画像数据
          </h2>
          <p className="max-w-md text-sm text-text-secondary">
            请先前往
            <Link href="/chat" className="mx-1 text-brand underline underline-offset-2 hover:text-brand-hover">
              对话
            </Link>
            页面开始学习对话，系统将根据你的输入自动生成个性化学习画像。
          </p>
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="flex flex-col items-center rounded-xl border border-border bg-bg-card p-6 shadow-sm">
            <h3 className="mb-4 text-base font-semibold text-text-primary">
              六维能力雷达图
            </h3>
            <RadarChart data={radarData} size={400} />
          </div>
          <ProfileCard />
        </div>
      )}
    </div>
  );
}
