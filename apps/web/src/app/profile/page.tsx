'use client';

import { RadarChart } from '@/components/profiling/radar-chart';
import { ProfileCard } from '@/components/profiling/profile-card';
import { useProfileStore } from '@/stores/profile-store';
import type { UserProfile } from '@edumap/shared-types';

function buildRadarData(profile: UserProfile, confidence: Record<string, number>) {
  const dims = [
    'learning_ability',
    'learning_motivation',
    'knowledge_coverage',
    'interaction_style',
    'focus_characteristics',
    'knowledge_base',
  ];

  return dims.map((key) => {
    const value = (profile as unknown as Record<string, unknown>)[key];
    let score = 0;

    if (typeof value === 'object' && value !== null) {
      const obj = value as Record<string, unknown>;

      if (key === 'knowledge_coverage') {
        const cov = value as {
          mastered: string[];
          learning: string[];
          not_started: string[];
        };
        const total = cov.mastered.length + cov.learning.length + cov.not_started.length;
        score = total > 0 ? cov.mastered.length / total : 0;
      } else if (key === 'knowledge_base') {
        const nums = Object.values(obj).filter(
          (v): v is number => typeof v === 'number',
        );
        score = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
      } else {
        const nums = Object.values(obj).filter(
          (v): v is number => typeof v === 'number' && v >= 0 && v <= 1,
        );
        score = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
      }
    } else if (typeof value === 'number') {
      score = value;
    }

    return {
      dimension: key,
      key,
      score: Math.max(0, Math.min(1, score)),
      confidence: confidence[key] ?? 0.5,
    };
  });
}

export default function ProfilePage() {
  const { profile, confidenceScores } = useProfileStore();

  const radarData = profile ? buildRadarData(profile, confidenceScores) : [];

  return (
    <div className="container mx-auto max-w-5xl p-4">
      <h1 className="mb-1 text-2xl font-bold text-gray-900">学习画像</h1>
      <p className="mb-6 text-sm text-gray-500">
        基于对话分析生成的六维学习能力画像
      </p>

      {!profile ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-gray-200 bg-white py-20 text-center">
          <div className="mb-4 text-5xl">📊</div>
          <h2 className="mb-2 text-lg font-semibold text-gray-700">
            尚无画像数据
          </h2>
          <p className="max-w-md text-sm text-gray-400">
            请先前往
            <a href="/chat" className="mx-1 text-primary underline underline-offset-2 hover:text-primary-dark">
              对话画像
            </a>
            页面进行对话分析，系统将根据你的学习对话自动生成个性化学习画像。
          </p>
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="flex flex-col items-center rounded-xl border bg-white p-6 shadow-sm">
            <h3 className="mb-4 text-base font-semibold text-gray-700">
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
