'use client';

import { useEffect, useState } from 'react';
import { getUserId } from '@/lib/user-id';
import Link from 'next/link';
import { ForgettingCurveList } from '@/components/dashboard/forgetting-curve-list';
import { ReviewPlan } from '@/components/dashboard/review-plan';
import { MasteryDistribution } from '@/components/dashboard/mastery-distribution';
import { ActivityHeatmap } from '@/components/dashboard/activity-heatmap';
import { LearningTrend } from '@/components/dashboard/learning-trend';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

interface DashboardData {
  alerts: any[];
  alert_count: number;
  mastery_distribution: {
    urgent: number;
    warning: number;
    ok: number;
    unknown: number;
  };
  total_kps_tracked: number;
  average_recall: number;
  all_states: any[];
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboard = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/learning-path/dashboard/${getUserId()}`);
      if (!res.ok) throw new Error(`Failed to load: ${res.statusText}`);
      const json: DashboardData = await res.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载失败');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  const reviewItems = (data?.alerts || []).slice(0, 5).map((a: any) => ({
    kp_id: a.kp_id,
    kp_name: a.kp_name,
    reason: a.alert_level === 'urgent'
      ? `回忆概率仅 ${Math.round(a.recall_probability * 100)}%，建议立即复习`
      : `回忆概率 ${Math.round(a.recall_probability * 100)}%，建议巩固`,
    priority: a.alert_level === 'urgent' ? 'high' as const : a.alert_level === 'warning' ? 'medium' as const : 'low' as const,
    estimated_minutes: 10,
  }));

  // Generate mock activity data (last 7 days)
  const activityData = Array.from({ length: 7 }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (6 - i));
    return {
      date: d.toISOString().split('T')[0],
      count: Math.floor(Math.random() * 5), // Will be replaced with real data
    };
  });

  return (
    <div className="container mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">📊 学习仪表盘</h1>
          <p className="text-sm text-text-secondary mt-1">
            {data
              ? `${data.total_kps_tracked} 个知识点追踪中 · 平均回忆率 ${Math.round(data.average_recall * 100)}%`
              : '加载中...'}
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            href="/learn/cs201"
            className="rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
          >
            继续学习
          </Link>
          <button
            onClick={fetchDashboard}
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
          >
            刷新
          </button>
        </div>
      </div>

      {error ? (
        <div className="flex h-[300px] items-center justify-center text-red-400">
          <div className="text-center">
            <p className="text-sm">加载失败: {error}</p>
            <button
              onClick={fetchDashboard}
              className="mt-3 rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
            >
              重试
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* Left column — forgetting curve alerts */}
          <div className="space-y-4">
            <ForgettingCurveList
              alerts={data?.alerts || []}
              isLoading={isLoading}
              onKpClick={(kpId) => window.open(`/learn/cs201?kp=${kpId}`, '_self')}
            />
            <ReviewPlan items={reviewItems} isLoading={isLoading} />
          </div>

          {/* Middle column — activity + trend */}
          <div className="space-y-4">
            <ActivityHeatmap data={activityData} isLoading={isLoading} />
            <LearningTrend
              data={
                data?.all_states
                  ? [
                      {
                        date: new Date(Date.now() - 86400000 * 6).toISOString().split('T')[0],
                        mastered: data.mastery_distribution.ok,
                        in_progress: data.mastery_distribution.warning,
                      },
                      {
                        date: new Date().toISOString().split('T')[0],
                        mastered: data.mastery_distribution.ok,
                        in_progress: data.mastery_distribution.warning + data.mastery_distribution.urgent,
                      },
                    ]
                  : []
              }
              isLoading={isLoading}
            />
          </div>

          {/* Right column — mastery distribution */}
          <div className="space-y-4">
            <MasteryDistribution
              distribution={
                data?.mastery_distribution || { urgent: 0, warning: 0, ok: 0, unknown: 0 }
              }
              total={data?.total_kps_tracked || 0}
              isLoading={isLoading}
            />

            {/* Summary cards */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-border bg-bg-card p-4 text-center">
                <p className="text-2xl font-bold text-brand">{data?.alert_count || 0}</p>
                <p className="text-xs text-text-light mt-1">待复习</p>
              </div>
              <div className="rounded-xl border border-border bg-bg-card p-4 text-center">
                <p className="text-2xl font-bold text-green-500">
                  {data?.mastery_distribution?.ok || 0}
                </p>
                <p className="text-xs text-text-light mt-1">已掌握</p>
              </div>
              <div className="rounded-xl border border-border bg-bg-card p-4 text-center">
                <p className="text-2xl font-bold text-yellow-500">
                  {Math.round((data?.average_recall || 0) * 100)}%
                </p>
                <p className="text-xs text-text-light mt-1">平均回忆率</p>
              </div>
              <div className="rounded-xl border border-border bg-bg-card p-4 text-center">
                <p className="text-2xl font-bold text-text-primary">
                  {data?.total_kps_tracked || 0}
                </p>
                <p className="text-xs text-text-light mt-1">知识点数</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
