'use client';

import { useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { SkillTreeCanvas, KpMastery } from '@/components/knowledge-graph/skill-tree-canvas';
import { ProgressSlider } from '@/components/knowledge-graph/progress-slider';
import { ProgressPanel } from '@/components/knowledge-graph/progress-panel';
import { useLearningPathStore } from '@/stores/learning-path-store';
import { useProfileStore } from '@/stores/profile-store';

export default function CoursePage() {
  const params = useParams();
  const courseId = params.courseId as string;

  const {
    path,
    recommendation,
    isLoading,
    error,
    fetchPath,
    recordProgress,
  } = useLearningPathStore();

  const profile = useProfileStore((s) => s.profile);

  // Fetch learning path on mount
  useEffect(() => {
    if (courseId) {
      fetchPath(courseId, 'anonymous');
    }
  }, [courseId, fetchPath]);

  // Build mastery map from path nodes
  const mastery: Record<string, KpMastery> = {};
  let masteredCount = 0;
  let learningCount = 0;
  let notStartedCount = 0;

  if (path?.nodes) {
    for (const node of path.nodes) {
      switch (node.status) {
        case 'completed':
          mastery[node.kp_id] = 'mastered';
          masteredCount++;
          break;
        case 'in_progress':
          mastery[node.kp_id] = 'learning';
          learningCount++;
          break;
        case 'ready':
          mastery[node.kp_id] = 'not_started';
          notStartedCount++;
          break;
        case 'locked':
          mastery[node.kp_id] = 'locked';
          notStartedCount++;
          break;
      }
    }
  }

  // Handle node click — mark as in_progress or show details
  const handleNodeClick = useCallback((kpId: string) => {
    const node = path?.nodes.find((n) => n.kp_id === kpId);
    if (node && (node.status === 'ready' || node.status === 'in_progress')) {
      recordProgress(kpId, 'in_progress');
    }
  }, [path, recordProgress]);

  const handleStartNext = useCallback(() => {
    if (recommendation?.next_kp_id) {
      recordProgress(recommendation.next_kp_id, 'in_progress');
    }
  }, [recommendation, recordProgress]);

  // Content type label mapping
  const contentLabels: Record<string, string> = {
    explanation: '讲解',
    exercise: '练习',
    visualization: '可视化',
    code: '代码',
  };

  return (
    <div className="container mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{courseId}</h1>
          <p className="text-sm text-gray-500 mt-1">
            {path ? `${path.total_count} 个知识点 · ${masteredCount} 已掌握` : ''}
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            href={`/generate?courseId=${courseId}`}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
          >
            生成资源
          </Link>
          <Link
            href="/chat"
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 transition-colors"
          >
            对话画像
          </Link>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-6">
        <ProgressSlider
          masteredCount={masteredCount}
          learningCount={learningCount}
          notStartedCount={notStartedCount}
          totalCount={path?.total_count ?? 0}
        />
      </div>

      {/* Main content */}
      {isLoading ? (
        <div className="flex h-[400px] items-center justify-center text-gray-400">
          <div className="flex flex-col items-center gap-2">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
            <span className="text-sm">加载学习路径...</span>
          </div>
        </div>
      ) : error ? (
        <div className="flex h-[400px] items-center justify-center text-red-400">
          <p className="text-sm">加载失败: {error}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Skill tree */}
          <div className="lg:col-span-2">
            <div className="rounded-xl border bg-white p-4" style={{ minHeight: '500px' }}>
              <SkillTreeCanvas
                courseId={courseId}
                mastery={mastery}
                recommendedKpId={recommendation?.next_kp_id}
                onNodeClick={handleNodeClick}
              />
            </div>
          </div>

          {/* Right panel */}
          <div className="space-y-4">
            <div className="rounded-xl border bg-white p-4">
              <ProgressPanel
                masteredCount={masteredCount}
                learningCount={learningCount}
                notStartedCount={notStartedCount}
                totalCount={path?.total_count ?? 0}
                nextKpName={recommendation?.next_kp_name}
                nextKpReason={recommendation?.reason}
                recommendedContentType={
                  recommendation?.recommended_content_type
                    ? contentLabels[recommendation.recommended_content_type] || recommendation.recommended_content_type
                    : undefined
                }
                estimatedSessionMin={recommendation?.estimated_session_min}
                onStartNext={handleStartNext}
              />
            </div>

            {/* Resource viewer placeholder */}
            <div className="rounded-xl border bg-white p-4">
              <h3 className="mb-2 text-sm font-medium text-gray-700">学习资源</h3>
              <p className="text-center text-xs text-gray-400 py-4">
                点击技能树节点查看资源
              </p>
            </div>

            {/* Personalization banner */}
            <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
              <details open>
                <summary className="cursor-pointer text-sm font-medium text-blue-700">
                  个性化适配说明
                </summary>
                <div className="mt-2 space-y-1 text-xs text-blue-600">
                  {recommendation ? (
                    <>
                      <p>推荐学习：{recommendation.next_kp_name}</p>
                      <p>理由：{recommendation.reason || '根据您的学习进度'}</p>
                      {recommendation.estimated_session_min && (
                        <p>建议时长：约 {recommendation.estimated_session_min} 分钟</p>
                      )}
                    </>
                  ) : (
                    <p>正在分析您的学习路径...</p>
                  )}
                  {profile && (
                    <p className="mt-2 pt-2 border-t border-blue-200">
                      学习画像已就绪，系统将根据您的进度动态调整
                    </p>
                  )}
                </div>
              </details>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
