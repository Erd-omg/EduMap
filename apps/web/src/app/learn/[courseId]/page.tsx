'use client';

import { useEffect, useCallback, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { SkillTreeCanvas, KpMastery } from '@/components/knowledge-graph/skill-tree-canvas';
import { ProgressSlider } from '@/components/knowledge-graph/progress-slider';
import { ProgressPanel } from '@/components/knowledge-graph/progress-panel';
import { ResourceViewer } from '@/components/resources/resource-viewer';
import { QuizViewer, type QuizQuestionData } from '@/components/quiz/quiz-viewer';
import { QuizResult } from '@/components/quiz/quiz-result';
import { DemoConsole } from '@/components/knowledge-graph/demo-console';
import { getUserId } from '@/lib/user-id';
import { useLearningPathStore } from '@/stores/learning-path-store';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

interface ResourceItem {
  id: string;
  name: string;
  type: string;
  content?: string;
  kp_id?: string;
  kp_name?: string;
  description?: string;
}

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

  // Resource viewer state
  const [selectedKpId, setSelectedKpId] = useState<string | null>(null);
  const [selectedKpName, setSelectedKpName] = useState<string>('');
  const [selectedKpDescription, setSelectedKpDescription] = useState<string>('');
  const [selectedKpDifficulty, setSelectedKpDifficulty] = useState<number>(3);
  const [selectedResources, setSelectedResources] = useState<ResourceItem[]>([]);
  const [resourcesLoading, setResourcesLoading] = useState(false);

  // Quiz state
  const [showQuiz, setShowQuiz] = useState(false);
  const [quizQuestions, setQuizQuestions] = useState<QuizQuestionData[]>([]);
  const [quizAnswers, setQuizAnswers] = useState<Record<string, string> | null>(null);
  const [quizScore, setQuizScore] = useState<number>(0);
  const [quizLoading, setQuizLoading] = useState(false);
  const [quizError, setQuizError] = useState<string | null>(null);

  // Demo mode state
  const [demoMode, setDemoMode] = useState(false);
  const [demoMastered, setDemoMastered] = useState(0);
  const [demoLearning, setDemoLearning] = useState(0);
  const [demoNotStarted, setDemoNotStarted] = useState(0);

  // Fetch learning path on mount; invalidate stale cache (>5min)
  useEffect(() => {
    if (courseId) {
      const state = useLearningPathStore.getState();
      const age = state.lastFetched ? Date.now() - state.lastFetched : Infinity;
      if (age > 300000 || !state.path?.nodes.length) {
        fetchPath(courseId, getUserId());
      }
    }
  }, [courseId, fetchPath]);

  // Build mastery map from path nodes
  const mastery: Record<string, KpMastery> = {};
  const progress: Record<string, number> = {};
  let masteredCount = 0;
  let learningCount = 0;
  let notStartedCount = 0;

  if (path?.nodes) {
    for (const node of path.nodes) {
      const nodeStatus = node.status === 'completed' && demoMode ? 'in_progress' as const : node.status;
      switch (nodeStatus) {
        case 'completed':
          mastery[node.kp_id] = 'mastered';
          progress[node.kp_id] = 100;
          masteredCount++;
          break;
        case 'in_progress':
          mastery[node.kp_id] = 'learning';
          progress[node.kp_id] = 50;
          learningCount++;
          break;
        case 'ready':
          mastery[node.kp_id] = 'not_started';
          progress[node.kp_id] = 0;
          notStartedCount++;
          break;
        case 'locked':
          mastery[node.kp_id] = 'locked';
          progress[node.kp_id] = 0;
          notStartedCount++;
          break;
      }
    }
  }

  // Override counts in demo mode
  const displayMastered = demoMode ? demoMastered : masteredCount;
  const displayLearning = demoMode ? demoLearning : learningCount;
  const displayNotStarted = demoMode ? demoNotStarted : notStartedCount;

  // Fetch resources for a selected knowledge point
  const fetchResources = useCallback(async (kpId: string, kpName: string) => {
    setResourcesLoading(true);
    setSelectedKpId(kpId);
    setSelectedKpName(kpName);
    // Reset quiz when selecting a different node
    setShowQuiz(false);
    setQuizQuestions([]);
    setQuizAnswers(null);
    try {
      // Fetch generated/system resources for this KP
      const res = await fetch(`${API_BASE}/api/v1/resources?kp_id=${kpId}&limit=20`);
      if (res.ok) {
        const data = await res.json();
        setSelectedResources(data.resources || []);
      } else {
        setSelectedResources([]);
      }
    } catch {
      setSelectedResources([]);
    } finally {
      setResourcesLoading(false);
    }
  }, []);

  // Generate quiz questions for current KP
  const handleStartQuiz = useCallback(async () => {
    if (!selectedKpId) return;
    setQuizLoading(true);
    setQuizError(null);
    setShowQuiz(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/learning-path/quiz/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kp_id: selectedKpId,
          kp_name: selectedKpName,
          kp_description: selectedKpDescription,
          difficulty: selectedKpDifficulty,
        }),
      });
      if (!res.ok) throw new Error(`Quiz generation failed: ${res.statusText}`);
      const data = await res.json();
      setQuizQuestions(data.questions || []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : '生成测验失败';
      setQuizError(msg);
      setQuizQuestions([]);
    } finally {
      setQuizLoading(false);
    }
  }, [selectedKpId, selectedKpName, selectedKpDescription, selectedKpDifficulty]);

  // Handle quiz submit
  const handleQuizSubmit = useCallback((answers: Record<string, string>, score: number) => {
    setQuizAnswers(answers);
    setQuizScore(score);
    // Record progress with quiz score
    if (selectedKpId) {
      recordProgress(selectedKpId, 'completed', score);
      // Also record to forgetting curve via learning path review endpoint
      fetch(`${API_BASE}/api/v1/learning-path/forgetting/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: getUserId(), kp_id: selectedKpId }),
      }).catch(() => {});
    }
  }, [selectedKpId, recordProgress]);

  // Handle continuing after quiz
  const handleQuizContinue = useCallback(() => {
    setShowQuiz(false);
    setQuizQuestions([]);
    setQuizAnswers(null);
  }, []);

  // Handle node click — zoom/focus graph + show resources (no progress recording)
  const handleNodeClick = useCallback((kpId: string) => {
    const node = path?.nodes.find((n) => n.kp_id === kpId);
    if (!node) return;
    // Show resources for this KP without recording progress
    fetchResources(kpId, node.name);
    setSelectedKpDescription(node.description);
    setSelectedKpDifficulty(node.difficulty);
  }, [path, fetchResources]);

  // Explicit "start learning" action — marks KP as in_progress
  const handleStartLearning = useCallback((kpId: string) => {
    const node = path?.nodes.find((n) => n.kp_id === kpId);
    if (!node) return;
    if (node.status === 'ready' || node.status === 'in_progress') {
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
          <h1 className="text-2xl font-bold text-text-primary">{courseId}</h1>
          <p className="text-sm text-text-secondary mt-1">
            {path ? `${path.total_count} 个知识点 · ${masteredCount} 已掌握` : ''}
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            href={`/generate?courseId=${courseId}`}
            className="rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-hover transition-colors"
          >
            生成资源
          </Link>
          <Link
            href="/chat"
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
          >
            对话画像
          </Link>
        </div>
      </div>

      {/* Progress bar + Demo controls */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-text-light">
            {demoMode ? '🎮 演示模式' : `${path?.total_count ?? 0} 个知识点`}
          </span>
          <button
            onClick={() => setDemoMode((v) => !v)}
            className={`rounded px-2 py-0.5 text-xs transition-colors ${
              demoMode
                ? 'bg-brand text-white'
                : 'border border-border text-text-secondary hover:bg-bg-secondary'
            }`}
          >
            {demoMode ? '退出演示' : '🎮 演示'}
          </button>
        </div>
        <ProgressSlider
          masteredCount={displayMastered}
          learningCount={displayLearning}
          notStartedCount={displayNotStarted}
          totalCount={path?.total_count ?? 0}
        />
        {demoMode && (
          <DemoConsole
            initialMastered={masteredCount}
            initialLearning={learningCount}
            initialNotStarted={notStartedCount}
            totalCount={path?.total_count ?? 0}
            onProgressUpdate={(m, l, ns) => {
              setDemoMastered(m);
              setDemoLearning(l);
              setDemoNotStarted(ns);
            }}
          />
        )}
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
                progress={progress}
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

            {/* Quiz panel */}
            {showQuiz ? (
              quizAnswers !== null ? (
                <QuizResult
                  questions={quizQuestions}
                  answers={quizAnswers}
                  score={quizScore}
                  kpName={selectedKpName}
                  onContinue={handleQuizContinue}
                  onRetry={() => {
                    setQuizAnswers(null);
                    setQuizQuestions([]);
                    handleStartQuiz();
                  }}
                />
              ) : (
                <div className="space-y-3">
                  <QuizViewer
                    questions={quizQuestions}
                    kpName={selectedKpName}
                    onSubmit={handleQuizSubmit}
                    onSkip={() => {
                      // Skip quiz — still mark as completed
                      if (selectedKpId) {
                        recordProgress(selectedKpId, 'completed');
                      }
                      setShowQuiz(false);
                    }}
                    isLoading={quizLoading}
                  />
                  {quizError && (
                    <p className="text-xs text-red-500 text-center">{quizError}</p>
                  )}
                </div>
              )
            ) : (
              /* Resource viewer */
              <ResourceViewer
                kpId={selectedKpId ?? undefined}
                kpName={selectedKpName}
                resources={selectedResources.length > 0 ? selectedResources as any : undefined}
                isLoading={resourcesLoading}
                content={selectedResources.find(r => r.description)?.description}
                onClose={selectedKpId ? () => setSelectedKpId(null) : undefined}
              />
            )}

            {/* Start learning button (when node is ready but not yet started) */}
            {selectedKpId && !showQuiz && (() => {
              const node = path?.nodes.find(n => n.kp_id === selectedKpId);
              if (node && (node.status === 'ready' || node.status === 'locked')) {
                return (
                  <button
                    onClick={() => handleStartLearning(selectedKpId)}
                    className="w-full rounded-lg border border-brand/30 bg-brand/5 px-4 py-2.5 text-sm font-medium text-brand
                               hover:bg-brand/10 transition-colors"
                  >
                    {node.status === 'locked' ? '🔒 前置知识未完成' : '🎯 开始学习这个知识点'}
                  </button>
                );
              }
              return null;
            })()}

            {/* Quiz trigger button */}
            {selectedKpId && !showQuiz && (
              <button
                onClick={handleStartQuiz}
                className="w-full rounded-lg border border-brand/30 bg-brand/5 px-4 py-2.5 text-sm font-medium text-brand
                           hover:bg-brand/10 transition-colors"
              >
                📝 完成学习并测验
              </button>
            )}

          </div>
        </div>
      )}
    </div>
  );
}
