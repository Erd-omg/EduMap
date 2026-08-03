'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { ResourceLibrary } from '@/components/resources/resource-library';
import { GenerationProgress } from '@/components/resources/generation-progress';
import { getUserId } from '@/lib/user-id';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
const SESSION_STORAGE_KEY = 'edumap-active-generation';

interface CourseItem {
  id: string;
  name: string;
}

interface KpNode {
  id: string;
  name: string;
  description: string;
  difficulty: number;
}

export default function GeneratePage() {
  const [courses, setCourses] = useState<CourseItem[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<string>('');
  const [kpNodes, setKpNodes] = useState<KpNode[]>([]);
  const [selectedKpId, setSelectedKpId] = useState<string>('');
  const [generating, setGenerating] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [resourceRefreshKey, setResourceRefreshKey] = useState(0);
  const restoredKpRef = useRef<string | null>(null);
  const restoredCourseRef = useRef<string | null>(null);
  const [restored, setRestored] = useState(false);

  // Restore persisted state from sessionStorage on mount
  useEffect(() => {
    // Restore KP selection
    const savedCourseId = sessionStorage.getItem('edumap-generate-course');
    const savedKpId = sessionStorage.getItem('edumap-generate-kp');
    if (savedCourseId) {
      setSelectedCourseId(savedCourseId);
      restoredCourseRef.current = savedCourseId;
    }
    if (savedKpId) {
      setSelectedKpId(savedKpId);
      restoredKpRef.current = savedKpId;
    }

    // Restore in-progress generation session
    const savedSession = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (savedSession) {
      fetch(`${API_BASE}/api/v1/orchestrator/status/${savedSession}`)
        .then((res) => {
          if (!res.ok) throw new Error('Not found');
          return res.json();
        })
        .then((data) => {
          // Set the session regardless of status; GenerationProgress will handle display
          setCurrentSessionId(savedSession);
          if (data.overall_status === 'completed' || data.overall_status === 'failed') {
            // Session already finished — keep sessionId so progress card shows final state
          }
        })
        .catch(() => {
          sessionStorage.removeItem(SESSION_STORAGE_KEY); // Expired, not found, or unavailable
        });
    }
    setRestored(true);
  }, []);

  // Persist sessionId to sessionStorage whenever it changes
  const updateSessionId = useCallback((id: string | null) => {
    setCurrentSessionId(id);
    if (id) {
      sessionStorage.setItem(SESSION_STORAGE_KEY, id);
    } else {
      sessionStorage.removeItem(SESSION_STORAGE_KEY);
    }
  }, []);

  // Persist KP selection to sessionStorage
  useEffect(() => {
    if (selectedCourseId) sessionStorage.setItem('edumap-generate-course', selectedCourseId);
  }, [selectedCourseId]);

  useEffect(() => {
    if (selectedKpId) sessionStorage.setItem('edumap-generate-kp', selectedKpId);
  }, [selectedKpId]);

  // Fetch courses on mount; preserve restored course from sessionStorage
  useEffect(() => {
    fetch(`${API_BASE}/api/v1/kg/courses`)
      .then((r) => r.json())
      .then((data) => {
        const list = (data.courses || data || []) as CourseItem[];
        setCourses(list);
        // Only auto-select first course if no course was restored from sessionStorage
        const restored = restoredCourseRef.current;
        if (!restored || !list.some((c) => c.id === restored)) {
          if (list.length > 0) setSelectedCourseId(list[0].id);
        }
      })
      .catch(() => {});
  }, []);

  // Fetch KPs when course changes; respect restored KP from sessionStorage
  useEffect(() => {
    if (!selectedCourseId) return;
    fetch(`${API_BASE}/api/v1/kg/courses/${selectedCourseId}/graph`)
      .then((r) => r.json())
      .then((data) => {
        const nodes = (data.nodes || []) as KpNode[];
        setKpNodes(nodes);
        const restored = restoredKpRef.current;
        if (restored && nodes.some((n) => n.id === restored)) {
          // Keep the restored KP selection
        } else if (nodes.length > 0) {
          setSelectedKpId(nodes[0].id);
        }
      })
      .catch(() => setKpNodes([]));
  }, [selectedCourseId]);

  const handleGenerate = useCallback(async () => {
    if (!selectedKpId) return;
    setGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/orchestrator/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_input: `为知识点 ${selectedKpId} 生成学习资源`,
          knowledge_point_id: selectedKpId,
          user_id: getUserId(),
        }),
      });
      if (res.ok) {
        const data = await res.json();
        updateSessionId(data.session_id);
      }
    } catch (err) {
      console.error('Generate failed:', err);
    } finally {
      setGenerating(false);
    }
  }, [selectedKpId, updateSessionId]);

  const handleGenerationComplete = useCallback(() => {
    // Refresh resource list immediately (backend sync already finished before workflow_complete)
    setResourceRefreshKey((k) => k + 1);
    updateSessionId(null);
  }, [updateSessionId]);

  const handleGenerationError = useCallback((_error: string) => {
    // Error is shown inside GenerationProgress; keep sessionId for user to see.
    // Even on failure/timeout the orchestrator may have synced partial
    // resources — refresh the library so they are not silently hidden.
    setResourceRefreshKey((k) => k + 1);
    setTimeout(() => {
      updateSessionId(null);
    }, 8000);
  }, [updateSessionId]);

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">多智能体资源生成</h1>
        <p className="mt-1 text-sm text-text-secondary">
          系统通过 6 个 AI 智能体协作完成从知识提取到评估的全流程：规划课程结构 → 验证依赖关系 → 设计讲解/练习/可视化内容 → 生成代码示例 → 审核质量 → 生成测评题目
        </p>
      </div>

      {/* AI 生成资源卡片 */}
      <div className="rounded-xl border border-brand/20 bg-brand/5 p-5">
        <h3 className="text-base font-semibold text-text-primary mb-4">🤖 开始生成</h3>
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs font-medium text-text-secondary mb-1.5">选择课程</label>
            <select
              value={selectedCourseId}
              onChange={(e) => setSelectedCourseId(e.target.value)}
              className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary outline-none focus:border-brand"
            >
              {courses.length === 0 && <option value="">加载中...</option>}
              {courses.map((c) => (
                <option key={c.id} value={c.id}>{c.name} ({c.id})</option>
              ))}
            </select>
          </div>

          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs font-medium text-text-secondary mb-1.5">选择知识点</label>
            <select
              value={selectedKpId}
              onChange={(e) => setSelectedKpId(e.target.value)}
              className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary outline-none focus:border-brand"
            >
              {kpNodes.length === 0 && <option value="">请先选择课程</option>}
              {kpNodes.map((kp) => (
                <option key={kp.id} value={kp.id}>{kp.name} (难度 {kp.difficulty})</option>
              ))}
            </select>
          </div>

          <button
            onClick={handleGenerate}
            disabled={!selectedKpId || generating || currentSessionId !== null}
            className="rounded-lg bg-brand px-6 py-2 text-sm font-medium text-white hover:bg-brand-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {generating ? '⏳ 启动中...' : currentSessionId ? '生成中' : '开始生成'}
          </button>
        </div>
      </div>

      {/* Generation progress — restored from sessionStorage on refresh */}
      {restored && (
        <GenerationProgress
          sessionId={currentSessionId}
          onComplete={handleGenerationComplete}
          onError={handleGenerationError}
        />
      )}

      {/* Resource library */}
      <ResourceLibrary refreshKey={resourceRefreshKey} />
    </div>
  );
}
