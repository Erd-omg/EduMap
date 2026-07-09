'use client';

import { create } from 'zustand';

export type KpStatus = 'ready' | 'in_progress' | 'completed' | 'locked';
export type KpMastery = 'mastered' | 'learning' | 'not_started' | 'locked';

export interface PathNode {
  kp_id: string;
  name: string;
  description: string;
  difficulty: number;
  status: KpStatus;
  prerequisites: string[];
  prerequisites_met: boolean;
  recommended_content_types: string[];
}

export interface PersonalizedPath {
  course_id: string;
  user_id: string;
  nodes: PathNode[];
  total_count: number;
  mastered_count: number;
  completed_count: number;
  progress_percent: number;
  created_at: string;
}

export interface PathRecommendation {
  next_kp_id: string;
  next_kp_name: string;
  reason: string;
  recommended_content_type: string;
  estimated_session_min: number;
}

interface LearningPathState {
  courseId: string | null;
  path: PersonalizedPath | null;
  recommendation: PathRecommendation | null;
  isLoading: boolean;
  error: string | null;

  fetchPath: (courseId: string, userId?: string) => Promise<void>;
  recordProgress: (kpId: string, status: string, quizScore?: number) => Promise<void>;
  reset: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export const useLearningPathStore = create<LearningPathState>((set, get) => ({
  courseId: null,
  path: null,
  recommendation: null,
  isLoading: false,
  error: null,

  fetchPath: async (courseId: string, userId: string = 'anonymous') => {
    set({ isLoading: true, error: null, courseId });

    try {
      const [pathRes, nextRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/learning-path/${courseId}?user_id=${userId}`),
        fetch(`${API_BASE}/api/v1/learning-path/${courseId}/next?user_id=${userId}`)
          .catch(() => null), // 404 is ok — all completed
      ]);

      if (!pathRes.ok) {
        throw new Error(`Failed to fetch path: ${pathRes.statusText}`);
      }

      const path: PersonalizedPath = await pathRes.json();

      let recommendation: PathRecommendation | null = null;
      if (nextRes && nextRes.ok) {
        recommendation = await nextRes.json();
      }

      set({ path, recommendation, isLoading: false });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      set({ error: msg, isLoading: false });
    }
  },

  recordProgress: async (kpId: string, status: string, quizScore?: number) => {
    const { courseId, path } = get();
    if (!courseId || !path) return;

    try {
      const res = await fetch(`${API_BASE}/api/v1/learning-path/progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: path.user_id,
          course_id: courseId,
          kp_id: kpId,
          status,
          quiz_score: quizScore,
        }),
      });

      if (!res.ok) throw new Error(`Progress update failed: ${res.statusText}`);

      const data = await res.json();
      set({
        path: data.updated_path,
        recommendation: data.next_recommendation,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      set({ error: msg });
    }
  },

  reset: () => set({
    courseId: null,
    path: null,
    recommendation: null,
    isLoading: false,
    error: null,
  }),
}));
