import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { UserProfile } from '@edumap/shared-types';
import { getUserId } from '@/lib/user-id';

const PROFILE_BASE =
  process.env.NEXT_PUBLIC_PROFILE_BASE || 'http://localhost:8001';
const SYNC_TIMEOUT_MS = 5000;

interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ProfileState {
  profile: UserProfile | null;
  confidenceScores: Record<string, number>;
  conversationHistory: ConversationMessage[];
  isAnalyzing: boolean;
  updateProfile: (profile: UserProfile, confidenceScores?: Record<string, number>) => void;
  updateConfidence: (scores: Record<string, number>) => void;
  addMessage: (role: 'user' | 'assistant', content: string) => void;
  setAnalyzing: (v: boolean) => void;
  reset: () => void;
}

export const useProfileStore = create<ProfileState>()(
  persist(
    (set, get) => ({
      profile: null,
      confidenceScores: {},
      conversationHistory: [],
      isAnalyzing: false,

      updateProfile: (profile, confidenceScores) =>
        set((s) => ({
          profile,
          ...(confidenceScores ? { confidenceScores } : {}),
        })),

      updateConfidence: (scores) => set({ confidenceScores: scores }),

      addMessage: (role, content) =>
        set((s) => ({
          conversationHistory: [...s.conversationHistory, { role, content }],
        })),

      setAnalyzing: (v) => set({ isAnalyzing: v }),

      /**
       * 把当前 store 中的 6 维画像 + confidence_scores + notifications_enabled
       * 统一上传到 profile-service（PUT /api/v1/profiles/{userId}）。
       *
       * 这是消除"双 Profile"问题的关键：原先前端 SSE 推过来的 6 维画像只存在
       * localStorage，后端推荐路由永远看不到；settings 页面写入的字段又会把
       * 结构化字段覆盖。这里把两者合入同一个 PUT，让 profile-service 成为
       * 唯一真理源。
       *
       * 失败仅记录 console，不阻塞 UI（同步是 best-effort）。
       */
      syncToBackend: async () => {
        const { profile, confidenceScores } = get();
        if (!profile) return;
        try {
          const profileData: Record<string, unknown> = { ...profile };
          if (typeof profileData.notifications_enabled !== 'boolean') {
            // 保留 settings 写入的通知开关；缺省视为 true
            profileData.notifications_enabled = true;
          }
          const ctrl = new AbortController();
          const timer = setTimeout(() => ctrl.abort(), SYNC_TIMEOUT_MS);
          try {
            const resp = await fetch(`${PROFILE_BASE}/api/v1/profiles/${getUserId()}`, {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                profile_data: profileData,
                confidence_scores: confidenceScores,
              }),
              signal: ctrl.signal,
            });
            // fetch 对 4xx/5xx 不抛异常，需显式检查；同步是 best-effort，仅记录
            if (!resp.ok) {
              console.warn('[profile-store] syncToBackend non-2xx:', resp.status);
            }
          } finally {
            clearTimeout(timer);
          }
        } catch (err) {
          // 离线/超时不影响 UI；仅调试可见
          console.warn('[profile-store] syncToBackend failed:', err);
        }
      },

      reset: () =>
        set({
          profile: null,
          confidenceScores: {},
          conversationHistory: [],
          isAnalyzing: false,
        }),
    }),
    {
      name: 'edumap-profile-storage',
      partialize: (state) => ({
        profile: state.profile,
        confidenceScores: state.confidenceScores,
        conversationHistory: state.conversationHistory.slice(-100), // keep last 100
      }),
    },
  ),
);
