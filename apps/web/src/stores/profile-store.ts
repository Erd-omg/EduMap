import { create } from 'zustand';
import type { UserProfile } from '@edumap/shared-types';

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

export const useProfileStore = create<ProfileState>((set) => ({
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

  reset: () =>
    set({
      profile: null,
      confidenceScores: {},
      conversationHistory: [],
      isAnalyzing: false,
    }),
}));
