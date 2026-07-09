'use client';

import { create } from 'zustand';

/** Matches the backend's AgentType */
export type AgentType =
  | 'orchestrator'
  | 'planner'
  | 'guardian'
  | 'designer'
  | 'coder'
  | 'assessment'
  | 'content_auditor';

/** Progress status for a single agent step */
export interface AgentStatus {
  phase: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'degraded';
  retryCount: number;
}

/** Preview of a generated resource */
export interface ResourcePreview {
  type: string;
  title: string;
  kp_id: string;
}

/** Input to start a generation workflow */
export interface GenerationInput {
  task_input: string;
  task_type?: string;
  user_id?: string;
}

/** Full state of a generation workflow */
interface GenerationState {
  sessionId: string | null;
  currentPhase: string | null;
  overallStatus: 'idle' | 'processing' | 'completed' | 'failed' | 'degraded';
  agentStatuses: Record<string, AgentStatus>;
  generatedResources: ResourcePreview[];
  errors: string[];
  isStreaming: boolean;

  // Actions
  startGeneration: (input: GenerationInput) => Promise<string | null>;
  updatePhase: (phase: string) => void;
  updateAgentStatus: (agent: string, status: AgentStatus) => void;
  addResource: (resource: ResourcePreview) => void;
  addError: (error: string) => void;
  setCompleted: (status: string) => void;
  reset: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export const useGenerationStore = create<GenerationState>((set) => ({
  sessionId: null,
  currentPhase: null,
  overallStatus: 'idle',
  agentStatuses: {},
  generatedResources: [],
  errors: [],
  isStreaming: false,

  startGeneration: async (input: GenerationInput) => {
    set({ overallStatus: 'processing', isStreaming: true, errors: [] });

    try {
      const res = await fetch(`${API_BASE}/api/v1/orchestrator/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_input: input.task_input,
          task_type: input.task_type || 'generate',
          user_id: input.user_id || 'anonymous',
          session_id: crypto.randomUUID(),
        }),
      });

      if (!res.ok) {
        throw new Error(`Failed to start generation: ${res.statusText}`);
      }

      const data = await res.json();
      set({ sessionId: data.session_id });

      return data.session_id;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      set({ overallStatus: 'failed', errors: [msg], isStreaming: false });
      return null;
    }
  },

  updatePhase: (phase: string) => set({ currentPhase: phase }),

  updateAgentStatus: (agent: string, status: AgentStatus) =>
    set((s) => ({
      agentStatuses: { ...s.agentStatuses, [agent]: status },
    })),

  addResource: (resource: ResourcePreview) =>
    set((s) => ({
      generatedResources: [...s.generatedResources, resource],
    })),

  addError: (error: string) =>
    set((s) => ({
      errors: [...s.errors, error],
    })),

  setCompleted: (status: string) =>
    set({
      overallStatus: status as GenerationState['overallStatus'],
      isStreaming: false,
    }),

  reset: () =>
    set({
      sessionId: null,
      currentPhase: null,
      overallStatus: 'idle',
      agentStatuses: {},
      generatedResources: [],
      errors: [],
      isStreaming: false,
    }),
}));
