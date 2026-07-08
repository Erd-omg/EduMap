/** Agent types for multi-agent system orchestration */

export type AgentType =
  | 'orchestrator'
  | 'planner'
  | 'guardian'
  | 'designer'
  | 'coder'
  | 'assessment'
  | 'content_auditor'
  | 'mentor';

export type AgentTaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface AgentTask {
  id: string;
  type: AgentType;
  status: AgentTaskStatus;
  input: unknown;
  output?: unknown;
  error?: string;
  created_at: string; // ISO 8601
  completed_at?: string; // ISO 8601
}

export type WorkflowPhaseStatus =
  | 'pending'
  | 'running'
  | 'passed'
  | 'failed'
  | 'degraded';

export interface WorkflowPhase {
  agent: AgentType;
  status: WorkflowPhaseStatus;
  retry_count: number;
  result?: unknown;
}

export interface GenerationWorkflow {
  phases: WorkflowPhase[];
}
