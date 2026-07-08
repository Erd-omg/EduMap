/** Resource types for learning content management */

export type ResourceType =
  | 'explanation'
  | 'mindmap'
  | 'exercise'
  | 'reading'
  | 'visualization'
  | 'code';

export interface PersonalizationMeta {
  rationale: string;
  target_gaps: string[];
  adapted_from: string | null;
  confidence: number; // 0-1
}

export interface AuditEntry {
  auditor: string;
  result: 'pass' | 'fail' | 'degraded';
  similarity_score: number; // 0-1
  reason?: string;
  timestamp: string; // ISO 8601
}

export interface Resource {
  id: string;
  type: ResourceType;
  title: string;
  content: string;
  knowledge_point_id: string;
  difficulty: number; // 1-5
  personalization: PersonalizationMeta;
  audit_log: AuditEntry[];
  regeneration_count: number;
  confidence_score: number; // 0-1
  confidence_factors: Record<string, number>;
  created_at: string; // ISO 8601
}
