/** Knowledge graph types for domain structure representation */

export interface KnowledgeNode {
  id: string;
  name: string;
  description: string;
  difficulty: 1 | 2 | 3 | 4 | 5;
  prerequisites: string[];
  merged_from_ids: string[];
  canonical: boolean;
}

export type RelationType = 'prerequisite' | 'related' | 'part_of';

export interface KnowledgeEdge {
  source: string;
  target: string;
  relation_type: RelationType;
}

export interface KnowledgeGraph {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
}
