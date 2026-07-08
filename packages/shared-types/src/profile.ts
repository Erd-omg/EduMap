/** User profile types representing the multi-dimensional learner model */

export interface LearningAbility {
  understanding_speed: number; // 0-1
  problem_solving: number;    // 0-1
  critical_thinking: number;  // 0-1
  memory_retention: number;   // 0-1
  analytical_ability: number; // 0-1
}

export interface LearningMotivation {
  intrinsic_interest: number;    // 0-1
  goal_oriented: number;         // 0-1
  extrinsic_motivation: number;  // 0-1
}

export interface KnowledgeCoverage {
  mastered: string[];
  learning: string[];
  not_started: string[];
}

export interface InteractionStyle {
  visual: number;     // 0-1
  textual: number;    // 0-1
  interactive: number; // 0-1
  auditory: number;   // 0-1
}

export interface FocusCharacteristics {
  avg_focus_duration_min: number;
  distraction_frequency: number;
  recommended_session_length: number;
}

export interface UserProfile {
  knowledge_base: Record<string, number>; // 学科知识水平 0-1
  learning_ability: LearningAbility;
  learning_motivation: LearningMotivation;
  knowledge_coverage: KnowledgeCoverage;
  interaction_style: InteractionStyle;
  focus_characteristics: FocusCharacteristics;
}
