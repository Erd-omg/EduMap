/** Shared radar chart data builder — extracts 6-dimension scores from a UserProfile. */

import type { UserProfile } from '@edumap/shared-types';

export interface RadarDataPoint {
  dimension: string;
  key: string;
  score: number;
  confidence: number;
}

const DIMENSIONS = [
  'learning_ability',
  'learning_motivation',
  'knowledge_coverage',
  'interaction_style',
  'focus_characteristics',
  'knowledge_base',
] as const;

export function buildRadarData(
  profile: UserProfile,
  confidence: Record<string, number>,
): RadarDataPoint[] {
  return DIMENSIONS.map((key) => {
    const value = (profile as unknown as Record<string, unknown>)[key];
    let score = 0;

    if (typeof value === 'object' && value !== null) {
      const obj = value as Record<string, unknown>;

      if (key === 'knowledge_coverage') {
        const cov = value as { mastered: string[]; learning: string[]; not_started: string[] };
        const total = cov.mastered.length + cov.learning.length + cov.not_started.length;
        score = total > 0 ? cov.mastered.length / total : 0;
      } else if (key === 'knowledge_base') {
        const nums = Object.values(obj).filter((v): v is number => typeof v === 'number');
        score = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
      } else {
        const nums = Object.values(obj).filter(
          (v): v is number => typeof v === 'number' && v >= 0 && v <= 1,
        );
        score = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
      }
    } else if (typeof value === 'number') {
      score = value;
    }

    return {
      dimension: key,
      key,
      score: Math.max(0, Math.min(1, score)),
      confidence: confidence[key] ?? 0.5,
    };
  });
}
