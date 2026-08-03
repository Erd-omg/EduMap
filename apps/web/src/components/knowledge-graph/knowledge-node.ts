import * as d3 from 'd3';

/** Mastery state -> color mapping (design doc spec) */
export const MASTERY_COLORS: Record<string, string> = {
  mastered: '#4CAF50',
  learning: '#FFC107',
  not_started: '#EF5350',
  locked: '#DCD6CD',
};

export const MASTERY_BG_COLORS: Record<string, string> = {
  mastered: '#D4E8D4',
  learning: '#F5E6C6',
  not_started: '#F0D4D4',
  locked: '#F5F2EB',
};

export interface NodeData {
  id: string;
  name: string;
  description: string;
  difficulty: number;
  category?: string;
  prerequisite_ids?: string[];
  mastery: 'mastered' | 'learning' | 'not_started' | 'locked';
  /** Progress percentage 0-100 */
  progress?: number;
  isRecommended?: boolean;
  isFocused?: boolean;
  /** x/y set by layout */
  x?: number;
  y?: number;
}

export function renderNode(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  node: NodeData,
  onClick?: (id: string) => void,
) {
  const isRoundedRect = node.category !== 'chapter';
  const color = MASTERY_COLORS[node.mastery] || MASTERY_COLORS.not_started;
  const bgColor = MASTERY_BG_COLORS[node.mastery] || MASTERY_BG_COLORS.not_started;

  const group = g
    .append('g')
    .attr('transform', `translate(${node.x || 0}, ${node.y || 0})`)
    .style('cursor', 'pointer')
    .on('click', () => onClick?.(node.id));

  // Node shape
  if (isRoundedRect) {
    group
      .append('rect')
      .attr('x', -60)
      .attr('y', -24)
      .attr('width', 120)
      .attr('height', 48)
      .attr('rx', 8)
      .attr('ry', 8)
      .attr('fill', bgColor)
      .attr('stroke', color)
      .attr('stroke-width', node.isRecommended ? 3 : 1.5)
      .attr('opacity', node.mastery === 'locked' ? 0.5 : 1);
  } else {
    // Hexagon for chapters
    const points = hexagonPoints(0, 0, 28);
    group
      .append('polygon')
      .attr('points', points.map((p) => p.join(',')).join(' '))
      .attr('fill', bgColor)
      .attr('stroke', color)
      .attr('stroke-width', node.isRecommended ? 3 : 1.5)
      .attr('opacity', node.mastery === 'locked' ? 0.5 : 1);
  }

  // Label
  group
    .append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', 4)
    .attr('fill', '#3D3A36')
    .attr('font-size', '11px')
    .attr('font-weight', node.isRecommended ? '600' : '400')
    .text(node.name.length > 12 ? node.name.slice(0, 11) + '…' : node.name);

  // Progress ring (top-right)
  if (node.progress !== undefined && node.progress > 0) {
    const ringR = 8;
    const circumference = 2 * Math.PI * ringR;
    const offset = circumference * (1 - node.progress / 100);

    const ringGroup = group.append('g').attr('transform', `translate(52, -20)`);

    ringGroup
      .append('circle')
      .attr('r', ringR)
      .attr('fill', 'none')
      .attr('stroke', bgColor)
      .attr('stroke-width', 2);

    ringGroup
      .append('circle')
      .attr('r', ringR)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', circumference)
      .attr('stroke-dashoffset', offset)
      .attr('transform', 'rotate(-90)');
  }

  // Recommended glow
  if (node.isRecommended) {
    group
      .insert('rect', ':first-child')
      .attr('x', -64)
      .attr('y', -28)
      .attr('width', 128)
      .attr('height', 56)
      .attr('rx', 10)
      .attr('ry', 10)
      .attr('fill', 'none')
      .attr('stroke', '#A3B5A6')
      .attr('stroke-width', 2)
      .attr('opacity', 0.5)
      .attr('stroke-dasharray', '4 3');
  }

  return group;
}

export function updateNodePosition(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  node: NodeData,
  duration = 400,
) {
  g.transition()
    .duration(duration)
    .ease(d3.easeCubicOut)
    .attr('transform', `translate(${node.x || 0}, ${node.y || 0})`);
}

function hexagonPoints(cx: number, cy: number, r: number): [number, number][] {
  const points: [number, number][] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    points.push([cx + r * Math.cos(angle), cy + r * Math.sin(angle)]);
  }
  return points;
}
