import * as d3 from 'd3';
import type { NodeData } from './knowledge-node';

export interface MinimapState {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Create/update the minimap in the bottom-right corner.
 * Shows a scaled-down view of the full graph with a viewport rectangle.
 */
export function renderMinimap(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  nodes: NodeData[],
  viewTransform: d3.ZoomTransform,
  graphWidth: number,
  graphHeight: number,
) {
  const mapWidth = 160;
  const mapHeight = 120;
  const padding = 8;
  const mapX = graphWidth - mapWidth - padding;
  const mapY = graphHeight - mapHeight - padding - 40; // above toolbar

  // Background
  let mapGroup = svg.select<SVGGElement>('g.minimap');
  if (mapGroup.empty()) {
    mapGroup = svg.append('g').attr('class', 'minimap');
  }

  // Clear and redraw
  mapGroup.selectAll('*').remove();

  // Map background
  mapGroup
    .append('rect')
    .attr('x', mapX)
    .attr('y', mapY)
    .attr('width', mapWidth)
    .attr('height', mapHeight)
    .attr('rx', 4)
    .attr('fill', '#FDFCF8')
    .attr('stroke', '#DCD6CD')
    .attr('stroke-width', 1)
    .attr('opacity', 0.9);

  // Compute bounds
  const xs = nodes.map((n) => n.x ?? 0);
  const ys = nodes.map((n) => n.y ?? 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  // Scale to fit
  const scaleX = (mapWidth - 16) / rangeX;
  const scaleY = (mapHeight - 16) / rangeY;
  const scale = Math.min(scaleX, scaleY, 1);

  // Node dots
  nodes.forEach((n) => {
    const nx = mapX + 8 + (n.x! - minX) * scale;
    const ny = mapY + 8 + (n.y! - minY) * scale;
    mapGroup
      .append('circle')
      .attr('cx', nx)
      .attr('cy', ny)
      .attr('r', 2)
      .attr('fill', '#7A7368')
      .attr('opacity', 0.6);
  });

  // Viewport rectangle
  const vx = mapX + 8 + (-viewTransform.x / viewTransform.k - minX) * scale;
  const vy = mapY + 8 + (-viewTransform.y / viewTransform.k - minY) * scale;
  const vw = (graphWidth / viewTransform.k) * scale;
  const vh = (graphHeight / viewTransform.k) * scale;

  mapGroup
    .append('rect')
    .attr('x', vx)
    .attr('y', vy)
    .attr('width', vw)
    .attr('height', vh)
    .attr('fill', 'none')
    .attr('stroke', '#A3B5A6')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '3 2');
}
