import * as d3 from 'd3';

export interface EdgeData {
  source: string;
  target: string;
  relationType: string;
}

/**
 * Render DAG edges with arrowheads.
 * Uses quadratic bezier curves for clean connections.
 */
export function renderEdges(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  edges: EdgeData[],
  nodePositions: Map<string, { x: number; y: number }>,
) {
  // Define arrowhead marker
  svg
    .select('defs')
    .selectAll('marker#arrowhead')
    .data([1])
    .join('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 16)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#DCD6CD');

  const linkGroup = svg.select('g.edges').selectAll('path').data(edges);

  // Enter
  linkGroup
    .enter()
    .append('path')
    .attr('fill', 'none')
    .attr('stroke', '#DCD6CD')
    .attr('stroke-width', 1.5)
    .attr('marker-end', 'url(#arrowhead)')
    .attr('stroke-dasharray', 1000)
    .attr('stroke-dashoffset', 1000)
    .transition()
    .duration(500)
    .delay((_, i) => i * 30)
    .attr('stroke-dashoffset', 0);

  // Update
  linkGroup
    .attr('d', (edge) => {
      const source = nodePositions.get(edge.source);
      const target = nodePositions.get(edge.target);
      if (!source || !target) return '';
      return createBezierPath(source, target);
    });

  // Exit
  linkGroup.exit().remove();
}

/**
 * Highlight the path between two nodes (prerequisite chain).
 */
export function highlightPath(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  nodeId: string,
  edges: EdgeData[],
  nodePositions: Map<string, { x: number; y: number }>,
) {
  // Find all edges where nodeId is the target (prerequisites of the node)
  // and where nodeId is the source (dependents of the node)
  const relevant = edges.filter(
    (e) => e.target === nodeId || e.source === nodeId,
  );

  // Reset all edges
  svg.selectAll('g.edges path').attr('opacity', 0.2);

  // Highlight relevant
  svg
    .selectAll('g.edges path')
    .filter((d: unknown) => {
      const edge = d as EdgeData;
      return edge.source === nodeId || edge.target === nodeId;
    })
    .attr('opacity', 1)
    .attr('stroke', '#8FA392')
    .attr('stroke-width', 2.5);
}

/**
 * Reset all edges to default state.
 */
export function resetEdgeHighlights(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
) {
  svg
    .selectAll('g.edges path')
    .attr('opacity', 1)
    .attr('stroke', '#DCD6CD')
    .attr('stroke-width', 1.5);
}

function createBezierPath(
  source: { x: number; y: number },
  target: { x: number; y: number },
): string {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const cx1 = source.x + dx * 0.5;
  const cy1 = source.y;
  const cx2 = target.x - dx * 0.5;
  const cy2 = target.y;
  return `M${source.x},${source.y} C${cx1},${cy1} ${cx2},${cy2} ${target.x},${target.y}`;
}
