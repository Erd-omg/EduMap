'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { renderNode, updateNodePosition, type NodeData } from './knowledge-node';
import { renderEdges, highlightPath, resetEdgeHighlights, type EdgeData } from './knowledge-edge';
import { renderMinimap } from './minimap';
import { ViewControls, type GraphLayout } from './view-controls';
import { GraphTooltip, type TooltipData } from './graph-tooltip';

export type KpMastery = 'mastered' | 'learning' | 'not_started' | 'locked';

interface KnowledgeNode {
  id: string;
  name: string;
  description: string;
  difficulty: number;
  prerequisites: string[];
}

interface KnowledgeEdge {
  source: string;
  target: string;
  relation_type: string;
}

interface KnowledgeGraph {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
}

interface SkillTreeCanvasProps {
  courseId: string;
  mastery: Record<string, KpMastery>;
  recommendedKpId?: string | null;
  onNodeClick?: (kpId: string) => void;
  className?: string;
  /** Progress data (0-100 per kpId) */
  progress?: Record<string, number>;
}

export function SkillTreeCanvas({
  courseId,
  mastery,
  recommendedKpId,
  onNodeClick,
  className,
  progress,
}: SkillTreeCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [layout, setLayout] = useState<GraphLayout>('hierarchical');
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ data: TooltipData; pos: { x: number; y: number } } | null>(null);

  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

  // === Fetch graph data ===
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/v1/kg/courses/${courseId}/graph`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: KnowledgeGraph) => {
        setGraph(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [courseId, API_BASE]);

  // === DAG layout computation ===
  const computeLayout = useCallback(
    (nodes: NodeData[], edges: EdgeData[]): NodeData[] => {
      const nodeMap = new Map(nodes.map((n) => [n.id, n]));
      const adjacency = new Map<string, string[]>();
      const inDegree = new Map<string, number>();

      nodes.forEach((n) => {
        adjacency.set(n.id, []);
        inDegree.set(n.id, 0);
      });
      // Only PREREQUISITE_OF edges determine the hierarchical layout;
      // RELATED_TO edges are lateral connections drawn later but
      // must NOT block topological sort (they create false dependencies).
      edges.forEach((e) => {
        if (e.relationType !== 'PREREQUISITE_OF') return;
        adjacency.get(e.source)?.push(e.target);
        inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
      });

      if (layout === 'hierarchical') {
        // Topological sort for hierarchical layout
        const levels = new Map<string, number>();
        const queue: string[] = [];
        nodes.forEach((n) => {
          if ((inDegree.get(n.id) || 0) === 0) {
            queue.push(n.id);
            levels.set(n.id, 0);
          }
        });

        while (queue.length > 0) {
          const current = queue.shift()!;
          const currentLevel = levels.get(current)!;
          for (const child of adjacency.get(current) || []) {
            const newLevel = currentLevel + 1;
            if (!levels.has(child) || levels.get(child)! < newLevel) {
              levels.set(child, newLevel);
            }
            inDegree.set(child, (inDegree.get(child) || 1) - 1);
            if (inDegree.get(child) === 0) {
              queue.push(child);
            }
          }
        }

        // Position nodes by level
        const levelGroups = new Map<number, string[]>();
        levels.forEach((level, id) => {
          if (!levelGroups.has(level)) levelGroups.set(level, []);
          levelGroups.get(level)!.push(id);
        });

        const padding = { top: 60, bottom: 60, left: 80, right: 80 };
        const width = (containerRef.current?.clientWidth || 800) - padding.left - padding.right;
        const height = 500 - padding.top - padding.bottom;
        const maxLevel = Math.max(...Array.from(levels.values()), 0);
        const levelHeight = maxLevel > 0 ? height / maxLevel : height;

        return nodes.map((n) => {
          const level = levels.get(n.id) || 0;
          const siblings = levelGroups.get(level) || [];
          const index = siblings.indexOf(n.id);
          const siblingCount = Math.max(siblings.length, 1);
          const nodeWidth = width / siblingCount;

          return {
            ...n,
            x: padding.left + nodeWidth * (index + 0.5),
            y: padding.top + level * levelHeight + 30,
          };
        });
      } else {
        // Force-directed layout
        const width = containerRef.current?.clientWidth || 800;
        const height = 500;

        const simNodes = nodes.map((n) => ({ ...n }));
        const simLinks = edges.map((e) => ({
          source: e.source,
          target: e.target,
        }));

        const simulation = d3
          .forceSimulation(simNodes as any)
          .force('link', d3.forceLink(simLinks as any).distance(120).strength(0.3))
          .force('charge', d3.forceManyBody().strength(-400))
          .force('center', d3.forceCenter(width / 2, height / 2))
          .force('collision', d3.forceCollide(40))
          .stop();

        simulation.tick(100);

        return nodes.map((n, i) => ({
          ...n,
          x: Math.max(60, Math.min(width - 60, (simNodes[i] as any).x || width / 2)),
          y: Math.max(40, Math.min(height - 40, (simNodes[i] as any).y || height / 2)),
        }));
      }
    },
    [layout],
  );

  // === D3 render ===
  useEffect(() => {
    if (!graph || !svgRef.current || !containerRef.current) return;

    const svgEl = svgRef.current;
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 500;

    // Prepare data
    const nodeDataMap = new Map(graph.nodes.map((n) => [n.id, n]));
    const edgeData: EdgeData[] = graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      relationType: e.relation_type,
    }));

    // Filter by focus node
    let activeNodeIds: Set<string> | null = null;
    if (focusNodeId) {
      activeNodeIds = new Set<string>();
      activeNodeIds.add(focusNodeId);
      edgeData.forEach((e) => {
        if (e.target === focusNodeId) activeNodeIds!.add(e.source);
        if (e.source === focusNodeId) activeNodeIds!.add(e.target);
      });
    }

    const nodes: NodeData[] = graph.nodes
      .filter((n) => !activeNodeIds || activeNodeIds.has(n.id))
      .map((n) => ({
        id: n.id,
        name: n.name,
        description: n.description,
        difficulty: n.difficulty,
        mastery: mastery[n.id] || 'not_started',
        progress: progress?.[n.id],
        isRecommended: n.id === recommendedKpId,
        isFocused: n.id === focusNodeId,
      }));

    const filteredEdges = edgeData.filter(
      (e) =>
        (!activeNodeIds || (activeNodeIds.has(e.source) && activeNodeIds.has(e.target))) &&
        nodeDataMap.has(e.source) &&
        nodeDataMap.has(e.target),
    );

    // Compute positions
    const positioned = computeLayout(nodes, filteredEdges);
    const posMap = new Map(positioned.map((n) => [n.id, { x: n.x || 0, y: n.y || 0 }]));

    // Setup SVG
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height).attr('viewBox', [0, 0, width, height]);

    svg.append('defs').append('g').attr('class', 'edges');

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        svg.select('g.main').attr('transform', event.transform);
        renderMinimap(svg, positioned, event.transform, width, height);
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    const mainGroup = svg.append('g').attr('class', 'main');

    // Render edges
    const edgeGroup = mainGroup.append('g');
    filteredEdges.forEach((edge) => {
      const source = posMap.get(edge.source);
      const target = posMap.get(edge.target);
      if (!source || !target) return;

      edgeGroup
        .append('path')
        .attr('d', createEdgePath(source, target))
        .attr('fill', 'none')
        .attr('stroke', '#DCD6CD')
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)')
        .attr('opacity', 0.7);
    });

    // Render nodes
    const nodeGroup = mainGroup.append('g');
    const nodeElements = new Map<string, d3.Selection<SVGGElement, unknown, null, undefined>>();

    positioned.forEach((node) => {
      const g = renderNode(nodeGroup, node, (id) => {
        onNodeClick?.(id);
        setFocusNodeId((prev) => (prev === id ? null : id));
      });
      nodeElements.set(node.id, g);

      // Mouse events for tooltip
      g.on('mouseenter', (event: MouseEvent) => {
        const n = nodeDataMap.get(node.id);
        if (!n) return;
        highlightPath(svg, node.id, filteredEdges, posMap);
        setTooltip({
          data: {
            name: node.name,
            description: n.description,
            difficulty: n.difficulty,
            mastery: node.mastery,
            progress: node.progress || 0,
            prerequisites: n.prerequisites || [],
          },
          pos: { x: event.clientX, y: event.clientY },
        });
      });

      g.on('mousemove', (event: MouseEvent) => {
        setTooltip((prev) =>
          prev ? { ...prev, pos: { x: event.clientX, y: event.clientY } } : null,
        );
      });

      g.on('mouseleave', () => {
        resetEdgeHighlights(svg);
        setTooltip(null);
      });
    });

    // Arrowhead marker
    svg.select('defs')
      .append('marker')
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

    // Empty state
    if (positioned.length === 0) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#B5ADA1')
        .attr('font-size', '14px')
        .text('暂无知识图谱数据');
    }

    // Resize observer
    const resizeObserver = new ResizeObserver(() => {
      const newWidth = container.clientWidth;
      svg.attr('width', newWidth).attr('viewBox', [0, 0, newWidth, height]);
    });
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, [graph, mastery, recommendedKpId, focusNodeId, layout, onNodeClick, computeLayout, progress]);

  // Focus mode exit
  const exitFocus = useCallback(() => setFocusNodeId(null), []);

  // Zoom controls
  const zoomIn = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 1.3);
  }, []);

  const zoomOut = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 0.7);
  }, []);

  const resetZoom = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.transform, d3.zoomIdentity);
  }, []);

  if (loading) {
    return (
      <div className="flex h-[500px] items-center justify-center text-text-light">
        <div className="flex flex-col items-center gap-2">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-brand border-t-transparent" />
          <span className="text-sm">加载知识图谱...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-[500px] items-center justify-center text-danger">
        <p>加载失败: {error}</p>
      </div>
    );
  }

  const focusNodeName = focusNodeId ? graph?.nodes.find((n) => n.id === focusNodeId)?.name : undefined;

  return (
    <div ref={containerRef} className={`relative w-full ${className ?? ''}`}>
      {graph && graph.nodes.length === 0 ? (
        <div className="flex h-[500px] items-center justify-center text-text-light">
          <p className="text-sm">暂无知识图谱数据</p>
        </div>
      ) : (
        <>
          <svg ref={svgRef} className="w-full" style={{ minHeight: '500px' }} />

          {/* View controls overlay */}
          <ViewControls
            layout={layout}
            onLayoutChange={setLayout}
            zoomIn={zoomIn}
            zoomOut={zoomOut}
            resetZoom={resetZoom}
            isFocusMode={!!focusNodeId}
            onExitFocus={exitFocus}
            focusedNodeName={focusNodeName}
            relatedCount={focusNodeId ? (graph?.edges.filter(e => e.source === focusNodeId || e.target === focusNodeId).length) : undefined}
          />

          {/* Tooltip */}
          <GraphTooltip data={tooltip?.data ?? null} position={tooltip?.pos ?? null} />
        </>
      )}
    </div>
  );
}

function createEdgePath(
  source: { x: number; y: number },
  target: { x: number; y: number },
): string {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const cx1 = source.x + dx * 0.4;
  const cy1 = source.y;
  const cx2 = target.x - dx * 0.4;
  const cy2 = target.y;
  return `M${source.x},${source.y} C${cx1},${cy1} ${cx2},${cy2} ${target.x},${target.y}`;
}
