'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';

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
}

const COLOR_MAP: Record<KpMastery, string> = {
  mastered: '#22c55e',
  learning: '#3b82f6',
  not_started: '#9ca3af',
  locked: '#e5e7eb',
};

const LABEL_MAP: Record<KpMastery, string> = {
  mastered: '已掌握',
  learning: '学习中',
  not_started: '未开始',
  locked: '未解锁',
};

export function SkillTreeCanvas({
  courseId,
  mastery,
  recommendedKpId,
  onNodeClick,
  className,
}: SkillTreeCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

  // Fetch course graph
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

  // D3 render
  useEffect(() => {
    if (!graph || !svgRef.current || !containerRef.current) return;

    const svgEl = svgRef.current;
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 500;
    const padding = { top: 40, right: 40, bottom: 40, left: 60 };

    // Clear previous
    d3.select(svgEl).selectAll('*').remove();

    const svg = d3.select(svgEl)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);

    // Add zoom behavior
    const g = svg.append('g');
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 2])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    svg.call(zoom);

    // Build hierarchical data (tree from prerequisite edges)
    const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));
    const childMap = new Map<string, string[]>();
    const rootCandidates = new Set(graph.nodes.map((n) => n.id));

    for (const edge of graph.edges) {
      if (edge.relation_type === 'PREREQUISITE_OF') {
        if (!childMap.has(edge.source)) childMap.set(edge.source, []);
        childMap.get(edge.source)!.push(edge.target);
        rootCandidates.delete(edge.target);
      }
    }

    // Root nodes have no prerequisites
    const roots = Array.from(rootCandidates);
    if (roots.length === 0 && graph.nodes.length > 0) {
      roots.push(graph.nodes[0].id);
    }

    // Build D3 hierarchy
    const stratify = d3.stratify<{ id: string; parentId: string | null }>()
      .id((d) => d.id)
      .parentId((d) => d.parentId);

    const flat: { id: string; parentId: string | null }[] = [];
    const visited = new Set<string>();

    function dfs(nodeId: string, parentId: string | null) {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      flat.push({ id: nodeId, parentId });
      const children = childMap.get(nodeId) || [];
      for (const child of children) {
        dfs(child, nodeId);
      }
    }

    for (const root of roots) {
      dfs(root, null);
    }

    // Add any remaining nodes not reached by DFS
    for (const node of graph.nodes) {
      if (!visited.has(node.id)) {
        flat.push({ id: node.id, parentId: roots[0] || null });
      }
    }

    if (flat.length === 0) {
      svg.append('text')
        .attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle').attr('fill', '#9ca3af')
        .text('暂无知识图谱数据');
      return;
    }

    let hierarchy: d3.HierarchyNode<{ id: string; parentId: string | null }>;
    try {
      hierarchy = stratify(flat);
    } catch {
      // Fallback: all as flat roots
      const root = d3.stratify<{ id: string; parentId: string | null }>()
        .id((d) => d.id)
        .parentId(() => null)(flat);
      hierarchy = root;
    }

    const treeLayout = d3.tree<{ id: string; parentId: string | null }>()
      .size([height - padding.top - padding.bottom, width - padding.left - padding.right])
      .separation((a, b) => (a.parent === b.parent ? 1 : 1.5));

    const rootLayout = treeLayout(
      hierarchy.sort((a, b) => d3.ascending(a.data.id, b.data.id))
    );

    const linkGroup = g.append('g').attr('fill', 'none').attr('stroke', '#d1d5db').attr('stroke-width', 1.5);
    const nodeGroup = g.append('g');

    // Links
    linkGroup.selectAll('path')
      .data(rootLayout.links())
      .join('path')
      .attr('d', d3.linkHorizontal<any, any>()
        .x((d) => d.y + padding.left)
        .y((d) => d.x + padding.top))
      .attr('stroke-opacity', 0.6);

    // Nodes
    const nodes = nodeGroup.selectAll('g')
      .data(rootLayout.descendants())
      .join('g')
      .attr('transform', (d) => `translate(${d.y + padding.left}, ${d.x + padding.top})`)
      .style('cursor', 'pointer')
      .on('click', (_event, d) => {
        onNodeClick?.(d.data.id);
      });

    // Node circles
    nodes.append('circle')
      .attr('r', (d) => {
        const kp = nodeMap.get(d.data.id);
        return kp ? Math.max(6, 12 - kp.difficulty) : 8;
      })
      .attr('fill', (d) => {
        const id = d.data.id;
        if (recommendedKpId === id) return '#f97316';
        return COLOR_MAP[mastery[id] || 'not_started'] || '#9ca3af';
      })
      .attr('stroke', (d) => {
        if (recommendedKpId === d.data.id) return '#ea580c';
        return '#fff';
      })
      .attr('stroke-width', (d) => recommendedKpId === d.data.id ? 3 : 2)
      .attr('opacity', (d) => {
        const m = mastery[d.data.id];
        return m === 'locked' ? 0.4 : 1;
      });

    // Recommended pulse animation (CSS)
    if (recommendedKpId) {
      const recNode = nodes.filter((d) => d.data.id === recommendedKpId);
      recNode.select('circle')
        .attr('class', 'skill-tree-pulse')
        .style('transform-origin', 'center')
        .style('animation', 'pulse-anim 2s ease-in-out infinite');
    }

    // Labels
    nodes.append('text')
      .attr('dx', (d) => {
        const kp = nodeMap.get(d.data.id);
        return kp && kp.difficulty >= 4 ? 0 : 14;
      })
      .attr('dy', 4)
      .attr('text-anchor', (d) => {
        const kp = nodeMap.get(d.data.id);
        return kp && kp.difficulty >= 4 ? 'middle' : 'start';
      })
      .attr('font-size', '11px')
      .attr('fill', '#374151')
      .text((d) => {
        const kp = nodeMap.get(d.data.id);
        return kp ? kp.name : d.data.id;
      });

    // Tooltips
    nodes.append('title')
      .text((d) => {
        const kp = nodeMap.get(d.data.id);
        const m = mastery[d.data.id] || 'not_started';
        if (!kp) return d.data.id;
        return `${kp.name}\n难度: ${kp.difficulty}/5\n状态: ${LABEL_MAP[m]}\n${kp.description}`;
      });

    // ResizeObserver
    const resizeObserver = new ResizeObserver(() => {
      const newWidth = container.clientWidth;
      svg.attr('width', newWidth).attr('viewBox', [0, 0, newWidth, height]);
    });
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();

  }, [graph, mastery, recommendedKpId, onNodeClick]);

  if (loading) {
    return (
      <div className="flex h-[500px] items-center justify-center text-gray-400">
        <div className="flex flex-col items-center gap-2">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
          <span className="text-sm">加载知识图谱...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-[500px] items-center justify-center text-red-400">
        <p>加载失败: {error}</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`w-full ${className ?? ''}`}>
      {graph && graph.nodes.length === 0 ? (
        <div className="flex h-[500px] items-center justify-center text-gray-400">
          <p className="text-sm">暂无知识图谱数据</p>
        </div>
      ) : (
        <svg ref={svgRef} className="w-full" style={{ minHeight: '500px' }} />
      )}
    </div>
  );
}
