'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

export interface RadarDataPoint {
  dimension: string;
  key: string;
  score: number;
  confidence: number;
}

interface RadarChartProps {
  data: RadarDataPoint[];
  size?: number;
  className?: string;
}

const DIMENSION_LABELS: Record<string, string> = {
  learning_ability: '学习能力',
  learning_motivation: '学习动机',
  knowledge_coverage: '知识覆盖',
  interaction_style: '交互偏好',
  focus_characteristics: '专注力',
  knowledge_base: '知识基础',
};

const LEVELS = 4;

export function RadarChart({ data, size = 280, className = '' }: RadarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: size, height: size });

  // Responsive: use ResizeObserver
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (w > 0) {
          setDimensions({ width: w, height: w });
        }
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // D3 rendering
  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;

    const { width, height } = dimensions;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(cx, cy) - 36;
    const angleSlice = (2 * Math.PI) / data.length;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg
      .append('g')
      .attr('transform', `translate(${cx},${cy})`);

    // Background grid: concentric polygons
    for (let level = 1; level <= LEVELS; level++) {
      const r = (radius / LEVELS) * level;
      const points: [number, number][] = [];
      for (let i = 0; i < data.length; i++) {
        const angle = angleSlice * i - Math.PI / 2;
        points.push([r * Math.cos(angle), r * Math.sin(angle)]);
      }
      g.append('polygon')
        .attr('points', points.map((p) => p.join(',')).join(' '))
        .attr('fill', 'none')
        .attr('stroke', '#e2e8f0')
        .attr('stroke-width', 1);

      // Level label (percentage)
      g.append('text')
        .attr('x', 4)
        .attr('y', -r)
        .attr('font-size', '9px')
        .attr('fill', '#94a3b8')
        .text(`${Math.round((level / LEVELS) * 100)}%`);
    }

    // Axes
    const axes = g
      .selectAll('.axis')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'axis');

    axes
      .append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', (_, i) => radius * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y2', (_, i) => radius * Math.sin(angleSlice * i - Math.PI / 2))
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1);

    // Labels with offset based on quadrant
    axes
      .append('text')
      .attr('x', (_, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        const base = (radius + 20) * Math.cos(angle);
        // Push labels outward that are near the horizontal center
        if (Math.abs(Math.cos(angle)) < 0.1) return base;
        return base * 1.1;
      })
      .attr('y', (_, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        const base = (radius + 20) * Math.sin(angle);
        return base + (Math.sin(angle) >= 0 ? 14 : -6);
      })
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '11px')
      .attr('font-weight', '500')
      .attr('fill', '#334155')
      .text((d) => DIMENSION_LABELS[d.key] || d.dimension);

    // Data polygon points
    const dataPoints = data.map((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const r = radius * Math.max(0, Math.min(1, d.score));
      return [r * Math.cos(angle), r * Math.sin(angle)] as [number, number];
    });

    const avgConfidence = d3.mean(data, (d) => d.confidence) || 0.5;

    // Fill polygon with animation
    const polygon = g
      .append('polygon')
      .attr('points', dataPoints.map((p) => p.join(',')).join(' '))
      .attr('fill', '#A3B5A6')
      .attr('fill-opacity', Math.max(0.1, Math.min(0.4, 0.15 + 0.25 * avgConfidence)))
      .attr('stroke', '#8FA392')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', Math.max(0.3, Math.min(0.9, 0.5 + 0.5 * avgConfidence)))
      .attr('stroke-linejoin', 'round');

    // Animate polygon growing from center
    const initialPoints = dataPoints.map(() => [0, 0] as [number, number]);
    polygon
      .attr('points', initialPoints.map((p) => p.join(',')).join(' '))
      .transition()
      .duration(600)
      .ease(d3.easeCubicOut)
      .attr('points', dataPoints.map((p) => p.join(',')).join(' '));

    // Data point circles
    axes
      .append('circle')
      .attr('cx', (d, i) => dataPoints[i][0])
      .attr('cy', (d, i) => dataPoints[i][1])
      .attr('r', 4)
      .attr('fill', '#A3B5A6')
      .attr('stroke', '#FDFCF8')
      .attr('stroke-width', 2)
      .style('opacity', 0)
      .transition()
      .delay(400)
      .duration(300)
      .style('opacity', 1);

    // Tooltip on hover
    axes
      .append('title')
      .text(
        (d) =>
          `${DIMENSION_LABELS[d.key] || d.dimension}: ${(d.score * 100).toFixed(0)}% (置信度: ${(d.confidence * 100).toFixed(0)}%)`,
      );
  }, [data, dimensions]);

  if (data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center rounded-xl border border-dashed border-border bg-bg-secondary/50 ${className}`}
        style={{ width: dimensions.width, height: dimensions.height }}
      >
        <p className="text-sm text-text-light">等待对话分析...</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={className}>
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="overflow-visible"
        role="img"
        aria-label={`六维能力雷达图，${data.length} 个维度`}
      />
    </div>
  );
}
