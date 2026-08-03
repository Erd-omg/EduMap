/**
 * 设计 Token — JS 侧色彩/间距常量映射
 *
 * 与 globals.css 中的 @theme 保持同步。
 * 在 JS 逻辑中需要引用具体色值时使用此文件，
 * 而非硬编码色值。
 */

// ===== 色彩 Token =====
export const colors = {
  // 基础色板
  bg: {
    primary: '#F5F2EB',
    secondary: '#E8E2D9',
    card: '#FDFCF8',
  },
  border: '#DCD6CD',
  text: {
    primary: '#3D3A36',
    secondary: '#7A7368',
    light: '#B5ADA1',
  },
  // 品牌
  brand: '#A3B5A6',
  brandHover: '#8FA392',
  accent: '#C4A882',
  // 功能色
  success: '#4CAF50',
  warning: '#FFC107',
  danger: '#EF5350',
  info: '#5B8DB8',
  successBg: '#D4E8D4',
  warningBg: '#F5E6C6',
  dangerBg: '#F0D4D4',
  // 多智能体色
  agents: {
    orchestrator: '#8FA3A8',
    planner: '#7C8FA3',
    guardian: '#8FA3B5',
    designer: '#C4A89A',
    coder: '#D4A89A',
    assessment: '#A8B5A0',
    auditor: '#B5A8A0',
    mentor: '#8FA3B5',
  },
} as const;

// ===== 间距 Token =====
export const spacing = {
  sidebar: {
    expanded: 240,
    collapsed: 56,
  },
  card: {
    padding: 16,
    radius: 12,
  },
  modal: {
    radius: 16,
    width: '90%',
    height: '90%',
  },
  section: {
    gap: 16,
  },
} as const;

// ===== 字体 Token =====
export const typography = {
  fontFamily: "Inter, 'PingFang SC', 'Noto Sans SC', system-ui, sans-serif",
  sizes: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
  },
} as const;

// ===== 断点 Token =====
export const breakpoints = {
  desktop: 1024,
  tablet: 640,
} as const;
