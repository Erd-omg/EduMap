import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import Home from '../page'

// Mock Next.js link
vi.mock('next/link', () => ({
  default: 'a',
}))

// Mock ColdStartBanner
vi.mock('@/components/cold-start/cold-start-banner', () => ({
  ColdStartBanner: () => <div data-testid="cold-start-banner" />,
}))

describe('HomePage', () => {
  it('renders the hero title and subtitle', () => {
    render(<Home />)
    expect(screen.getByText('智图')).toBeTruthy()
    expect(screen.getByText('EduMap')).toBeTruthy()
    expect(screen.getByText('多智能体驱动的个性化学习平台')).toBeTruthy()
  })

  it('renders all four feature cards', () => {
    render(<Home />)
    expect(screen.getByText('对话画像 + 智能辅导')).toBeTruthy()
    expect(screen.getByText('资源库')).toBeTruthy()
    expect(screen.getByText('学习路径')).toBeTruthy()
    expect(screen.getByText('个人画像')).toBeTruthy()
  })

  it('renders feature descriptions', () => {
    render(<Home />)
    expect(screen.getByText('AI 画像分析与学习辅导二合一。对话构建学习画像，随时提问获取精准解答')).toBeTruthy()
    expect(screen.getByText('上传学习资料，管理系统生成的讲解、练习、可视化等资源')).toBeTruthy()
    expect(screen.getByText('个性化知识图谱技能树，自适应推荐下一步学习')).toBeTruthy()
    expect(screen.getByText('查看六维画像雷达图和学习进度')).toBeTruthy()
  })

  it('renders the footer version text', () => {
    render(<Home />)
    expect(screen.getByText('智图 EduMap v0.1.0 — Phase 5 MVP')).toBeTruthy()
  })

  it('renders ColdStartBanner', () => {
    render(<Home />)
    expect(screen.getByTestId('cold-start-banner')).toBeTruthy()
  })

  it('renders feature cards as links with correct hrefs', () => {
    render(<Home />)
    const chatLink = screen.getByText('对话画像 + 智能辅导').closest('a')
    expect(chatLink?.getAttribute('href')).toBe('/chat')

    const resourcesLink = screen.getByText('资源库').closest('a')
    expect(resourcesLink?.getAttribute('href')).toBe('/generate')

    const pathLink = screen.getByText('学习路径').closest('a')
    expect(pathLink?.getAttribute('href')).toBe('/learn/cs201')

    const profileLink = screen.getByText('个人画像').closest('a')
    expect(profileLink?.getAttribute('href')).toBe('/profile')
  })
})
