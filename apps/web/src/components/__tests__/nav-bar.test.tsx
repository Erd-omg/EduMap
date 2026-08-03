import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { NavBar } from '../layout/nav-bar'

// Mock Next.js router hooks
vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
}))

vi.mock('next/link', () => ({
  default: 'a',
}))

describe('NavBar', () => {
  it('renders the logo and app name', () => {
    render(<NavBar />)
    expect(screen.getByText('智图 EduMap')).toBeTruthy()
  })

  it('renders all navigation items', () => {
    render(<NavBar />)
    expect(screen.getByText('首页')).toBeTruthy()
    expect(screen.getByText('对话')).toBeTruthy()
    expect(screen.getByText('资源库')).toBeTruthy()
    expect(screen.getByText('学习路径')).toBeTruthy()
  })

  it('marks the current page with aria-current', () => {
    render(<NavBar />)
    // On /chat, the '对话' link should have aria-current="page"
    const chatLink = screen.getByText('对话').closest('a')
    expect(chatLink?.getAttribute('aria-current')).toBe('page')
  })

  it('does not mark inactive links with aria-current', () => {
    render(<NavBar />)
    const homeLink = screen.getByText('首页').closest('a')
    expect(homeLink?.getAttribute('aria-current')).toBeFalsy()
  })
})
