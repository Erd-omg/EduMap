import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PersonalizationBanner } from '../resources/personalization-banner'

describe('PersonalizationBanner', () => {
  it('renders with rationale', () => {
    render(<PersonalizationBanner rationale="基于您的视觉型偏好" />)
    expect(screen.getByText('基于您的视觉型偏好')).toBeTruthy()
  })

  it('renders next Kp name', () => {
    render(<PersonalizationBanner nextKpName="数组" />)
    expect(screen.getByText('数组')).toBeTruthy()
  })

  it('renders target gaps', () => {
    render(<PersonalizationBanner targetGaps={['链表', '树']} />)
    expect(screen.getByText('针对薄弱环节：链表、树')).toBeTruthy()
  })

  it('renders session length', () => {
    render(<PersonalizationBanner sessionLength={25} />)
    expect(screen.getByText('建议学习时长：约 25 分钟')).toBeTruthy()
  })

  it('renders confidence bar', () => {
    render(<PersonalizationBanner confidence={0.75} />)
    expect(screen.getByText('75%')).toBeTruthy()
  })

  it('renders default message when no props', () => {
    render(<PersonalizationBanner />)
    expect(screen.getByText('本材料根据您的学习画像进行了个性化调整')).toBeTruthy()
  })

  it('renders the summary text', () => {
    render(<PersonalizationBanner />)
    expect(screen.getByText('个性化适配说明')).toBeTruthy()
  })
})
