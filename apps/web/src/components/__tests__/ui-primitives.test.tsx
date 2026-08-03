import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Button } from '../ui/button'
import { Card, CardHeader, CardContent } from '../ui/card'
import { Badge } from '../ui/badge'

describe('Button', () => {
  it('renders with default variant', () => {
    render(<Button>Click me</Button>)
    const btn = screen.getByText('Click me')
    expect(btn).toBeTruthy()
    expect(btn.tagName).toBe('BUTTON')
  })

  it('renders as a child element', () => {
    render(<Button asChild><span>Child</span></Button>)
    expect(screen.getByText('Child')).toBeTruthy()
  })

  it('applies disabled state', () => {
    render(<Button disabled>Disabled</Button>)
    const btn = screen.getByText('Disabled') as HTMLButtonElement
    expect(btn.disabled).toBe(true)
  })
})

describe('Card', () => {
  it('renders card with header and content', () => {
    render(
      <Card>
        <CardHeader>Header</CardHeader>
        <CardContent>Content</CardContent>
      </Card>,
    )
    expect(screen.getByText('Header')).toBeTruthy()
    expect(screen.getByText('Content')).toBeTruthy()
  })
})

describe('Badge', () => {
  it('renders with content', () => {
    render(<Badge>New</Badge>)
    expect(screen.getByText('New')).toBeTruthy()
  })
})
