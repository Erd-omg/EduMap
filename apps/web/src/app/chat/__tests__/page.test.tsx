import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ChatPage from '../page'

// Mock ChatWindow component
vi.mock('@/components/profiling/chat-window', () => ({
  ChatWindow: () => <div data-testid="chat-window">聊天窗口</div>,
}))

describe('ChatPage', () => {
  it('renders ChatWindow component', () => {
    render(<ChatPage />)
    expect(screen.getByTestId('chat-window')).toBeTruthy()
    expect(screen.getByText('聊天窗口')).toBeTruthy()
  })
})
