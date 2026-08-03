import { describe, it, expect, beforeEach } from 'vitest'
import { useChatStore, type Message } from '../chat-store'

describe('ChatStore', () => {
  beforeEach(() => {
    useChatStore.setState({
      sessionMessages: {},
      activeSessionId: null,
      isStreaming: false,
      streamingContent: '',
      sources: [],
      sessions: [],
    })
  })

  it('starts with default state', () => {
    const state = useChatStore.getState()
    expect(state.activeSessionId).toBeNull()
    expect(state.isStreaming).toBe(false)
    expect(state.sessions).toEqual([])
    expect(state.sessionMessages).toEqual({})
  })

  it('createSession creates a new session and sets it active', () => {
    const id = useChatStore.getState().createSession()
    const state = useChatStore.getState()
    expect(id).toBeTruthy()
    expect(state.activeSessionId).toBe(id)
    expect(state.sessions).toHaveLength(1)
    expect(state.sessions[0].id).toBe(id)
  })

  it('addMessage adds a message to the active session', () => {
    const id = useChatStore.getState().createSession()

    const msg: Message = {
      id: 'msg-1',
      type: 'ai_text',
      role: 'assistant',
      content: 'Hello!',
      timestamp: Date.now(),
    }
    useChatStore.getState().addMessage(msg)

    const messages = useChatStore.getState().sessionMessages[id]
    expect(messages).toHaveLength(1)
    expect(messages[0].content).toBe('Hello!')
  })

  it('auto-titles session from first user message', () => {
    const id = useChatStore.getState().createSession()

    useChatStore.getState().addMessage({
      id: 'msg-1', type: 'ai_text', role: 'user',
      content: 'Can you explain arrays?', timestamp: Date.now(),
    })

    const session = useChatStore.getState().sessions.find((s) => s.id === id)
    expect(session?.title).toContain('Can you explain')
  })

  it('switchSession changes active session', () => {
    const id1 = useChatStore.getState().createSession()
    const id2 = useChatStore.getState().createSession()

    useChatStore.getState().switchSession(id1)
    expect(useChatStore.getState().activeSessionId).toBe(id1)

    useChatStore.getState().switchSession(id2)
    expect(useChatStore.getState().activeSessionId).toBe(id2)
  })

  it('deleteSession removes session and its messages', () => {
    const id = useChatStore.getState().createSession()
    useChatStore.getState().addMessage({
      id: 'msg-1', type: 'ai_text', role: 'user',
      content: 'test', timestamp: Date.now(),
    })

    useChatStore.getState().deleteSession(id)

    const state = useChatStore.getState()
    expect(state.sessions.find((s) => s.id === id)).toBeUndefined()
    expect(state.sessionMessages[id]).toBeUndefined()
  })

  it('setStreaming toggles streaming state', () => {
    useChatStore.getState().setStreaming(true)
    expect(useChatStore.getState().isStreaming).toBe(true)

    useChatStore.getState().setStreaming(false)
    expect(useChatStore.getState().isStreaming).toBe(false)
  })

  it('appendStreamingContent accumulates tokens', () => {
    useChatStore.getState().appendStreamingContent('Hello')
    useChatStore.getState().appendStreamingContent(' World')
    expect(useChatStore.getState().streamingContent).toBe('Hello World')
  })

  it('finalizeMessage moves streaming content to messages', () => {
    const id = useChatStore.getState().createSession()
    useChatStore.getState().setStreamingContent('Final content')

    useChatStore.getState().finalizeMessage()

    const messages = useChatStore.getState().sessionMessages[id]
    expect(messages).toHaveLength(1)
    expect(messages[0].content).toBe('Final content')
    expect(useChatStore.getState().streamingContent).toBe('')
    expect(useChatStore.getState().isStreaming).toBe(false)
  })

  it('getMessages returns messages for active session', () => {
    const id = useChatStore.getState().createSession()

    useChatStore.getState().addMessage({
      id: 'msg-1', type: 'ai_text', role: 'user',
      content: 'Q', timestamp: Date.now(),
    })
    useChatStore.getState().addMessage({
      id: 'msg-2', type: 'ai_text', role: 'assistant',
      content: 'A', timestamp: Date.now(),
    })

    const messages = useChatStore.getState().getMessages()
    expect(messages).toHaveLength(2)
  })

  it('clearMessages removes messages from active session', () => {
    const id = useChatStore.getState().createSession()
    useChatStore.getState().addMessage({
      id: 'msg-1', type: 'ai_text', role: 'user',
      content: 'test', timestamp: Date.now(),
    })

    useChatStore.getState().clearMessages()
    expect(useChatStore.getState().sessionMessages[id]).toEqual([])
  })
})
