'use client';
import { useCallback, useEffect, useRef, useState } from 'react';

interface SSENamedEvent {
  event: string;
  data: string;
}

interface SSEOptions {
  onMessage?: (data: string) => void;
  onEvent?: (event: SSENamedEvent) => void;
  onError?: (error: Event) => void;
  autoConnect?: boolean;
}

interface SSEState {
  data: string | null;
  error: Event | null;
  isConnected: boolean;
}

/** List of event names the orchestrator SSE stream may emit. */
const ORCHESTRATOR_EVENTS = [
  'phase_change',
  'agent_start',
  'agent_complete',
  'agent_error',
  'audit_result',
  'resource_generated',
  'workflow_complete',
  'workflow_error',
  'heartbeat',
] as const;

export function useSSE(url: string, options: SSEOptions = {}) {
  const { onMessage, onEvent, onError, autoConnect = true } = options;
  const [state, setState] = useState<SSEState>({
    data: null,
    error: null,
    isConnected: false,
  });
  const eventSourceRef = useRef<EventSource | null>(null);
  const retryRef = useRef(0);
  const maxRetries = 5;

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource(url);
    eventSourceRef.current = es;

    // Named event listeners (orchestrator streaming)
    if (onEvent) {
      for (const eventName of ORCHESTRATOR_EVENTS) {
        es.addEventListener(eventName, ((event: MessageEvent) => {
          onEvent({ event: eventName, data: event.data });
        }) as EventListener);
      }
    }

    es.onopen = () => {
      retryRef.current = 0;
      setState((prev) => ({ ...prev, isConnected: true, error: null }));
    };

    es.onmessage = (event) => {
      setState((prev) => ({ ...prev, data: event.data }));
      onMessage?.(event.data);
    };

    es.onerror = (event) => {
      setState((prev) => ({ ...prev, error: event, isConnected: false }));
      onError?.(event);
      es.close();

      if (retryRef.current < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, retryRef.current), 30000);
        retryRef.current++;
        setTimeout(() => connect(), delay);
      }
    };
  }, [url, onMessage, onEvent, onError]);

  const disconnect = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setState({ data: null, error: null, isConnected: false });
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    return () => disconnect();
  }, [autoConnect, connect, disconnect]);

  return { ...state, connect, disconnect };
}
