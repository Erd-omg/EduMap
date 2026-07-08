'use client';
import { useCallback, useEffect, useRef, useState } from 'react';

interface SSEOptions {
  onMessage?: (data: string) => void;
  onError?: (error: Event) => void;
  autoConnect?: boolean;
}

interface SSEState {
  data: string | null;
  error: Event | null;
  isConnected: boolean;
}

export function useSSE(url: string, options: SSEOptions = {}) {
  const { onMessage, onError, autoConnect = true } = options;
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
  }, [url, onMessage, onError]);

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
