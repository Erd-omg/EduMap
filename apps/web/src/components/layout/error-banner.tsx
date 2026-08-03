'use client';

import { useState, useEffect } from 'react';

interface ErrorBannerProps {
  message?: string;
  onDismiss?: () => void;
}

export function ErrorBanner({ message: externalMsg, onDismiss: externalDismiss }: ErrorBannerProps) {
  const [internalError, setInternalError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    // Monitor online/offline status
    const handleOnline = () => { setIsOnline(true); setIsVisible(false); };
    const handleOffline = () => { setIsOnline(false); setIsVisible(true); setInternalError('🔌 网络连接已断开'); };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    setIsOnline(navigator.onLine);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (externalMsg) {
      setInternalError(null);
      setIsVisible(true);
    }
  }, [externalMsg]);

  const dismiss = () => {
    setIsVisible(false);
    setInternalError(null);
    externalDismiss?.();
  };

  if (!isVisible && !externalMsg && isOnline) return null;

  const displayMessage = externalMsg || internalError;
  if (!displayMessage) return null;

  return (
    <div className="fixed bottom-4 right-4 z-[80] max-w-sm animate-slide-up">
      <div className="rounded-xl border border-red-200 bg-red-50 shadow-lg p-4 flex items-start gap-3">
        <span className="text-sm shrink-0">{'🔌'}</span>
        <p className="flex-1 text-sm text-red-700">{displayMessage}</p>
        <button
          onClick={dismiss}
          className="shrink-0 text-red-400 hover:text-red-600 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
