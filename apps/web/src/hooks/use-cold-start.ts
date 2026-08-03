'use client';

import { useState, useEffect, useCallback } from 'react';
import { getUserId } from '@/lib/user-id';

export interface ColdStartResult {
  confidence: number;
  verdict: 'sufficient' | 'moderate' | 'insufficient';
  factors: Record<string, unknown>;
}

const STORAGE_KEY = 'edumap-cold-start-cache';
const CACHE_TTL_MS = 2 * 60 * 1000; // 2 minutes — short enough to reflect uploads

interface CacheEntry {
  data: ColdStartResult;
  timestamp: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

/** Clear the cold start cache so next useColdStart() call refetches. */
export function clearColdStartCache(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore
  }
}

export function useColdStart(userId: string = getUserId()): {
  loading: boolean;
  result: ColdStartResult | null;
  error: string | null;
  refresh: () => void;
} {
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<ColdStartResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const fetchColdStart = useCallback(async (force = false) => {
    setLoading(true);
    setError(null);

    try {
      // Check cache first (unless forced refresh)
      if (!force) {
        try {
          const cached = localStorage.getItem(STORAGE_KEY);
          if (cached) {
            const entry: CacheEntry = JSON.parse(cached);
            if (Date.now() - entry.timestamp < CACHE_TTL_MS) {
              setResult(entry.data);
              setLoading(false);
              return;
            }
          }
        } catch {
          // Ignore cache read errors
        }
      }

      const res = await fetch(`${API_BASE}/api/v1/kg/eval/cold-start?user_id=${userId}`);
      if (!res.ok) {
        throw new Error(`Cold start check failed: ${res.statusText}`);
      }

      const data: ColdStartResult = await res.json();
      setResult(data);

      // Update cache
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ data, timestamp: Date.now() }));
      } catch {
        // Ignore cache write errors
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);
      // Default to non-blocking on error
      setResult({ confidence: 0.7, verdict: 'sufficient', factors: {} });
    } finally {
      setLoading(false);
    }
  }, [userId]);

  // Fetch on mount and when refreshKey changes
  useEffect(() => {
    fetchColdStart();
  }, [fetchColdStart, refreshKey]);

  // Re-fetch when page becomes visible again (e.g. user returns from /generate after upload)
  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        fetchColdStart();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [fetchColdStart]);

  const refresh = useCallback(() => {
    clearColdStartCache();
    setRefreshKey((k) => k + 1);
  }, []);

  return { loading, result, error, refresh };
}
