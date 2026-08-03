'use client';

/**
 * Persistent user ID utility.
 *
 * Generates a UUID on first access and stores it in localStorage so the same
 * browser gets a consistent identity across page reloads.  This is a short-term
 * fix until a proper auth system is wired in — at that point this module becomes
 * a thin wrapper around the authenticated user ID.
 */

const STORAGE_KEY = 'edumap-user-id';

let _cached: string | null = null;

function _generateId(): string {
  // crypto.randomUUID() is available in all modern browsers
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older environments
  return `user-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function getUserId(): string {
  if (_cached) return _cached;

  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      _cached = stored;
      return stored;
    }
    const id = _generateId();
    localStorage.setItem(STORAGE_KEY, id);
    _cached = id;
    return id;
  }

  // SSR safety — return a temporary value
  return 'anonymous';
}

/** Clear the stored user ID (useful for testing or logout). */
export function clearUserId(): void {
  _cached = null;
  if (typeof window !== 'undefined') {
    localStorage.removeItem(STORAGE_KEY);
  }
}
