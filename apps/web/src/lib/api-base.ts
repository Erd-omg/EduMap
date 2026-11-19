/**
 * Base URL of backend-core, resolved once for the whole browser app.
 *
 * On a LAN deployment the browser must reach the backend at the *host it used
 * to open the page* (`http://192.168.x.y:8000`), not at that machine's
 * `localhost` — `localhost` in the browser means the visitor's own computer.
 * So the order is:
 *
 * 1. `NEXT_PUBLIC_API_BASE_URL` / `NEXT_PUBLIC_API_BASE` (either spelling; the
 *    compose file historically set the former while the code read the latter),
 *    when it is a real URL — set during `next build`, not at runtime.
 * 2. Otherwise, the page's own origin with the backend's port swapped in. On
 *    `localhost` this yields `http://localhost:8000` (unchanged dev default);
 *    on `http://192.168.1.20:3000` it yields `http://192.168.1.20:8000`.
 *
 * Caveat (documented in docs/deployment-lan.md): with approach 2, changing the
 * backend port means editing `NEXT_PUBLIC_API_BACKEND_PORT` (build-time).
 */

const BACKEND_PORT =
  process.env.NEXT_PUBLIC_API_BACKEND_PORT || '8000';

function _envBase(): string | null {
  const configured =
    process.env.NEXT_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE;
  if (!configured) return null;
  // A build-time value pointing at loopback is useless for LAN visitors, and
  // it is what the old `.env` shipped.
  if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(configured)) {
    return null;
  }
  return configured.replace(/\/$/, '');
}

function _originBase(): string {
  if (typeof window === 'undefined') return `http://localhost:${BACKEND_PORT}`;
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:${BACKEND_PORT}`;
}

/** backend-core base URL for the browser (no trailing slash). */
export const API_BASE: string = _envBase() ?? _originBase();

/**
 * profile-service base URL. Unlike backend-core it has no LAN fallback: the
 * service runs on :8001 and is reachable from the browser on the same host.
 */
export const PROFILE_BASE: string =
  process.env.NEXT_PUBLIC_PROFILE_BASE &&
  !/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(process.env.NEXT_PUBLIC_PROFILE_BASE)
    ? process.env.NEXT_PUBLIC_PROFILE_BASE.replace(/\/$/, '')
    : (typeof window === 'undefined' ? 'http://localhost:8001' : `${window.location.protocol}//${window.location.hostname}:8001`);
