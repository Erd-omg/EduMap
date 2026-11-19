/**
 * `api-base` resolves the backend URL for the *browser*. The bug it guards:
 * on a LAN deployment the page is opened at http://192.168.x.y:3000 but the
 * code hard-coded http://localhost:8000 — i.e. the visitor's own machine.
 * These tests set `window.location` to what a LAN visitor sees and assert the
 * backend URL follows the page's host.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type ApiBaseModule = typeof import('../api-base');

async function load(env: Record<string, string | undefined>, href: string): Promise<ApiBaseModule> {
  vi.resetModules();
  for (const [k, v] of Object.entries(env)) {
    if (v === undefined) vi.stubEnv(k, undefined);
    else vi.stubEnv(k, v);
  }
  vi.stubGlobal('window', { location: new URL(href) });
  return import('../api-base');
}

describe('API_BASE', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('follows the host the visitor used, not their localhost', async () => {
    const { API_BASE } = await load({ NEXT_PUBLIC_API_BASE: undefined, NEXT_PUBLIC_API_BASE_URL: undefined }, 'http://192.168.1.20:3000/learn/cs301');
    expect(API_BASE).toBe('http://192.168.1.20:8000');
  });

  it('keeps the dev default on localhost', async () => {
    const { API_BASE } = await load({ NEXT_PUBLIC_API_BASE: undefined, NEXT_PUBLIC_API_BASE_URL: undefined }, 'http://localhost:3000/');
    expect(API_BASE).toBe('http://localhost:8000');
  });

  it('honours an explicit non-loopback build-time URL', async () => {
    const { API_BASE } = await load({ NEXT_PUBLIC_API_BASE: 'http://api.lan.example:9000/' }, 'http://192.168.1.20:3000/');
    expect(API_BASE).toBe('http://api.lan.example:9000');
  });

  it('ignores a build-time loopback URL on a LAN visit', async () => {
    // apps/web/.env ships exactly this value; trusting it would send every
    // LAN request to the visitor's own localhost.
    const { API_BASE } = await load({ NEXT_PUBLIC_API_BASE: 'http://localhost:8000' }, 'http://192.168.1.20:3000/');
    expect(API_BASE).toBe('http://192.168.1.20:8000');
  });

  it('reads the compose spelling NEXT_PUBLIC_API_BASE_URL too', async () => {
    // docker-compose.yml set `NEXT_PUBLIC_API_BASE_URL` while the code read
    // `NEXT_PUBLIC_API_BASE`; both must count. The host deliberately differs
    // from the page's so falling back to the origin would fail this test.
    const { API_BASE } = await load({ NEXT_PUBLIC_API_BASE_URL: 'http://10.9.9.9:8000' }, 'http://10.0.0.9:3000/');
    expect(API_BASE).toBe('http://10.9.9.9:8000');
  });
});
