"""One-off backfill: rebind system-generated resources to real KG nodes.

The orchestrator used to persist planner-invented KP ids (e.g.
``kp-linkedlist-basic``) that do not exist in the knowledge graph, so the
graph panel could never match them.  New generations bind to the node the
user selected on /generate; this script repairs existing rows by mapping
each invented id to the longest ``-``-separated prefix that IS a KG node
(``kp-linkedlist-basic`` -> ``kp-linkedlist``).

Run inside the backend container (has asyncpg + httpx + env):
    docker compose exec backend-core python scripts/db/backfill_resource_kp.py

Idempotent — rows already bound to a real node are left untouched.
"""

from __future__ import annotations

import asyncio
import os

import asyncpg
import httpx


async def _get_kg_node_ids(base_url: str) -> set[str]:
    """Fetch every KG node id across all courses."""
    node_ids: set[str] = set()
    async with httpx.AsyncClient(base_url=base_url, timeout=15) as client:
        courses_resp = await client.get("/api/v1/kg/courses")
        courses_resp.raise_for_status()
        courses = courses_resp.json().get("courses", [])
        for course in courses:
            graph_resp = await client.get(
                f"/api/v1/kg/courses/{course['id']}/graph"
            )
            if not graph_resp.is_success:
                continue
            for node in graph_resp.json().get("nodes", []):
                nid = node.get("id")
                if nid:
                    node_ids.add(nid)
    return node_ids


def _resolve_longest_prefix(kp_id: str, node_ids: set[str]) -> str | None:
    """Return the longest dash-prefix of kp_id that exists as a KG node."""
    parts = kp_id.split("-")
    for i in range(len(parts), 1, -1):
        candidate = "-".join(parts[:i])
        if candidate in node_ids:
            return candidate
    return None


async def main() -> None:
    dsn = os.environ.get(
        "DATABASE_URL", "postgresql://edumap:edumap_dev@localhost:5432/edumap"
    )
    # Some envs carry the SQLAlchemy-style "postgresql+asyncpg://" scheme,
    # which asyncpg rejects — normalize it to a plain postgresql:// DSN.
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = "postgresql://" + dsn[len("postgresql+asyncpg://"):]
    base_url = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")

    node_ids = await _get_kg_node_ids(base_url)
    print(f"Loaded {len(node_ids)} KG node ids")

    conn = await asyncpg.connect(dsn=dsn)
    try:
        rows = await conn.fetch(
            "SELECT id, kp_id FROM resources WHERE source = 'system_generated'"
        )
        updated = 0
        for row in rows:
            kp = row["kp_id"]
            if not kp or kp in node_ids:
                continue
            mapped = _resolve_longest_prefix(kp, node_ids)
            if mapped:
                await conn.execute(
                    "UPDATE resources SET kp_id = $1, updated_at = NOW() WHERE id = $2",
                    mapped,
                    row["id"],
                )
                updated += 1
                print(f"  {kp} -> {mapped}")
            else:
                print(f"  {kp} -> (no matching KG node, left as-is)")
        print(f"Done. Mapped {updated} / {len(rows)} resources.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
