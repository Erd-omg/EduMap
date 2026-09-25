"""Checkpoint retention for the orchestrator's durable graph state (I-6).

Why this exists
---------------
LangGraph checkpoints every super-step and never removes them on its own —
its own troubleshooting docs list "**Checkpoints growing unboundedly**" as a
known production failure mode. A generation run visits ~9 nodes, so a
long-lived deployment accumulates rows linearly with traffic and the table
becomes the largest thing in the database.

Retention policy
----------------
**Whole-thread deletion, driven by thread age.**

Why not "prune within a thread" (keep only the latest checkpoint)? That was
the original design, and it does not work here: ``AsyncPostgresSaver`` does
**not** implement ``aprune`` — it inherits the base class's
``raise NotImplementedError``, and the same is true of ``adelete_for_runs``.
Only ``adelete_thread`` (whole-thread) is genuinely implemented. Mocking the
saver hid this completely; it surfaced only when run against a real database.

So the unit of retention is the **thread**, and the trigger is age:

* a session's checkpoints serve progress reporting and crash recovery while the
  session is live, and are dead weight once it is old;
* deleting the whole thread is one call to a supported API and cannot corrupt
  the graph's history, whereas synthesising per-checkpoint ``DELETE``s would
  have to reproduce the saver's internal constraints by hand.

The consequence to be aware of: within the retention window a long-running
thread keeps every checkpoint. That is the correct trade — the alternative
options are unsupported or unsafe.

Note on what this does *not* solve: without an internal prune, a single very
long-lived session still accumulates. Sessions are reaped by age, so the bound
is "checkpoints per session × sessions within the TTL window", which is what
the TTL is for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Threads whose newest checkpoint is older than this are removed wholesale.
# Seven days covers "debug last week's failed run" without keeping months of
# dead rows.
DEFAULT_THREAD_TTL_DAYS = 7


@dataclass
class PruneReport:
    """Outcome of one retention pass — reported so the job is observable."""

    threads_examined: int = 0
    threads_deleted: int = 0
    threads_kept: int = 0
    errors: int = 0

    def to_dict(self) -> dict:
        return {
            "threads_examined": self.threads_examined,
            "threads_deleted": self.threads_deleted,
            "threads_kept": self.threads_kept,
            "errors": self.errors,
        }


async def list_thread_ids(saver: Any, *, limit: int = 1000) -> list[str]:
    """Distinct thread ids currently holding checkpoints.

    Queried from the saver's own pool rather than through a new connection, so
    retention cannot drift from what the saver actually wrote.
    """
    pool = getattr(saver, "conn", None) or getattr(saver, "_pool", None)
    if pool is None:
        return []
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id LIMIT %s",
                (limit,),
            )
            rows = await cur.fetchall()
    return [r[0] for r in rows]


async def delete_thread(saver: Any, thread_id: str) -> bool:
    """Delete all checkpoints for one thread. Returns True on success."""
    try:
        await saver.adelete_thread(thread_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete checkpoints for thread %s: %s", thread_id, exc)
        return False


async def thread_last_activity(saver: Any, *, limit: int = 1000) -> dict[str, datetime]:
    """Newest checkpoint timestamp per thread, from the ``checkpoints`` table.

    The timestamp is **not** in the checkpoint metadata — a real
    ``CheckpointTuple.metadata`` carries only ``{step, source, parents}``. It
    lives in the ``checkpoint`` JSONB payload under ``ts``. Reading it from
    there is what makes retention actually work; an earlier version looked for
    ``metadata["created_at"]``, always got ``None``, and therefore never
    identified a single stale thread while appearing to succeed.

    Querying every thread's latest ``ts`` in one statement also avoids an
    ``aget_tuple`` round-trip per thread.

    Returns a ``{thread_id: newest_ts}`` mapping; threads whose timestamp is
    missing or unparseable are omitted (the caller keeps them).
    """
    pool = getattr(saver, "conn", None) or getattr(saver, "_pool", None)
    if pool is None:
        return {}
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Cast to timestamptz and MAX *that*, rather than MAX over text.
            #
            # Ordering the text form happens to work while every timestamp has
            # the same fractional precision, which is what LangGraph writes
            # today (always 6 digits) — so this is not fixing an observed
            # mis-ordering. It removes the *dependency* on that formatting: a
            # value with fewer fractional digits ("…:21.5") does not compare
            # correctly against "…:21.45" as text, and a future writer (or a
            # different ts source) could produce it. Casting makes the
            # comparison chronological by construction.
            await cur.execute(
                """
                SELECT thread_id, MAX((checkpoint ->> 'ts')::timestamptz) AS newest
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY thread_id
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()

    out: dict[str, datetime] = {}
    for thread_id, newest in rows:
        parsed = _as_utc(newest)
        if parsed is not None:
            out[thread_id] = parsed
    return out


def _as_utc(value: Any) -> datetime | None:
    """Normalise a timestamp value to an aware UTC datetime, or None.

    psycopg returns a ``datetime`` for a ``timestamptz`` column, but tests and
    alternate drivers may hand back a string, so both are accepted. A naive
    datetime is assumed UTC (consistent with the rest of the codebase).
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().strip('"')
        if not cleaned:
            return None
        try:
            value = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


async def is_thread_stale(
    saver: Any,
    thread_id: str,
    *,
    ttl: timedelta,
    now: datetime | None = None,
    last_activity: datetime | None = None,
) -> bool:
    """Whether a thread's newest checkpoint is older than *ttl*.

    Threads whose age cannot be determined return False (kept) — deleting state
    we cannot date is a destructive guess, and the cost of keeping it is a few
    rows.

    Args:
        last_activity: pre-fetched newest timestamp, to avoid a per-thread
            query. When None it is looked up individually.
    """
    if last_activity is None:
        activities = await thread_last_activity(saver)
        last_activity = activities.get(thread_id)
        if last_activity is None:
            # Either the thread vanished or its timestamp is unreadable. Both
            # mean "cannot date it" → keep.
            return False

    now = now or datetime.now(timezone.utc)
    return (now - last_activity) > ttl


async def run_retention(
    saver: Any,
    *,
    ttl: timedelta = timedelta(days=DEFAULT_THREAD_TTL_DAYS),
    now: datetime | None = None,
    max_threads: int = 1000,
    stale_thread_ids: Sequence[str] | None = None,
) -> PruneReport:
    """One retention pass over the checkpoint store.

    Deletes every thread whose newest checkpoint is older than *ttl*. Newer
    threads are left entirely alone — see the module docstring for why
    per-thread pruning is not an option with this saver.

    Args:
        saver: the checkpointer (``AsyncPostgresSaver``).
        ttl: threads older than this are removed.
        now: injectable clock, for tests.
        max_threads: cap on threads inspected per pass, so retention cannot
            monopolise the pool.
        stale_thread_ids: Optional pre-computed stale set. It is a **filter**
            over discovered threads, not a substitute for discovery — passing
            an empty sequence means "nothing is stale", NOT "do nothing", and
            threads are still enumerated from the saver. (An earlier version
            treated it as the thread list, which made an empty sequence a
            silent no-op.)

    Returns:
        A :class:`PruneReport`. Never raises: retention is housekeeping and
        must not be able to break the service.
    """
    report = PruneReport()
    if saver is None:
        return report

    try:
        thread_ids = await list_thread_ids(saver, limit=max_threads)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retention: could not list threads: %s", exc)
        report.errors += 1
        return report

    report.threads_examined = len(thread_ids)

    if stale_thread_ids is not None:
        precomputed = set(stale_thread_ids)
        stale = [tid for tid in thread_ids if tid in precomputed]
    else:
        # One query for every thread's newest timestamp, rather than one per
        # thread — with a large table the per-thread form is O(n) round-trips.
        activities = await thread_last_activity(saver, limit=max_threads)
        stale = []
        for tid in thread_ids:
            if await is_thread_stale(
                saver, tid, ttl=ttl, now=now, last_activity=activities.get(tid)
            ):
                stale.append(tid)

    for tid in stale:
        if await delete_thread(saver, tid):
            report.threads_deleted += 1
        else:
            report.errors += 1

    report.threads_kept = len(thread_ids) - report.threads_deleted - report.errors

    logger.info(
        "Checkpoint retention: examined %d threads, deleted %d, kept %d, %d errors",
        report.threads_examined,
        report.threads_deleted,
        report.threads_kept,
        report.errors,
    )
    return report
