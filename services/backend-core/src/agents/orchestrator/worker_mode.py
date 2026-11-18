"""Refuse multi-worker deployments — the process keeps state single-worker needs.

Several pieces of this service live in **this process's memory** and nowhere
else:

* ``router._sse_queues`` / ``router._tasks`` — the live event stream and the
  handle the cancel endpoint cancels.  A ``/cancel`` or an SSE reconnect that
  lands on another worker cannot find them: it answers 409 ("a different
  worker owns it"), or connects to a queue that will never receive an event.
* ``learning_path.QuizStore`` — answer keys for ``/quiz/grade``.  As of the
  Redis-backed store this one *is* shared; it is listed here for contrast:
  the others still are not.

``uvicorn --workers N`` looks like a throughput knob; it is an architecture
change.  Until those stores move to Redis, the honest behaviour is to refuse
to start rather than to serve users a split brain where progress stalls at
"running" and cancel does nothing.

Three guards, each covering a different way the flag can enter a deployment:

1. :func:`guard_run` — the launch point (``uvicorn --workers 4`` or
   ``uvicorn.run(..., workers=4)``).  With ``--workers`` the parent forks
   children via ``spawn``; the children's environment never carries the flag,
   and uvicorn 0.51 has no server registry, so an ASGI app *cannot* see it.
   The parent is the only process that knows, so the check happens there.
2. :func:`verify_single_worker` — the running process, from ``lifespan``: it
   also refuses a process whose orchestrator graph has no agents wired.
3. static checks — ``Dockerfile`` and ``docs/backend-core.md`` carry a
   ``# edumap:single-worker`` sentinel; ``test_TestDeploymentFiles``
   fails if those files start using ``--workers`` or lose the sentinel.  A
   sentinel in a doc file cannot drift meaning at runtime, unlike a guess about
   how uvicorn decides to spawn.

The consequence to state plainly: a bare ``uvicorn --workers 4`` against
``src.main:app`` is **not** refused at runtime — nothing in the child can see
it.  Deployments that matter go through the Dockerfile (pinned to one worker)
or a script using :func:`guard_run`; the docs say so.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Anything above one worker is refused.  Not "warn": a warning in a container
# log nobody reads is how this would ship broken.
MAX_SUPPORTED_WORKERS = 1

_REFUSAL = (
    "backend-core does not support multiple uvicorn workers: the SSE event "
    "queues and cancel handles for /orchestrator/generate live in process "
    "memory, so a second worker makes cancellation and live progress "
    "unreliable. Start with a single worker (the Dockerfile does), or move "
    "that state to Redis first (see src/agents/orchestrator/worker_mode.py)."
)


def declared_workers(server: object) -> int | None:
    """How many workers the server was asked to run, or ``None`` if unknown.

    ``None`` is what a worker child sees for a plain ``--workers 1``/no-flag
    launch; it is treated as one.  Anything ``< 1`` is likewise treated as one
    (uvicorn would run a single process).
    """
    config = getattr(server, "config", None)
    workers = getattr(config, "workers", None)
    if not isinstance(workers, int):
        return None
    return workers


async def verify_single_worker(server: object | None = None, *, require_configured: bool = True) -> None:
    """Raise ``RuntimeError`` when this process must not serve traffic.

    Called from the app's ``lifespan`` start-up (last step before serving), so
    a refused process fails its ASGI startup and uvicorn exits with the error
    printed instead of listening on the port.

    .. warning:: **Incomplete for ``--workers N``** — see :func:`guard_run`.

    ``server`` is the bound uvicorn ``Server`` when the caller can find it.
    Without one (e.g. an ASGI startup check) the worker count is only known
    when ``uvicorn.run(..., workers=N)`` set it on ``config`` before children
    were spawned; CLI ``--workers N`` is **not** visible to a child, so this
    function alone cannot refuse it.  The dependable interception point for
    that flag is :func:`guard_run`.

    Args:
        server: the uvicorn ``Server`` bound to this app, if available.
        require_configured: also refuse when the orchestrator graph has no
            agents wired — a process that would fail every generation.
    """
    workers = declared_workers(server)
    if workers is not None and workers > MAX_SUPPORTED_WORKERS:
        logger.error("Refusing to start with --workers %d: %s", workers, _REFUSAL)
        raise RuntimeError(_REFUSAL)

    if require_configured:
        from src.agents.orchestrator.graph import is_configured

        if not is_configured():
            raise RuntimeError(
                "Orchestrator graph is not configured (no agents wired); "
                "every /orchestrator/generate would fail. Refusing to serve."
            )

    logger.info("Worker-mode check passed (single worker, graph configured)")


def guard_run(run, *args: object, **kwargs: object):
    """Wrap ``uvicorn.run`` so a multi-worker launch is refused **before** forking.

    Use it instead of ``uvicorn.run`` in launch scripts::

        from uvicorn import run
        from src.agents.orchestrator.worker_mode import guard_run
        guard_run(run, "src.main:app", workers=4)   # raises RuntimeError

    ``uvicorn.run`` takes ``workers`` as a keyword; the check happens in the
    parent process, before any child exists, which is the only place a
    CLI-equivalent ``workers=N`` value is reliably visible.  Why not patch
    ``uvicorn.Server``: uvicorn 0.51 offers no server registry, so the app
    cannot find its own ``Server`` instance from inside ``lifespan``, and a
    child process spawned by ``--workers`` never sees the flag.

    The class method ``Server.run`` carries the marker ``_edumap_guarded``
    when a run was guarded, so the *deployment-level* rule still holds: a
    launch that bypasses this wrapper is itself a bug (see the paired guard in
    the Dockerfile / compose file and ``test_TestDeploymentFiles``).
    """
    workers = kwargs.get("workers")
    if isinstance(workers, int) and workers > MAX_SUPPORTED_WORKERS:
        logger.error("Refusing to start with workers=%d: %s", workers, _REFUSAL)
        raise RuntimeError(_REFUSAL)
    return run(*args, **kwargs)
