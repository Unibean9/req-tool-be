"""Durable turn worker CLI entrypoint.

Runs as a separate process from the web app: it polls `AgentTurnJobService.claim()` for
claimable jobs and executes each one with `execute_claimed_job`, alongside the recovery scanner
loop that requeues/dead-letters expired leases. Only turns admitted with `execution_mode="durable"`
in their cohort ever land here — no default/env config enables that mode, so this entrypoint has
no observable effect unless a caller opts a turn into it explicitly.

Unlike the web app's `lifespan`, this entrypoint never runs Alembic migrations itself: it assumes
the schema is already at the revision the web process (or an operator's migration step) has
already applied, and simply fails fast against a database that is not.
"""

import asyncio
import logging
import signal
import uuid

from app.config import settings
from app.database import async_session_factory
from app.graphs.checkpointer import DelegatingCheckpointer
from app.graphs.graph import build_graph
from app.services.agent_service import AgentService
from app.services.agent_turn_job_service import AgentTurnJobService
from app.services.agent_turn_worker import _recovery_scanner_loop, execute_claimed_job

logger = logging.getLogger(__name__)


async def _claim_and_run_one(*, worker_id: str, compiled_graph) -> bool:
    """Claim at most one job and, if one was claimed, run it to completion.

    Returns whether a job was claimed, so the poll loop can skip its idle sleep and immediately
    try again after a busy claim (there may be more queued work behind it).
    """
    async with async_session_factory() as db:
        job = await AgentTurnJobService(db).claim(worker_id=worker_id)
    if job is None:
        return False

    async with async_session_factory() as db:
        agent_service = AgentService(db=db, graph=compiled_graph, session_factory=async_session_factory)
        run_graph_kwargs = await agent_service.build_run_graph_kwargs_for_turn(turn_id=job.turn_id)

    await execute_claimed_job(
        agent_service=agent_service,
        session_factory=async_session_factory,
        job=job,
        worker_id=worker_id,
        run_graph_kwargs=run_graph_kwargs,
    )
    return True


async def _poll_loop(*, worker_id: str, compiled_graph, stop_event: asyncio.Event) -> None:
    """Claim/execute jobs one at a time until `stop_event` is set.

    Never cancels an in-flight `_claim_and_run_one()` call when shutdown is requested: the loop
    only checks `stop_event` between iterations, so a turn already claimed by this worker always
    runs to completion (or fails and reports through the normal `complete()` path) rather than
    being torn down mid-execution.
    """
    while not stop_event.is_set():
        try:
            claimed = await _claim_and_run_one(worker_id=worker_id, compiled_graph=compiled_graph)
        except Exception:
            logger.exception("agent_turn_worker_poll_iteration_failed worker_id=%s", worker_id)
            claimed = False
        if not claimed:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=settings.agent_turn_worker_poll_interval_seconds)
            except TimeoutError:
                pass


async def run_worker() -> None:
    worker_id = f"worker-{uuid.uuid4()}"
    compiled_graph = build_graph(DelegatingCheckpointer(async_session_factory))
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, AttributeError):
            # add_signal_handler does not exist on asyncio's event loop API on some platforms
            # (e.g. Windows); fall back to the standard signal module there.
            signal.signal(sig, lambda *_args: stop_event.set())

    logger.info("agent_turn_worker_starting worker_id=%s", worker_id)
    poll_task = asyncio.create_task(
        _poll_loop(worker_id=worker_id, compiled_graph=compiled_graph, stop_event=stop_event)
    )
    scanner_task = asyncio.create_task(
        _recovery_scanner_loop(
            session_factory=async_session_factory,
            interval_seconds=settings.agent_turn_recovery_scan_interval_seconds,
        )
    )
    try:
        await stop_event.wait()
        logger.info("agent_turn_worker_shutdown_requested worker_id=%s", worker_id)
        # The poll loop checks stop_event between iterations and never gets cancelled mid-job;
        # only the recovery scanner (which never holds a job lease) is cancelled directly.
        await poll_task
    finally:
        scanner_task.cancel()
        try:
            await scanner_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
