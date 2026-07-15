"""Executes one already-claimed `AgentTurnJob` to completion.

This module runs a job whose lease the caller already holds (via `AgentTurnJobService.claim`); it
never claims a job itself and is not a worker loop or CLI entrypoint — the polling loop that claims
jobs and calls `execute_claimed_job` is a separate, later concern.

While the turn's graph executes, a background task renews the job's lease on a fixed interval so a
long-running turn does not get reclaimed out from under a still-live worker. Heartbeat renewal and
graph execution are independent asyncio tasks, each opening its own short-lived database session —
no transaction here is ever held open across an LLM/tool call.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.agent import AgentTurnJob, AgentTurnJobStatus
from app.services.agent_service import AgentService
from app.services.agent_turn_job_service import AgentTurnJobService

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncSession]


async def _heartbeat_loop(
    *,
    session_factory: SessionFactory,
    job_id: uuid.UUID,
    worker_id: str,
    lease_generation: int,
    interval_seconds: float,
) -> None:
    """Renew the job's lease every `interval_seconds` until cancelled.

    A `False` return from `renew_heartbeat` means the lease has already been reclaimed by another
    worker. This loop only logs that and keeps running instead of cancelling the caller's graph
    task: cancelling mid-execution could tear down an open transaction, whereas letting the graph
    run to its next mutation lets the existing command/checkpoint/outcome ownership fence reject
    the stale write on its own.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        async with session_factory() as db:
            renewed = await AgentTurnJobService(db).renew_heartbeat(
                job_id=job_id, worker_id=worker_id, lease_generation=lease_generation
            )
        if not renewed:
            logger.warning(
                "agent_turn_job_heartbeat_stale job_id=%s worker_id=%s lease_generation=%s",
                job_id,
                worker_id,
                lease_generation,
            )


async def _run_one_scan(*, session_factory: SessionFactory) -> list[AgentTurnJob]:
    """Run exactly one `reclaim_expired()` scan through a fresh short-lived session and log a
    structured line for each job it requeues or dead-letters.

    Split out from the loop below so a test can exercise the scan-and-log behavior directly
    without driving a real `while True` loop or cancelling a background task.
    """
    async with session_factory() as db:
        reclaimed = await AgentTurnJobService(db).reclaim_expired()
    for job in reclaimed:
        logger.info(
            "agent_turn_job_reclaimed job_id=%s turn_id=%s attempt=%s status=%s",
            job.id,
            job.turn_id,
            job.attempt,
            job.status.value,
        )
    return reclaimed


async def _recovery_scanner_loop(
    *,
    session_factory: SessionFactory,
    interval_seconds: float,
) -> None:
    """Periodically scan for and reclaim jobs whose lease has expired, requeueing them (or
    dead-lettering them past the attempt cap) via `AgentTurnJobService.reclaim_expired()`.

    Mirrors `_heartbeat_loop`'s shape: sleep, open a short-lived session, do one unit of work, log
    the outcome, repeat forever until the caller's task is cancelled. Not itself started from any
    CLI/entrypoint yet — a future polling worker will spawn this alongside its claim loop.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        await _run_one_scan(session_factory=session_factory)


async def execute_claimed_job(
    *,
    agent_service: AgentService,
    session_factory: SessionFactory,
    job: AgentTurnJob,
    worker_id: str,
    run_graph_kwargs: dict[str, Any],
) -> None:
    """Run one already-claimed job's turn and report the outcome back to the job ledger.

    `job` must already be claimed by `worker_id` at `job.lease_generation` — establishing that
    claim is the caller's responsibility, not this function's. `run_graph_kwargs` carries every
    `AgentService._run_graph` keyword argument this wrapper does not itself derive from `job`
    (session_id, project_id, artifact_type, step_key, workflow_area, agent_role, missing_context,
    llm_client, strong_llm_client, focused_artifact_id, initial_state, resume_command,
    allow_empty_completion, turn_id); `owner_id`/`ownership_generation` are always injected here
    from `job.lease_owner`/`job.lease_generation` and must not also be present in
    `run_graph_kwargs`.

    Always cancels the heartbeat task and calls `AgentTurnJobService.complete()` in a `finally`
    block, regardless of whether graph execution succeeded, raised a `StaleTurnOwnershipError`, or
    raised anything else. `complete()` returning `False` (the job was already reclaimed by another
    worker before this worker could report) is a normal race outcome, not an error to raise.
    """
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            session_factory=session_factory,
            job_id=job.id,
            worker_id=worker_id,
            lease_generation=job.lease_generation,
            interval_seconds=settings.agent_turn_job_heartbeat_interval_seconds,
        )
    )
    status = AgentTurnJobStatus.SUCCEEDED
    error: str | None = None
    try:
        # No public AgentService wrapper exists yet for "run this turn's graph under an explicit
        # execution lease" — this reuses the same private method the inline execution path calls,
        # rather than duplicating its graph-invocation/outcome-projection logic here.
        await agent_service._run_graph(
            **run_graph_kwargs,
            owner_id=job.lease_owner,
            ownership_generation=job.lease_generation,
        )
    except Exception as exc:
        status = AgentTurnJobStatus.FAILED
        error = str(exc) or exc.__class__.__name__
        logger.exception("agent_turn_job_execution_failed job_id=%s worker_id=%s", job.id, worker_id)
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            # A heartbeat-loop failure (e.g. a transient DB error between renews) must never skip
            # reporting the job's outcome below — the lease will simply expire and be reclaimed.
            logger.exception("agent_turn_job_heartbeat_task_failed job_id=%s worker_id=%s", job.id, worker_id)

    async with session_factory() as db:
        completed = await AgentTurnJobService(db).complete(
            job_id=job.id,
            worker_id=worker_id,
            lease_generation=job.lease_generation,
            status=status,
            error=error,
        )
    if not completed:
        logger.info(
            "agent_turn_job_complete_already_reclaimed job_id=%s worker_id=%s", job.id, worker_id
        )
