"""Durable job data layer for a turn's execution: enqueue/claim/renew/reclaim/complete.

`AgentTurnJob` is not a second turn identity — it carries no sequence of its own, only a
unique `turn_id` reference to the already-admitted `AgentTurnEnvelope` plus the
`TurnExecutionState` version the caller expected at enqueue time. Every claim/renew/reclaim/
complete operation here is a single row-locked Postgres transaction (`SELECT ... FOR UPDATE` or
`FOR UPDATE SKIP LOCKED`); none of them retries at the application layer in place of a real
database lock, and none of them holds a transaction open across an LLM/tool call — this module
has no LLM/tool call in it at all, and the functions here are built so a future caller that adds
one cannot accidentally nest it inside one of these transactions.

`lease_generation` is always written in lockstep with `TurnExecutionState.ownership_generation`
inside the same transaction: the execution-state column is the actual fence source of truth, and
this module never lets the two drift apart.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import (
    AgentTurnEnvelope,
    AgentTurnJob,
    AgentTurnJobStatus,
    TurnExecutionState,
)

LEASE_SECONDS = 60
MAX_ATTEMPTS_BEFORE_DEAD_LETTER = 5


def _lease_expired(lease_expires_at: datetime | None, now: datetime) -> bool:
    """No lease at all counts as expired (claimable); SQLite round-trips DateTime(timezone=True)
    as naive while Postgres keeps it aware, so normalize before comparing."""
    if lease_expires_at is None:
        return True
    if lease_expires_at.tzinfo is None:
        lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
    return lease_expires_at < now


class AgentTurnJobService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def enqueue(
        self,
        *,
        turn_id: uuid.UUID,
        expected_transition_version: int,
        cohort: dict,
    ) -> AgentTurnJob:
        """Idempotent per turn, not per call: a second enqueue for a turn that already has a job
        returns the existing job instead of creating a duplicate. The unique constraint on
        `turn_id` is the actual exactly-once guarantee; a concurrent-insert race is resolved by
        catching the constraint violation and reading back the winner's row, the same pattern
        already used for this codebase's other logical-identity ledgers.
        """
        if self.db.in_transaction():
            await self.db.commit()
        try:
            async with self.db.begin():
                existing = (
                    await self.db.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == turn_id))
                ).scalar_one_or_none()
                if existing is not None:
                    return existing
                job = AgentTurnJob(
                    turn_id=turn_id,
                    expected_transition_version=expected_transition_version,
                    cohort=cohort,
                    status=AgentTurnJobStatus.QUEUED,
                )
                self.db.add(job)
                await self.db.flush()
                return job
        except IntegrityError as exc:
            # A concurrent enqueue for the same turn committed first; fall through and read its
            # row back in a fresh transaction instead of surfacing the loser's constraint error.
            integrity_error = exc
        async with self.db.begin():
            existing = (
                await self.db.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == turn_id))
            ).scalar_one_or_none()
            if existing is None:
                # The IntegrityError was not actually the expected turn_id race (e.g. a foreign
                # key violation on a bad turn_id); re-raise the real cause instead of masking it.
                raise integrity_error
            return existing

    async def claim(self, *, worker_id: str) -> AgentTurnJob | None:
        """Claim exactly one claimable job: `queued`, or `claimed`/`running` with an expired
        lease (self-healing at claim time, in addition to the separate `reclaim_expired` scan).
        Bumps `TurnExecutionState.ownership_generation` in the same transaction as the job lease
        write — that column, not this job's own copy, is the fence's source of truth.

        Head-of-line ordering per session is enforced at the SQL level, not by an app-side probe:
        the candidate set is restricted to, per session, only the job with the lowest
        `session_sequence` among its non-terminal jobs. A later-sequence job for the same session
        never enters the candidate set at all while an earlier one is still non-terminal,
        regardless of that earlier job's own lock/lease state — this avoids the stale-snapshot
        race a plain non-locking "is an earlier job still claimed" read would be exposed to.
        """
        now = datetime.now(UTC)
        if self.db.in_transaction():
            await self.db.commit()
        async with self.db.begin():
            non_terminal = [AgentTurnJobStatus.QUEUED, AgentTurnJobStatus.CLAIMED, AgentTurnJobStatus.RUNNING]
            head_seq = (
                select(
                    AgentTurnEnvelope.session_id.label("session_id"),
                    func.min(AgentTurnEnvelope.session_sequence).label("min_seq"),
                )
                .join(AgentTurnJob, AgentTurnJob.turn_id == AgentTurnEnvelope.id)
                .where(AgentTurnJob.status.in_(non_terminal))
                .group_by(AgentTurnEnvelope.session_id)
                .subquery()
            )
            rows = (
                await self.db.execute(
                    select(AgentTurnJob)
                    .join(AgentTurnEnvelope, AgentTurnEnvelope.id == AgentTurnJob.turn_id)
                    .join(
                        head_seq,
                        and_(
                            head_seq.c.session_id == AgentTurnEnvelope.session_id,
                            head_seq.c.min_seq == AgentTurnEnvelope.session_sequence,
                        ),
                    )
                    .where(AgentTurnJob.status.in_(non_terminal))
                    .with_for_update(of=AgentTurnJob, skip_locked=True)
                )
            ).scalars().all()
            for job in rows:
                if job.status != AgentTurnJobStatus.QUEUED and not _lease_expired(job.lease_expires_at, now):
                    continue
                state = (
                    await self.db.execute(
                        select(TurnExecutionState).where(TurnExecutionState.turn_id == job.turn_id).with_for_update()
                    )
                ).scalar_one_or_none()
                if state is None:
                    continue
                if job.status != AgentTurnJobStatus.QUEUED:
                    # Self-healing an expired lease is still a retry attempt: apply the same
                    # attempt/dead-letter accounting as reclaim_expired, so a crash-looping
                    # worker cannot dodge the dead-letter cap by out-racing the recovery scanner.
                    job.attempt += 1
                    if job.attempt >= MAX_ATTEMPTS_BEFORE_DEAD_LETTER:
                        job.status = AgentTurnJobStatus.DEAD_LETTER
                        job.lease_owner = None
                        job.lease_expires_at = None
                        state.ownership_generation += 1
                        job.lease_generation = state.ownership_generation
                        continue
                state.ownership_generation += 1
                job.lease_owner = worker_id
                job.lease_generation = state.ownership_generation
                job.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)
                job.heartbeat_at = now
                job.status = AgentTurnJobStatus.CLAIMED
                return job
        return None

    async def renew_heartbeat(self, *, job_id: uuid.UUID, worker_id: str, lease_generation: int) -> bool:
        """CAS on `(job_id, lease_owner, lease_generation)`: extend the lease only if the caller
        is still the current owner at the generation it last observed; otherwise the caller has
        already been reclaimed and must stop, not keep executing."""
        now = datetime.now(UTC)
        if self.db.in_transaction():
            await self.db.commit()
        async with self.db.begin():
            job = (
                await self.db.execute(select(AgentTurnJob).where(AgentTurnJob.id == job_id).with_for_update())
            ).scalar_one_or_none()
            if job is None:
                return False
            if job.lease_owner != worker_id or job.lease_generation != lease_generation:
                return False
            if job.status not in (AgentTurnJobStatus.CLAIMED, AgentTurnJobStatus.RUNNING):
                return False
            job.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)
            job.heartbeat_at = now
            return True

    async def complete(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        lease_generation: int,
        status: AgentTurnJobStatus,
        error: str | None = None,
    ) -> bool:
        """Same CAS discipline as `renew_heartbeat`: only transition to a terminal status if the
        generation still matches. A stale caller that already lost its lease to a reclaim gets
        `False` and must not overwrite the new owner's state."""
        if status not in (AgentTurnJobStatus.SUCCEEDED, AgentTurnJobStatus.FAILED):
            raise ValueError("complete() only accepts SUCCEEDED or FAILED as a terminal status")
        if self.db.in_transaction():
            await self.db.commit()
        async with self.db.begin():
            job = (
                await self.db.execute(select(AgentTurnJob).where(AgentTurnJob.id == job_id).with_for_update())
            ).scalar_one_or_none()
            if job is None:
                return False
            if job.lease_owner != worker_id or job.lease_generation != lease_generation:
                return False
            if job.status not in (AgentTurnJobStatus.CLAIMED, AgentTurnJobStatus.RUNNING):
                return False
            job.status = status
            if error is not None:
                job.last_error = error
            return True

    async def reclaim_expired(self) -> list[AgentTurnJob]:
        """Scan for stale leases and requeue (or dead-letter past the attempt cap) each one under
        its own row lock. This is the function a recovery scanner calls periodically; this
        increment only needs it to be correct and tested, not actually scheduled.
        """
        now = datetime.now(UTC)
        if self.db.in_transaction():
            await self.db.commit()
        candidate_ids = (
            await self.db.execute(
                select(AgentTurnJob.id).where(
                    AgentTurnJob.status.in_([AgentTurnJobStatus.CLAIMED, AgentTurnJobStatus.RUNNING])
                )
            )
        ).scalars().all()
        # The scan above auto-begins a transaction of its own; close it before the loop opens
        # its own explicit per-job transaction below.
        if self.db.in_transaction():
            await self.db.commit()
        reclaimed: list[AgentTurnJob] = []
        for job_id in candidate_ids:
            async with self.db.begin():
                job = (
                    await self.db.execute(select(AgentTurnJob).where(AgentTurnJob.id == job_id).with_for_update())
                ).scalar_one_or_none()
                if job is None:
                    continue
                # Re-check under lock: the owner may have renewed or completed between the scan
                # above and acquiring this row lock, in which case there is nothing to reclaim.
                if job.status not in (AgentTurnJobStatus.CLAIMED, AgentTurnJobStatus.RUNNING):
                    continue
                if not _lease_expired(job.lease_expires_at, now):
                    continue
                state = (
                    await self.db.execute(
                        select(TurnExecutionState).where(TurnExecutionState.turn_id == job.turn_id).with_for_update()
                    )
                ).scalar_one_or_none()
                if state is None:
                    continue
                # Only bump the fence if it still matches the generation this job's stale lease
                # was minted against; if it already moved on, some other path already reclaimed it.
                if state.ownership_generation != job.lease_generation:
                    continue
                state.ownership_generation += 1
                job.lease_generation = state.ownership_generation
                job.lease_owner = None
                job.lease_expires_at = None
                job.attempt += 1
                job.status = (
                    AgentTurnJobStatus.DEAD_LETTER
                    if job.attempt >= MAX_ATTEMPTS_BEFORE_DEAD_LETTER
                    else AgentTurnJobStatus.QUEUED
                )
                reclaimed.append(job)
        return reclaimed
