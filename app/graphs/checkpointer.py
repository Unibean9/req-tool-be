import asyncio
import base64
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graphs.analysis.turn_event_log import emit_turn_event
from app.graphs.analysis.turn_outcome_projector import check_ownership_fence
from app.models.agent import AgentCheckpoint, AgentSession, AgentTurnEnvelope, AgentTurnEventType

logger = logging.getLogger(__name__)

# Per-session locks serialize in-process writes; row locks cover multi-worker checkpoint writes.
_session_locks: dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


class DelegatingCheckpointer(BaseCheckpointSaver):
    """Delegates to per-session AgentSessionCheckpointer based on thread_id in config."""

    def __init__(self, session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]]):
        super().__init__()
        self.session_factory = session_factory

    def _for(self, config: RunnableConfig) -> "AgentSessionCheckpointer | AgentCheckpointHistorySaver":
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        # checkpoint_version is snapshotted onto configurable at graph-config build time
        # (AgentService._make_config) from the already-loaded AgentSession row — never read live
        # here. Absent entirely (e.g. a config built outside AgentService) defaults to "v1", the
        # only cohort every pre-existing session/config shape is valid for.
        checkpoint_version = configurable.get("checkpoint_version", "v1")
        if checkpoint_version == "v2":
            return AgentCheckpointHistorySaver(session_id=thread_id, session_factory=self.session_factory)
        return AgentSessionCheckpointer(session_id=thread_id, session_factory=self.session_factory)

    async def aget_tuple(self, config: RunnableConfig) -> "CheckpointTuple | None":
        return await self._for(config).aget_tuple(config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._for(config).aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await self._for(config).aput_writes(config, writes, task_id, task_path)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        async for item in self._for(config).alist(config, filter=filter, before=before, limit=limit):
            yield item


class AgentSessionCheckpointer(BaseCheckpointSaver):
    def __init__(
        self,
        *,
        session_id: str,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ):
        super().__init__()
        self.session_id = uuid.UUID(session_id)
        self.session_factory = session_factory

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        serde_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint)
        checkpoint_id = checkpoint["id"]
        async with _lock_for(str(self.session_id)):
            async with self.session_factory() as db:
                session = await self._get_session(db, for_update=True)
                prior = session.graph_checkpoint or {}
                # Keep only writes already recorded for THIS checkpoint (LangGraph
                # writes a step's values before its aput); drop superseded checkpoints'
                # writes so stale interrupts never leak into the next turn's resume.
                kept = [item for item in prior.get("pending_writes", []) if item.get("checkpoint_id") == checkpoint_id]
                session.graph_checkpoint = {
                    "data": base64.b64encode(checkpoint_bytes).decode("ascii"),
                    "serde_type": serde_type,
                    "metadata": dict(metadata),
                    "new_versions": dict(new_versions),
                    "checkpoint_id": checkpoint_id,
                    "pending_writes": kept,
                }
                await db.commit()

        return self._checkpoint_config(config, checkpoint)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        async with self.session_factory() as db:
            session = await self._get_session(db)
            payload = session.graph_checkpoint or {}

        if not payload.get("data"):
            return None

        checkpoint = self._load_checkpoint(payload)
        current_id = payload.get("checkpoint_id")
        pending = [
            self._load_pending_write(item)
            for item in payload.get("pending_writes", [])
            if current_id is None or item.get("checkpoint_id") == current_id
        ]
        return CheckpointTuple(
            config=self._checkpoint_config(config, checkpoint),
            checkpoint=checkpoint,
            metadata=payload.get("metadata") or {},
            parent_config=None,
            pending_writes=pending,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",  # noqa: ARG002 — required by BaseCheckpointSaver override signature
    ) -> None:
        checkpoint_id = (config.get("configurable") or {}).get("checkpoint_id")
        async with _lock_for(str(self.session_id)):
            async with self.session_factory() as db:
                session = await self._get_session(db, for_update=True)
                payload = dict(session.graph_checkpoint or {})
                cid = checkpoint_id or payload.get("checkpoint_id")
                payload["pending_writes"] = [
                    *payload.get("pending_writes", []),
                    *[self._dump_pending_write(task_id, channel, value, cid) for channel, value in writes],
                ]
                session.graph_checkpoint = payload
                await db.commit()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        # Required by the BaseCheckpointSaver.alist signature; this single-checkpoint
        # store ignores the query filters and returns the one tuple it holds.
        filter: dict[str, Any] | None = None,  # noqa: ARG002
        before: RunnableConfig | None = None,  # noqa: ARG002
        limit: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[CheckpointTuple]:
        item = await self.aget_tuple(config or {"configurable": {"thread_id": str(self.session_id)}})
        if item is not None:
            yield item

    async def _get_session(self, db: AsyncSession, *, for_update: bool = False) -> AgentSession:
        stmt = select(AgentSession).where(AgentSession.id == self.session_id)
        if for_update:
            stmt = stmt.with_for_update()
        session = (await db.execute(stmt)).scalar_one()
        return session

    def _load_checkpoint(self, payload: dict[str, Any]) -> Checkpoint:
        checkpoint_bytes = base64.b64decode(payload["data"])
        return self.serde.loads_typed((payload["serde_type"], checkpoint_bytes))

    def _dump_pending_write(
        self, task_id: str, channel: str, value: Any, checkpoint_id: str | None = None
    ) -> dict[str, Any]:
        serde_type, value_bytes = self.serde.dumps_typed(value)
        return {
            "task_id": task_id,
            "channel": channel,
            "serde_type": serde_type,
            "data": base64.b64encode(value_bytes).decode("ascii"),
            "checkpoint_id": checkpoint_id,
        }

    def _load_pending_write(self, item: dict[str, Any]) -> tuple[str, str, Any]:
        value_bytes = base64.b64decode(item["data"])
        value = self.serde.loads_typed((item["serde_type"], value_bytes))
        return (item["task_id"], item["channel"], value)

    def _checkpoint_config(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        configurable = dict(config.get("configurable") or {})
        configurable["thread_id"] = configurable.get("thread_id") or str(self.session_id)
        configurable["checkpoint_id"] = checkpoint["id"]
        return {**config, "configurable": configurable}


class MissingCheckpointHistoryTurnContextError(Exception):
    """Raised when `AgentCheckpointHistorySaver.aput` is invoked without turn_id/turn_owner_id/
    turn_ownership_generation in `configurable`.

    A v2-cohort session is only ever admitted through the turn control plane, so every real write
    path always has this context; hitting this means a config was built outside that path (e.g. a
    test or a future caller) — the saver refuses to write unchecked rather than silently allowing an
    unfenced checkpoint into history.
    """

    def __init__(self, *, session_id: uuid.UUID) -> None:
        self.session_id = session_id
        super().__init__(
            f"session_id={session_id} is on the checkpoint v2 cohort but its checkpointer was "
            "invoked without turn_id/turn_owner_id/turn_ownership_generation in configurable"
        )


class StaleCheckpointAppendError(Exception):
    """Raised when a v2 checkpoint append's `parent_checkpoint_id` does not match the session's
    current history head — a stale owner replaying an old config, or two owners forking history
    concurrently. Mirrors `StaleTurnOwnershipError`'s shape: a plain exception the caller decides how
    to log/audit, never a silent overwrite."""

    def __init__(
        self,
        *,
        session_id: uuid.UUID,
        expected_parent_checkpoint_id: str | None,
        actual_head_checkpoint_id: str | None,
    ) -> None:
        self.session_id = session_id
        self.expected_parent_checkpoint_id = expected_parent_checkpoint_id
        self.actual_head_checkpoint_id = actual_head_checkpoint_id
        super().__init__(
            f"stale checkpoint append: session_id={session_id} "
            f"expected_parent_checkpoint_id={expected_parent_checkpoint_id} "
            f"actual_head_checkpoint_id={actual_head_checkpoint_id}"
        )


class AgentCheckpointHistorySaver(BaseCheckpointSaver):
    """Checkpoint v2: appends full history to `agent_checkpoints` under parent/fence CAS.

    Only ever instantiated by `DelegatingCheckpointer` for a config whose `checkpoint_version` is
    `"v2"` — a session cohort snapshotted once at creation time (`AgentService.create_session`), never
    read live here. `AgentSessionCheckpointer` (v1) stays completely unrelated to and unaffected by
    this class; it remains the reader for every `checkpoint_version == "v1"` session forever.

    Unlike v1's single mutable blob, `alist()` here returns real history: every checkpoint ever
    appended for the session, newest first, walking the `created_at`/`parent_checkpoint_id` chain the
    CAS append rule below guarantees is linear (no code path may assume it returns only one row).
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ):
        super().__init__()
        self.session_id = uuid.UUID(session_id)
        self.session_factory = session_factory

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config.get("configurable") or {}
        turn_id_raw = configurable.get("turn_id")
        owner_id = configurable.get("turn_owner_id")
        ownership_generation = configurable.get("turn_ownership_generation")
        if not turn_id_raw or owner_id is None or ownership_generation is None:
            raise MissingCheckpointHistoryTurnContextError(session_id=self.session_id)
        turn_id = uuid.UUID(str(turn_id_raw))

        serde_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint)
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = configurable.get("checkpoint_id")

        async with _lock_for(str(self.session_id)):
            async with self.session_factory() as db:
                session = await self._get_session(db, for_update=True)
                # Same-transaction fence check before the CAS append — a stale owner's write never
                # lands, regardless of what parent_checkpoint_id it thinks it has.
                await check_ownership_fence(db, turn_id, owner_id, ownership_generation)

                head = await self._get_row(db)
                actual_head_checkpoint_id = head.checkpoint_id if head is not None else None
                if actual_head_checkpoint_id != parent_checkpoint_id:
                    raise StaleCheckpointAppendError(
                        session_id=self.session_id,
                        expected_parent_checkpoint_id=parent_checkpoint_id,
                        actual_head_checkpoint_id=actual_head_checkpoint_id,
                    )

                envelope = await db.get(AgentTurnEnvelope, turn_id)
                session_sequence = envelope.session_sequence if envelope is not None else 0
                db.add(
                    AgentCheckpoint(
                        session_id=self.session_id,
                        turn_id=turn_id,
                        checkpoint_id=checkpoint_id,
                        parent_checkpoint_id=parent_checkpoint_id,
                        session_sequence=session_sequence,
                        ownership_generation=ownership_generation,
                        serde_type=serde_type,
                        data=checkpoint_bytes,
                        checkpoint_metadata=dict(metadata),
                        new_versions=dict(new_versions),
                        pending_writes=[],
                    )
                )
                # Flush the checkpoint insert on its own, outside emit_turn_event's dedup savepoint.
                # That savepoint's `except IntegrityError` is meant to swallow only the event row's own
                # unique-violation (a race re-delivering the same event); flushing the checkpoint insert
                # first keeps a genuine constraint violation on the checkpoint itself from being
                # misattributed to event dedup and rolled back silently under this commit.
                await db.flush()
                await emit_turn_event(
                    db,
                    session_row=session,
                    turn_id=turn_id,
                    event_type=AgentTurnEventType.CHECKPOINT_APPENDED,
                    parent_checkpoint_id=parent_checkpoint_id,
                    payload={"checkpoint_id": checkpoint_id},
                )
                await db.commit()

        return self._checkpoint_config(config, checkpoint)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        checkpoint_id = (config.get("configurable") or {}).get("checkpoint_id")
        async with self.session_factory() as db:
            row = await self._get_row(db, checkpoint_id=checkpoint_id)
        if row is None:
            return None
        return self._tuple_from_row(config, row)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",  # noqa: ARG002 — required by BaseCheckpointSaver override signature
    ) -> None:
        checkpoint_id = (config.get("configurable") or {}).get("checkpoint_id")
        async with _lock_for(str(self.session_id)):
            async with self.session_factory() as db:
                row = await self._get_row(db, checkpoint_id=checkpoint_id, for_update=True)
                if row is None:
                    # No explicit checkpoint_id (or it does not resolve): fall back to the current
                    # head, same leniency v1's aput_writes has for a missing checkpoint_id.
                    row = await self._get_row(db, for_update=True)
                if row is None:
                    # No checkpoint has ever been appended for this session yet — there is nothing to
                    # scope these pending writes to. Logged, not raised: a durable worker's first
                    # write of a brand-new turn must not crash on this.
                    logger.warning(
                        "agent_checkpoint_history_pending_write_without_checkpoint session_id=%s task_id=%s",
                        self.session_id,
                        task_id,
                    )
                    return
                row.pending_writes = [
                    *(row.pending_writes or []),
                    *[
                        self._dump_pending_write(task_id, channel, value, row.checkpoint_id)
                        for channel, value in writes
                    ],
                ]
                await db.commit()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: ARG002 — no metadata filter support yet
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        base_config = config or {"configurable": {"thread_id": str(self.session_id)}}
        async with self.session_factory() as db:
            stmt = select(AgentCheckpoint).where(AgentCheckpoint.session_id == self.session_id)
            if before is not None:
                before_checkpoint_id = (before.get("configurable") or {}).get("checkpoint_id")
                if before_checkpoint_id is not None:
                    before_row = await self._get_row(db, checkpoint_id=before_checkpoint_id)
                    if before_row is not None:
                        stmt = stmt.where(AgentCheckpoint.created_at < before_row.created_at)
            stmt = stmt.order_by(AgentCheckpoint.created_at.desc())
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = (await db.execute(stmt)).scalars().all()
        for row in rows:
            yield self._tuple_from_row(base_config, row)

    def _tuple_from_row(self, config: RunnableConfig, row: AgentCheckpoint) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed((row.serde_type, bytes(row.data)))
        pending = [self._load_pending_write(item) for item in (row.pending_writes or [])]
        parent_config: RunnableConfig | None = None
        if row.parent_checkpoint_id:
            parent_config = {
                "configurable": {"thread_id": str(self.session_id), "checkpoint_id": row.parent_checkpoint_id}
            }
        return CheckpointTuple(
            config=self._checkpoint_config(config, checkpoint),
            checkpoint=checkpoint,
            metadata=row.checkpoint_metadata or {},
            parent_config=parent_config,
            pending_writes=pending,
        )

    def _checkpoint_config(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        configurable = dict(config.get("configurable") or {})
        configurable["thread_id"] = configurable.get("thread_id") or str(self.session_id)
        configurable["checkpoint_id"] = checkpoint["id"]
        return {**config, "configurable": configurable}

    async def _get_session(self, db: AsyncSession, *, for_update: bool = False) -> AgentSession:
        stmt = select(AgentSession).where(AgentSession.id == self.session_id)
        if for_update:
            stmt = stmt.with_for_update()
        return (await db.execute(stmt)).scalar_one()

    async def _get_row(
        self, db: AsyncSession, *, checkpoint_id: str | None = None, for_update: bool = False
    ) -> AgentCheckpoint | None:
        stmt = select(AgentCheckpoint).where(AgentCheckpoint.session_id == self.session_id)
        if checkpoint_id is not None:
            stmt = stmt.where(AgentCheckpoint.checkpoint_id == checkpoint_id)
        else:
            stmt = stmt.order_by(AgentCheckpoint.created_at.desc()).limit(1)
        if for_update:
            stmt = stmt.with_for_update()
        return (await db.execute(stmt)).scalars().first()

    def _dump_pending_write(
        self, task_id: str, channel: str, value: Any, checkpoint_id: str | None = None
    ) -> dict[str, Any]:
        serde_type, value_bytes = self.serde.dumps_typed(value)
        return {
            "task_id": task_id,
            "channel": channel,
            "serde_type": serde_type,
            "data": base64.b64encode(value_bytes).decode("ascii"),
            "checkpoint_id": checkpoint_id,
        }

    def _load_pending_write(self, item: dict[str, Any]) -> tuple[str, str, Any]:
        value_bytes = base64.b64decode(item["data"])
        value = self.serde.loads_typed((item["serde_type"], value_bytes))
        return (item["task_id"], item["channel"], value)
