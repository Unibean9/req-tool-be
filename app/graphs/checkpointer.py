import asyncio
import base64
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, AsyncContextManager

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

from app.models.agent import AgentSession

# Per-session locks serialize the read-modify-write on the single graph_checkpoint
# JSON column. LangGraph issues checkpoint writes concurrently within a turn; without
# this, concurrent aput/aput_writes clobber each other's pending_writes, corrupting
# resume non-deterministically. All checkpoint ops for a thread run in one event loop,
# so an in-process asyncio.Lock is sufficient.
_session_locks: dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


class DelegatingCheckpointer(BaseCheckpointSaver):
    """Delegates to per-session AgentSessionCheckpointer based on thread_id in config."""

    def __init__(self, session_factory: Callable[[], AsyncContextManager[AsyncSession]]):
        super().__init__()
        self.session_factory = session_factory

    def _for(self, config: RunnableConfig) -> "AgentSessionCheckpointer":
        thread_id = config["configurable"]["thread_id"]
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
        session_factory: Callable[[], AsyncContextManager[AsyncSession]],
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
                session = await self._get_session(db)
                prior = session.graph_checkpoint or {}
                # Keep only writes already recorded for THIS checkpoint (LangGraph
                # writes a step's values before its aput); drop superseded checkpoints'
                # writes so stale interrupts never leak into the next turn's resume.
                kept = [
                    item for item in prior.get("pending_writes", [])
                    if item.get("checkpoint_id") == checkpoint_id
                ]
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
        task_path: str = "",
    ) -> None:
        checkpoint_id = (config.get("configurable") or {}).get("checkpoint_id")
        async with _lock_for(str(self.session_id)):
            async with self.session_factory() as db:
                session = await self._get_session(db)
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
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        item = await self.aget_tuple(config or {"configurable": {"thread_id": str(self.session_id)}})
        if item is not None:
            yield item

    async def _get_session(self, db: AsyncSession) -> AgentSession:
        session = (
            await db.execute(
                select(AgentSession).where(AgentSession.id == self.session_id),
            )
        ).scalar_one()
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
