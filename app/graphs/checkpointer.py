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
        payload = {
            "data": base64.b64encode(checkpoint_bytes).decode("ascii"),
            "serde_type": serde_type,
            "metadata": dict(metadata),
            "new_versions": dict(new_versions),
            "pending_writes": [],
        }
        async with self.session_factory() as db:
            session = await self._get_session(db)
            session.graph_checkpoint = payload
            await db.commit()

        return self._checkpoint_config(config, checkpoint)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        async with self.session_factory() as db:
            session = await self._get_session(db)
            payload = session.graph_checkpoint or {}

        if not payload.get("data"):
            return None

        checkpoint = self._load_checkpoint(payload)
        return CheckpointTuple(
            config=self._checkpoint_config(config, checkpoint),
            checkpoint=checkpoint,
            metadata=payload.get("metadata") or {},
            parent_config=None,
            pending_writes=[self._load_pending_write(item) for item in payload.get("pending_writes", [])],
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        async with self.session_factory() as db:
            session = await self._get_session(db)
            payload = dict(session.graph_checkpoint or {})
            payload["pending_writes"] = [
                *payload.get("pending_writes", []),
                *[self._dump_pending_write(task_id, channel, value) for channel, value in writes],
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

    def _dump_pending_write(self, task_id: str, channel: str, value: Any) -> dict[str, Any]:
        serde_type, value_bytes = self.serde.dumps_typed(value)
        return {
            "task_id": task_id,
            "channel": channel,
            "serde_type": serde_type,
            "data": base64.b64encode(value_bytes).decode("ascii"),
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
