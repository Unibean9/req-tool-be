import uuid

import pytest
from sqlalchemy import select

from app.models.agent import AgentRun, AgentSession
from tests.conftest import TestSessionFactory


async def _make_session(db) -> AgentSession:
    session = AgentSession(
        project_id=uuid.uuid4(),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db.add(session)
    await db.commit()
    return session


@pytest.mark.asyncio
async def test_agent_run_persists_token_usage_and_latency():
    async with TestSessionFactory() as db:
        session = await _make_session(db)
        run = AgentRun(
            session_id=session.id,
            analysis_result={},
            token_usage={"input": 10, "output": 20, "total": 30},
            latency_ms=150,
        )
        db.add(run)
        await db.commit()
        run_id = run.id

    async with TestSessionFactory() as db:
        loaded = (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one()
        assert loaded.token_usage == {"input": 10, "output": 20, "total": 30}
        assert loaded.latency_ms == 150


@pytest.mark.asyncio
async def test_agent_run_columns_are_nullable():
    async with TestSessionFactory() as db:
        session = await _make_session(db)
        run = AgentRun(session_id=session.id, analysis_result={})
        db.add(run)
        await db.commit()
        run_id = run.id

    async with TestSessionFactory() as db:
        loaded = (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one()
        assert loaded.token_usage is None
        assert loaded.latency_ms is None
