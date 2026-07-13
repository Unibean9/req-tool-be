import base64
import uuid
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from langgraph.graph import END, StateGraph
from sqlalchemy import select

from app.graphs.checkpointer import AgentSessionCheckpointer
from app.graphs.decision_graph import create_node
from app.graphs.state import WorkflowState, build_initial_workflow_state
from app.models.agent import AgentSession
from tests.conftest import TestSessionFactory
from tests.factories import _session_factory
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_aput_persists_base64_checkpoint(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()

    result_config = await checkpointer.aput(_config(session.id), checkpoint, {"source": "unit"}, {"messages": "1"})

    stored = await _load_agent_session(session.id)
    raw = stored.graph_checkpoint
    assert result_config["configurable"]["thread_id"] == str(session.id)
    assert result_config["configurable"]["checkpoint_id"] == checkpoint["id"]
    assert raw["metadata"] == {"source": "unit"}
    assert raw["new_versions"] == {"messages": "1"}
    assert isinstance(base64.b64decode(raw["data"]), bytes)
    assert raw["serde_type"]


@pytest.mark.asyncio
async def test_aget_tuple_returns_checkpoint_after_aput(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    config = _config(session.id)

    await checkpointer.aput(config, checkpoint, {"step": 1}, {"messages": "1"})
    loaded = await checkpointer.aget_tuple(config)

    assert isinstance(loaded, CheckpointTuple)
    assert loaded.config["configurable"]["thread_id"] == str(session.id)
    assert loaded.config["configurable"]["checkpoint_id"] == checkpoint["id"]
    assert loaded.checkpoint == checkpoint
    assert loaded.metadata == {"step": 1}
    assert loaded.parent_config is None


@pytest.mark.asyncio
async def test_checkpoint_round_trip_preserves_data(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    checkpoint["channel_values"]["analysis_result"] = {"confidence": 0.82, "next_action": "ask_human"}

    await checkpointer.aput(_config(session.id), checkpoint, {"source": "round-trip"}, {})
    loaded = await checkpointer.aget_tuple(_config(session.id))

    assert loaded is not None
    assert loaded.checkpoint == checkpoint


@pytest.mark.asyncio
async def test_aget_tuple_without_checkpoint_returns_none(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())

    assert await checkpointer.aget_tuple(_config(session.id)) is None


@pytest.mark.asyncio
async def test_aput_writes_persists_pending_writes_and_alist_returns_latest(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    config = await checkpointer.aput(_config(session.id), checkpoint, {"source": "list"}, {})

    await checkpointer.aput_writes(config, [("messages", [{"role": "user", "content": "ok"}])], task_id="task-1")
    latest = [item async for item in checkpointer.alist(config)]

    assert len(latest) == 1
    assert latest[0].checkpoint == checkpoint
    assert latest[0].pending_writes == [("task-1", "messages", [{"role": "user", "content": "ok"}])]


@pytest.mark.asyncio
async def test_checkpointer_opens_new_session_for_each_call(client, db_session):
    session = await _create_agent_session(client, db_session)
    calls = 0

    @asynccontextmanager
    async def session_factory():
        nonlocal calls
        calls += 1
        async with TestSessionFactory() as db:
            yield db

    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=session_factory)

    await checkpointer.aput(_config(session.id), _checkpoint(), {}, {})
    await checkpointer.aget_tuple(_config(session.id))

    assert calls == 2


@pytest.mark.asyncio
async def test_aput_falls_back_to_session_id_when_config_has_no_thread_id(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())

    config = await checkpointer.aput({}, _checkpoint(), {}, {})

    assert config["configurable"]["thread_id"] == str(session.id)


@pytest.mark.asyncio
async def test_write_paths_lock_session_row(client, db_session, monkeypatch):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    original_get_session = checkpointer._get_session
    flags = []

    async def tracked_get_session(db, *, for_update=False):
        flags.append(for_update)
        return await original_get_session(db, for_update=for_update)

    monkeypatch.setattr(checkpointer, "_get_session", tracked_get_session)

    config = await checkpointer.aput(_config(session.id), _checkpoint(), {}, {})
    await checkpointer.aput_writes(config, [("messages", [{"role": "user", "content": "ok"}])], task_id="task-1")

    assert flags == [True, True]


@pytest.mark.asyncio
async def test_turn_failed_resume_preserves_plain_overwrite_state_via_real_graph(client, db_session):
    """Empirically validates the phase-03 checkpoint-consistency assumption (decision 5) against a
    REAL compiled LangGraph graph and the REAL AgentSessionCheckpointer — not mocked _run_graph.

    Turn 1 completes normally and commits draft_body/decision_nodes to the checkpoint (prior
    analytical progress). Turn 2's node raises mid-node — simulating a crash — so its task never
    reaches aput_writes/aput and nothing about it is committed. Turn 3 then invokes the graph with
    ONLY the minimal partial-state update TURN_FAILED continuation uses (new message + turn_count
    reset, per agent_service.py's is_turn_failed branch) — never build_initial_workflow_state — and
    must still see turn 1's plain-overwrite fields untouched, proving the partial-state merge
    preserves prior channels rather than wiping them."""
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    config = {"configurable": {"thread_id": str(session.id)}}

    calls = {"n": 0}
    node_state = {"n1": create_node(kind="decision", statement="use REST", origin={"turn": 1})}

    async def work_node(state: WorkflowState) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"draft_body": "draft-from-completed-turn-1", "decision_nodes": node_state, "turn_count": 1}
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-node")
        return {"turn_count": state["turn_count"] + 1}

    builder = StateGraph(WorkflowState)
    builder.add_node("work", work_node)
    builder.set_entry_point("work")
    builder.add_edge("work", END)
    graph = builder.compile(checkpointer=checkpointer)

    # Turn 1: completes normally, commits draft_body/decision_nodes to the real checkpoint.
    initial_state = build_initial_workflow_state(artifact_type="goal", workflow_area="analysis", step_key=None)
    turn1_result = await graph.ainvoke(initial_state, config)
    assert turn1_result["draft_body"] == "draft-from-completed-turn-1"
    assert turn1_result["decision_nodes"] == node_state

    # Turn 2: the node crashes mid-turn (like a cancelled/timed-out node per checkpointer.py's
    # aput/aput_writes ordering) — nothing about this turn is ever committed.
    crash_state = {"messages": [{"role": "user", "content": "keep going"}], "turn_count": 0}
    with pytest.raises(RuntimeError, match="simulated crash mid-node"):
        await graph.ainvoke(crash_state, config)

    # Turn 3: the TURN_FAILED continuation's exact minimal partial-state shape (agent_service.py's
    # is_turn_failed branch) — never build_initial_workflow_state's full-default dict.
    resume_state = {"messages": [{"role": "user", "content": "tiep tuc"}], "turn_count": 0}
    turn3_result = await graph.ainvoke(resume_state, config)

    assert turn3_result["draft_body"] == "draft-from-completed-turn-1"
    assert turn3_result["decision_nodes"] == node_state


async def _create_agent_session(client, db_session) -> AgentSession:
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    session = AgentSession(
        project_id=uuid.UUID(project["id"]),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.commit()
    return session


async def _load_agent_session(session_id: uuid.UUID) -> AgentSession:
    async with TestSessionFactory() as db:
        return (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()


def _config(session_id: uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(session_id)}}


def _checkpoint() -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [{"role": "user", "content": "Hello"}]}
    checkpoint["channel_versions"] = {"messages": "1"}
    checkpoint["versions_seen"] = {"analyze": {"messages": "1"}}
    return checkpoint
