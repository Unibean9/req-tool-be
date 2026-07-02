"""Every gate decision emits exactly one structured `gate=` log line.

Uses `caplog` and the existing per-gate fixtures; no new end-to-end scaffolding. Each test asserts a
single line from the `app.graphs.gate_logging` logger with the expected gate name and verdict.
"""

import logging
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.graphs.agent_tools import (
    CRITIQUE_ROUNDS_MAX,
    _finalize_gate_open,
    _run_critique_impl,
    _run_readiness_check_impl,
    _write_draft_impl,
)
from app.graphs.critique import _invoke_judge
from app.graphs.gate_logging import log_gate_decision
from app.graphs.nodes import orchestrator_node
from app.models.agent import AgentToolCall
from app.models.artifact import ArtifactType
from app.schemas.artifact_synthesis import ArtifactReadinessState
from app.services.agent_service import AgentService
from tests.conftest import TestSessionFactory
from tests.integration.test_graph_nodes import _config, _make_agent_run, _session_factory, _state
from tests.unit.test_run_critique_tool import _draft_state, _scripted_client
from tests.unit.test_tool_parity import _focused_items, _make_agent_session, _project

GATE_LOGGER = "app.graphs.gate_logging"


def _gate_lines(caplog):
    return [r for r in caplog.records if r.name == GATE_LOGGER]


@pytest.fixture(autouse=True)
def _capture_gate_debug(caplog):
    caplog.set_level(logging.DEBUG, logger=GATE_LOGGER)


# --- helper format -----------------------------------------------------------


def test_helper_levels_and_format(caplog):
    log_gate_decision("demo", "pass", score=0.91, session_id="s1")
    log_gate_decision("demo", "blocked", reason="nope")
    lines = _gate_lines(caplog)
    assert len(lines) == 2
    assert lines[0].levelno == logging.DEBUG
    assert "gate=demo verdict=pass score=0.91 session_id=s1" == lines[0].getMessage()
    assert lines[1].levelno == logging.INFO
    assert "gate=demo verdict=blocked" in lines[1].getMessage()


# --- critique ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_critique_pass_logs_one_line(caplog):
    config = {"configurable": {"llm_client": _scripted_client(0.9, [], []), "thread_id": "t1"}}
    await _run_critique_impl("draft", "completeness", _draft_state(), config, "c1")
    lines = [r for r in _gate_lines(caplog) if "gate=critique" in r.getMessage()]
    assert len(lines) == 1
    assert "verdict=pass" in lines[0].getMessage()


@pytest.mark.asyncio
async def test_critique_fail_logs_one_line(caplog):
    config = {"configurable": {"llm_client": _scripted_client(0.4, ["x"], ["y"]), "thread_id": "t1"}}
    await _run_critique_impl("draft", "completeness", _draft_state(), config, "c1")
    lines = [r for r in _gate_lines(caplog) if "gate=critique" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert "verdict=fail" in lines[0].getMessage()


# --- finalize ----------------------------------------------------------------


def test_finalize_blocked_logs_reason(caplog):
    assert _finalize_gate_open({}) is False
    lines = [r for r in _gate_lines(caplog) if "gate=finalize" in r.getMessage()]
    assert len(lines) == 1
    assert "verdict=blocked" in lines[0].getMessage()
    assert "critique_not_passed" in lines[0].getMessage()


def test_finalize_open_at_rounds_cap(caplog):
    state = {
        "quality_report": {"quality_gate_result": "pass"},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT},
        "critique_rounds": CRITIQUE_ROUNDS_MAX,
    }
    assert _finalize_gate_open(state) is True
    lines = [r for r in _gate_lines(caplog) if "gate=finalize" in r.getMessage()]
    assert len(lines) == 1
    assert "verdict=open" in lines[0].getMessage()


# --- diagnosis ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnosis_logs_one_line(caplog):
    await orchestrator_node(_state(artifact_type="brd"), None)
    lines = [r for r in _gate_lines(caplog) if "gate=diagnosis" in r.getMessage()]
    assert len(lines) == 1


# --- judge parse failure -----------------------------------------------------


@pytest.mark.asyncio
async def test_judge_unparseable_no_longer_silent(caplog):
    client = AsyncMock()
    client.generate = AsyncMock(side_effect=ValueError("truncated JSON"))
    await _invoke_judge("body", "completeness", llm_client=client)
    lines = [r for r in _gate_lines(caplog) if "gate=judge_unparseable" in r.getMessage()]
    assert len(lines) == 1
    assert "verdict=degraded" in lines[0].getMessage()
    assert "ValueError" in lines[0].getMessage()


# --- candidate-readiness persist rejection -----------------------------------


def test_persist_rejection_logs_422(caplog):
    svc = AgentService.__new__(AgentService)
    snapshot = {"artifact_type": "vision_objectives", "body": "", "focused_artifact_id": str(uuid.uuid4())}
    meta = {"artifact_type": "vision_objectives", "focused_artifact_id": snapshot["focused_artifact_id"]}
    with pytest.raises(HTTPException):
        svc._validate_candidate_readiness_for_persist(snapshot, meta)
    lines = [r for r in _gate_lines(caplog) if "gate=candidate_readiness_persist" in r.getMessage()]
    assert len(lines) == 1
    assert "verdict=rejected_422" in lines[0].getMessage()


# --- readiness (reuses the shared DB helpers) --------------------------------


@pytest.mark.asyncio
async def test_readiness_check_logs_one_line(caplog, client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _run_readiness_check_impl("target", state, config, "call_1")
    lines = [r for r in _gate_lines(caplog) if "gate=readiness" in r.getMessage()]
    assert len(lines) == 1


# --- deterministic proposal gate ---------------------------------------------


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_deterministic_gate_logs_block(mock_interrupt, caplog, client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _write_draft_impl("", "## Vision\nConcrete.", state, config, "call_1")
    lines = [r for r in _gate_lines(caplog) if "gate=deterministic_proposal" in r.getMessage()]
    assert len(lines) == 1
    assert "verdict=blocked" in lines[0].getMessage()
    async with TestSessionFactory() as db:
        rows = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalars().all()
        assert rows == []
