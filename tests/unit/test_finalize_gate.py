"""Finalize gate: working_draft non-empty AND critique_rounds > 0 AND quality gate passed AND candidate_readiness sufficient."""

import hashlib
import uuid
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from app.graphs.agent_tools import _finalize_impl, current_draft_body, get_available_tools
from app.schemas.artifact_synthesis import ArtifactReadinessState
from app.models.artifact import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    ChangeSource,
    VersionStatus,
)


def _names(state):
    return {t.name for t in get_available_tools(state)}


def _hash(body: str) -> str:
    return hashlib.md5(body.encode()).hexdigest()[:8]


def _passing_state(draft: str = "Một bản nháp", critique_rounds: int = 1) -> dict:
    """A state where every finalize-gate condition is satisfied (pass gate + fresh hash + sufficient readiness)."""
    return {
        "messages": [],
        "working_draft": draft,
        "critique_rounds": critique_rounds,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash(draft),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }


def test_finalize_not_available_when_critique_rounds_zero():
    state = _passing_state(critique_rounds=0)
    assert "finalize" not in _names(state)


def test_finalize_available_when_critique_rounds_positive():
    assert "finalize" in _names(_passing_state())


def test_finalize_not_available_without_working_draft():
    state = _passing_state()
    state["working_draft"] = None
    assert "finalize" not in _names(state)


def test_finalize_available_for_db_draft_without_working_draft():
    # DB-draft-only session (draft_body set, no in-session working_draft) finalizes via the same
    # current_draft_body source — the menu gate no longer requires an in-session working_draft.
    draft = "Bản nháp tải từ DB"
    state = {
        "messages": [],
        "working_draft": None,
        "draft_body": draft,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash(draft),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    names = _names(state)
    assert "finalize" in names
    assert "run_readiness_check" in names


def test_finalize_not_available_when_gate_fails():
    state = _passing_state()
    state["quality_report"] = {"quality_gate_result": "fail", "blocking_issues": ["thiếu metric"]}
    assert "finalize" not in _names(state)


def test_finalize_not_available_when_draft_edited_after_critique():
    state = _passing_state()
    state["working_draft"] = "Bản nháp đã sửa sau critique"
    assert "finalize" not in _names(state)


def test_finalize_available_on_escape_hatch_at_rounds_cap():
    from app.graphs.agent_tools import CRITIQUE_ROUNDS_MAX

    state = _passing_state(critique_rounds=CRITIQUE_ROUNDS_MAX)
    # Stale hash, but the rounds cap is reached → escape hatch keeps finalize available.
    state["last_critiqued_draft_hash"] = "deadbeef"
    assert "finalize" in _names(state)


@pytest.mark.asyncio
async def test_finalize_interrupt_triggers_when_available():
    """_finalize_impl interrupts for human confirmation (the approval step) rather than erroring."""

    class _Session:
        status = None
        interrupt_type = None

    session_row = _Session()

    class _Result:
        def scalar_one(self_inner):
            return session_row

    class _DB:
        async def execute(self_inner, *a, **k):
            return _Result()

        async def commit(self_inner):
            return None

    def _factory():
        class _Ctx:
            async def __aenter__(self_inner):
                return _DB()

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    config = {"configurable": {"session_factory": _factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        "working_draft": "draft",
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass"},
        "last_critiqued_draft_hash": _hash("draft"),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert session_row.status is not None


@pytest.mark.asyncio
async def test_finalize_hard_blocks_when_gate_fails():
    """Even if reached directly, _finalize_impl refuses a failing gate — ToolMessage, no interrupt."""
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        "working_draft": "draft",
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "fail", "blocking_issues": ["thiếu tiêu chí đo lường"]},
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    msg = command.update["messages"][0]
    assert "Không thể finalize" in msg.content
    assert "thiếu tiêu chí đo lường" in msg.content


@pytest.mark.asyncio
async def test_finalize_hard_blocks_without_current_draft_body():
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        "critique_rounds": 2,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash("draft"),
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    assert "Không thể finalize" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_current_draft_body_loads_exact_focused_artifact(db_session):
    project_id = uuid.uuid4()
    parent = Artifact(
        project_id=project_id,
        type=ArtifactType.BRD,
        status=ArtifactStatus.DRAFT,
        title="BRD",
        extra_metadata={},
    )
    db_session.add(parent)
    await db_session.flush()
    child = Artifact(
        project_id=project_id,
        parent_id=parent.id,
        type=ArtifactType.VISION_OBJECTIVES,
        status=ArtifactStatus.DRAFT,
        title="Vision",
        extra_metadata={},
    )
    db_session.add(child)
    await db_session.flush()
    version = ArtifactVersion(
        artifact_id=child.id,
        version_number=1,
        title="Vision",
        body="Tầm nhìn đã chốt",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    child.current_version_id = version.id
    await db_session.flush()

    @asynccontextmanager
    async def session_factory():
        yield db_session

    state = {
        "focused_artifact_id": str(child.id),
        "draft_body": "stale",
    }
    config = {"configurable": {"session_factory": session_factory}}
    assert await current_draft_body(state, config) == "Tầm nhìn đã chốt"


@pytest.mark.asyncio
async def test_finalize_hard_blocks_when_focused_artifact_was_not_critiqued():
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        "focused_artifact_id": "00000000-0000-0000-0000-000000000001",
        "draft_body": "Draft item B",
        "critique_rounds": 0,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash("Draft section A"),
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    msg = command.update["messages"][0]
    assert "Không thể finalize" in msg.content
    assert "bản nháp hiện tại" in msg.content
