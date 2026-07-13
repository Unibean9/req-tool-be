"""finalize must not commit a draft that has a VIOLATION section or is missing a required heading,
even when the menu gate is open (quality gate pass, hash match, readiness sufficient) — the new
guard sits right after the existing hard-block.
"""

import hashlib
from unittest.mock import patch

import pytest

from app.graphs.agent_tools import _finalize_impl
from app.graphs.agent_tools.draft_lifecycle import (
    _draft_structural_violations,
    _iter_draft_sections,
)
from app.schemas.artifact_synthesis import ArtifactReadinessState

_CLEAN_BODY = "\n\n".join(
    [
        "## Vision\nHelp students schedule study groups faster.",
        "## Objectives\nReduce time to form a group below 10 minutes.",
        "## Success Metrics\n- Successful scheduling rate reaches 80% within one semester.",
    ]
)


def _hash(body: str) -> str:
    return hashlib.md5(body.encode()).hexdigest()[:8]


def _state_for(body: str) -> dict:
    return {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "vision_objectives",
        "decision_nodes": {},
        "draft_body": body,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash(body),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }


def _working_session_factory():
    """session_factory tối thiểu cho predecessor check + cập nhật status session."""

    class _Session:
        status = None
        interrupt_type = None

    session_row = _Session()

    class _Result:
        def scalar_one(self_inner):
            return session_row

        def scalar(self_inner):
            return 1

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

    return _factory, session_row


@pytest.mark.asyncio
async def test_finalize_succeeds_on_clean_draft():
    factory, _ = _working_session_factory()
    config = {"configurable": {"session_factory": factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = _state_for(_CLEAN_BODY)

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert "tool_errors" not in command.update


@pytest.mark.asyncio
async def test_finalize_blocked_on_section_violation():
    body = "\n\n".join(
        [
            "## Vision\nHelp students schedule study groups faster.",
            "## Objectives\n_(cần bổ sung)_",
            "## Success Metrics\n- Successful scheduling rate reaches 80% within one semester.",
        ]
    )
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = _state_for(body)

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    assert command.update["tool_errors"][0]["code"] == "finalize_structural_violation"
    msg = command.update["messages"][0]
    assert msg.status == "error"
    assert "## Objectives" in msg.content
    assert "unfilled required cells" in msg.content


@pytest.mark.asyncio
async def test_finalize_blocked_on_missing_required_heading():
    body = "\n\n".join(
        [
            "## Vision\nHelp students schedule study groups faster.",
            "## Objectives\nReduce time to form a group below 10 minutes.",
        ]
    )
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = _state_for(body)

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    assert command.update["tool_errors"][0]["code"] == "finalize_structural_violation"
    msg = command.update["messages"][0]
    assert msg.status == "error"
    assert "## Success Metrics: missing required heading" in msg.content


def test_iter_draft_sections_splits_by_heading():
    sections = _iter_draft_sections(_CLEAN_BODY)

    assert [heading for heading, _ in sections] == ["## Vision", "## Objectives", "## Success Metrics"]
    assert sections[0][1] == "Help students schedule study groups faster.\n"


def test_iter_draft_sections_ignores_content_before_first_heading():
    sections = _iter_draft_sections("orphan text\n\n## Vision\nBody.")

    assert sections == [("## Vision", "Body.")]


def test_draft_structural_violations_empty_on_clean_body():
    assert _draft_structural_violations("vision_objectives", _CLEAN_BODY) == []


def test_draft_structural_violations_reports_missing_heading():
    body = "## Vision\nHelp students schedule study groups faster."

    violations = _draft_structural_violations("vision_objectives", body)

    assert "## Objectives: missing required heading" in violations
    assert "## Success Metrics: missing required heading" in violations
