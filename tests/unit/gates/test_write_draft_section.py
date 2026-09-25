"""write_draft_section: accumulate a multi-section draft across several small calls instead of one
large write_draft body -- either one call per heading, or (append=true/done=false) several row
batches for a single heading whose content is a large table -- and _resolve_proposed_body's assembly
of the accumulated sections once every required heading is present and done.
"""

from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import _write_draft_impl, _write_draft_section_impl
from app.models.agent import AgentToolCall
from app.models.artifact import ArtifactType
from tests.conftest import TestSessionFactory
from tests.factories import (
    _config,
    _focused_items,
    _make_agent_run,
    _make_agent_session,
    _project,
    _session_factory,
    _state,
)

# vision_objectives requires exactly these three headings (see app/documents/registry.py).
HEADINGS = ("## Vision", "## Objectives", "## Success Metrics")


async def _seed(client, db_session):
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
    return state, config, run, agent_session


@pytest.mark.asyncio
async def test_unknown_heading_is_rejected_with_the_valid_list(client, db_session):
    state, _config, _run, _session = await _seed(client, db_session)

    command = await _write_draft_section_impl("## Not A Real Heading", "content", state, "call_1")

    errors = command.update.get("tool_errors") or []
    assert errors and errors[0]["code"] == "unknown_draft_section"
    for heading in HEADINGS:
        assert heading in errors[0]["message"]
    assert "draft_sections" not in command.update


@pytest.mark.asyncio
async def test_missing_content_is_a_recoverable_missing_arg(client, db_session):
    state, _config, _run, _session = await _seed(client, db_session)

    command = await _write_draft_section_impl("## Vision", "   ", state, "call_1")

    errors = command.update.get("tool_errors") or []
    assert errors and errors[0]["code"] == "missing_required_arg"


@pytest.mark.asyncio
async def test_partial_save_lists_remaining_headings_and_accumulates(client, db_session):
    state, _config, _run, _session = await _seed(client, db_session)

    first = await _write_draft_section_impl("## Vision", "A concrete vision statement.", state, "call_1")
    assert first.update["draft_sections"] == {
        "## Vision": {"content": "## Vision\nA concrete vision statement.", "done": True}
    }
    message = first.update["messages"][0].content
    assert "## Objectives" in message
    assert "## Success Metrics" in message
    assert "write_draft" in message

    # Second call sees the first section via state (as a real turn would carry it forward).
    state["draft_sections"] = first.update["draft_sections"]
    second = await _write_draft_section_impl("## Objectives", "- Ship the thing.", state, "call_2")
    assert second.update["draft_sections"] == {
        "## Vision": {"content": "## Vision\nA concrete vision statement.", "done": True},
        "## Objectives": {"content": "## Objectives\n- Ship the thing.", "done": True},
    }
    assert "## Success Metrics" in second.update["messages"][0].content
    assert "## Vision" not in second.update["messages"][0].content.split("still needed:", 1)[-1]


@pytest.mark.asyncio
async def test_last_section_reports_ready_for_write_draft(client, db_session):
    state, _config, _run, _session = await _seed(client, db_session)
    state["draft_sections"] = {
        "## Vision": {"content": "## Vision\nA concrete vision statement.", "done": True},
        "## Objectives": {"content": "## Objectives\n- Ship the thing.", "done": True},
    }

    command = await _write_draft_section_impl("## Success Metrics", "- Adoption reaches 80%.", state, "call_3")

    assert set(command.update["draft_sections"]) == set(HEADINGS)
    assert all(entry["done"] for entry in command.update["draft_sections"].values())
    assert "call write_draft now" in command.update["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_done_false_keeps_the_heading_open_and_prompts_for_more_batches(client, db_session):
    state, _config, _run, _session = await _seed(client, db_session)

    command = await _write_draft_section_impl(
        "## Success Metrics", "- Metric 1.", state, "call_1", append=False, done=False
    )

    entry = command.update["draft_sections"]["## Success Metrics"]
    assert entry == {"content": "## Success Metrics\n- Metric 1.", "done": False}
    message = command.update["messages"][0].content
    assert "not yet complete" in message
    assert "append=true" in message


@pytest.mark.asyncio
async def test_append_true_concatenates_onto_the_existing_batch(client, db_session):
    state, _config, _run, _session = await _seed(client, db_session)
    state["draft_sections"] = {
        "## Success Metrics": {"content": "## Success Metrics\n- Metric 1.", "done": False}
    }

    command = await _write_draft_section_impl(
        "## Success Metrics", "- Metric 2.", state, "call_2", append=True, done=True
    )

    entry = command.update["draft_sections"]["## Success Metrics"]
    assert entry == {"content": "## Success Metrics\n- Metric 1.\n- Metric 2.", "done": True}


@pytest.mark.asyncio
async def test_append_false_replaces_rather_than_concatenates(client, db_session):
    """append defaults to False: a fresh call for an already-saved heading (e.g. a revision) must
    replace its content, not silently accumulate onto stale text."""
    state, _config, _run, _session = await _seed(client, db_session)
    state["draft_sections"] = {
        "## Vision": {"content": "## Vision\nOld statement.", "done": True}
    }

    command = await _write_draft_section_impl("## Vision", "New statement.", state, "call_2")

    entry = command.update["draft_sections"]["## Vision"]
    assert entry == {"content": "## Vision\nNew statement.", "done": True}


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_assembles_from_completed_sections_ignoring_the_passed_body(
    mock_interrupt, client, db_session
):
    """Once every required heading is saved and done, write_draft's own `body` argument is
    irrelevant -- _resolve_proposed_body assembles the real body from draft_sections instead, so the
    model can pass a short placeholder for the final call rather than regenerating the whole
    document."""
    state, config, run, _session = await _seed(client, db_session)
    state["draft_sections"] = {
        "## Vision": {"content": "## Vision\nA concrete vision statement.", "done": True},
        "## Objectives": {"content": "## Objectives\n- Ship the thing.", "done": True},
        "## Success Metrics": {"content": "## Success Metrics\n- Adoption reaches 80%.", "done": True},
    }

    command = await _write_draft_impl("Vision", "placeholder", state, config, "call_4")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    assert command.update["draft_body"] == (
        "## Vision\nA concrete vision statement.\n\n"
        "## Objectives\n- Ship the thing.\n\n"
        "## Success Metrics\n- Adoption reaches 80%."
    )
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        assert row.input_snapshot["body"] == command.update["draft_body"]


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_falls_back_to_passed_body_when_a_section_is_still_open(
    mock_interrupt, client, db_session
):
    """A section saved with done=False (mid-batch) must not be treated as complete -- write_draft
    falls back to whatever body the model passed instead of assembling a cut-off table."""
    state, config, _run, _session = await _seed(client, db_session)
    state["draft_sections"] = {
        "## Vision": {"content": "## Vision\nA concrete vision statement.", "done": True},
        "## Objectives": {"content": "## Objectives\n- Ship the thing.", "done": True},
        "## Success Metrics": {"content": "## Success Metrics\n- Metric 1.", "done": False},
    }
    complete_body = "\n\n".join(
        [
            "## Vision\nA concrete vision statement.",
            "## Objectives\n- Ship the thing.",
            "## Success Metrics\n- Adoption reaches 80%.",
        ]
    )

    command = await _write_draft_impl("Vision", complete_body, state, config, "call_5")

    assert not (command.update.get("tool_errors") or [])
    assert command.update["draft_body"] == complete_body


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_falls_back_to_passed_body_when_sections_incomplete(mock_interrupt, client, db_session):
    """A partial draft_sections accumulator (model called write_draft early) must not clobber a
    real body the model supplies -- only a COMPLETE, all-done set of sections overrides write_draft's
    body."""
    state, config, _run, _session = await _seed(client, db_session)
    state["draft_sections"] = {"## Vision": {"content": "## Vision\nA concrete vision statement.", "done": True}}
    complete_body = "\n\n".join(
        [
            "## Vision\nA concrete vision statement.",
            "## Objectives\n- Ship the thing.",
            "## Success Metrics\n- Adoption reaches 80%.",
        ]
    )

    command = await _write_draft_impl("Vision", complete_body, state, config, "call_5")

    assert not (command.update.get("tool_errors") or [])
    assert command.update["draft_body"] == complete_body
