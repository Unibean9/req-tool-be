"""Cold-start no longer has a menu gate.

write_draft always appears in the menu; tool returns recoverable feedback itself if model draft ngay when
cold-start is still thin.
"""

import pytest

from app.graphs.agent_tools import _write_draft_impl, get_available_tools


def _base_state(**overrides):
    state = {
        "messages": [],
        "user_confirmed": True,
    }
    state.update(overrides)
    return state


def _names(state):
    return {t.name for t in get_available_tools(state)}


def test_write_draft_available_when_no_nodes_and_no_elicit():
    state = _base_state(decision_nodes={}, session_elicit_count=0)

    assert "write_draft" in _names(state)


def test_write_draft_available_after_elicit_runs():
    state = _base_state(decision_nodes={}, session_elicit_count=1)

    assert "write_draft" in _names(state)


def test_write_draft_available_when_nodes_exist():
    state = _base_state(decision_nodes={"N1": {"id": "N1"}}, session_elicit_count=0)

    assert "write_draft" in _names(state)


@pytest.mark.asyncio
async def test_write_draft_self_rejects_thin_cold_start():
    # user_confirmed=None: confirm_intent was never called — a true cold-start with no context gathered.
    state = _base_state(decision_nodes={}, session_elicit_count=0, user_confirmed=None)

    command = await _write_draft_impl("Draft", "Content draft qua som", state, {"configurable": {}}, "tc1")

    assert command.update["tool_errors"][0]["code"] == "cold_start_requires_elicitation"
    assert "elicit" in command.update["messages"][0].content
