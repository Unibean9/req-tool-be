"""Cold-start gate removed.

write_draft is always in the menu; if the model calls it prematurely the tool returns
feedback directly instead of the menu filtering it out.
"""

from app.graphs.agent_tools import get_available_tools


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
