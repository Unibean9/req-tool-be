"""dismiss_question tool (plan 260703 Phase 3, Part A).

Marks a parked open_question node dismissed with an auditable reason and drops it out of the
resurfacing/blocker set (scan_parked_questions). Menu offers it only once a parked open_question exists.
"""

import pytest

from app.graphs.agent_tools import (
    _dismiss_question_impl,
    get_all_analyzer_tools,
    get_available_tools,
)
from app.graphs.decision_graph import scan_parked_questions


@pytest.fixture
def graph_on(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)


@pytest.fixture
def graph_off(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", False)


def _names(state):
    return {t.name for t in get_available_tools(state)}


@pytest.mark.asyncio
async def test_dismiss_requires_reason(graph_on, decision_graph_factory):
    nodes = decision_graph_factory({"id": "Q1", "kind": "open_question", "status": "parked"})
    state = {"messages": [], "decision_nodes": nodes}

    command = await _dismiss_question_impl("Q1", "", state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "missing_required_arg"


@pytest.mark.asyncio
async def test_dismiss_requires_node_id(graph_on, decision_graph_factory):
    nodes = decision_graph_factory({"id": "Q1", "kind": "open_question", "status": "parked"})
    state = {"messages": [], "decision_nodes": nodes}

    command = await _dismiss_question_impl("", "not needed anymore", state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "missing_required_arg"


@pytest.mark.asyncio
async def test_dismiss_rejects_unknown_id(graph_on):
    state = {"messages": [], "decision_nodes": {}}

    command = await _dismiss_question_impl("ghost", "no longer relevant", state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "dismiss_target_invalid"


@pytest.mark.asyncio
async def test_dismiss_rejects_non_open_question(graph_on, decision_graph_factory):
    nodes = decision_graph_factory({"id": "N1", "kind": "decision", "status": "confirmed"})
    state = {"messages": [], "decision_nodes": nodes}

    command = await _dismiss_question_impl("N1", "no longer relevant", state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "dismiss_target_invalid"


@pytest.mark.asyncio
async def test_dismiss_success_sets_status_and_audit(graph_on, decision_graph_factory):
    nodes = decision_graph_factory({"id": "Q1", "kind": "open_question", "status": "parked", "blocks": ["R1"]})
    state = {"messages": [], "turn_count": 3, "decision_nodes": nodes}

    command = await _dismiss_question_impl("Q1", "Out of scope for v1.", state, "tc1")

    node = command.update["decision_nodes"]["Q1"]
    assert node["status"] == "dismissed"
    assert node["dismissal"] == {"reason": "Out of scope for v1.", "turn": 3, "dismissed_by": "agent"}
    assert "Dismissed Q1" in command.update["messages"][0].content
    # The reason is not echoed back as an instruction — only a fixed confirmation string.
    assert "Out of scope for v1." not in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_dismissed_node_drops_out_of_resurfacing(graph_on, decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "status": "confirmed"},
        {"id": "Q1", "kind": "open_question", "status": "parked", "blocks": ["R1"]},
    )
    state = {"messages": [], "decision_nodes": nodes}
    assert scan_parked_questions(nodes)  # sanity: Q1 is actionable before dismissal

    command = await _dismiss_question_impl("Q1", "Answered elsewhere.", state, "tc1")

    assert scan_parked_questions(command.update["decision_nodes"]) == []


@pytest.mark.asyncio
async def test_dismiss_noop_when_flag_off(graph_off, decision_graph_factory):
    nodes = decision_graph_factory({"id": "Q1", "kind": "open_question", "status": "parked"})
    state = {"messages": [], "decision_nodes": nodes}

    command = await _dismiss_question_impl("Q1", "reason", state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"


def test_dismiss_question_in_registry():
    names = {tool.name for tool in get_all_analyzer_tools()}
    assert "dismiss_question" in names


def test_menu_hides_dismiss_without_parked_open_question(graph_on, decision_graph_factory):
    nodes = decision_graph_factory({"id": "N1", "kind": "decision", "status": "confirmed"})
    state = {"messages": [], "user_confirmed": True, "decision_nodes": nodes}
    assert "dismiss_question" not in _names(state)


def test_menu_offers_dismiss_when_parked_open_question_exists(graph_on, decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "status": "confirmed"},
        {"id": "Q1", "kind": "open_question", "status": "parked", "blocks": ["R1"]},
    )
    state = {"messages": [], "user_confirmed": True, "decision_nodes": nodes}
    assert "dismiss_question" in _names(state)
