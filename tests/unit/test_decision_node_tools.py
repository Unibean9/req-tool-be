"""Decision-graph tool wrappers: flag gating + state write via Command.update.

The pure graph functions are covered in test_decision_node_crud/supersede; these assert the tool
layer wires them into decision_nodes state and that DECISION_GRAPH_ENABLED gates every write.
"""

import pytest

from app.graphs.agent_tools import (
    _create_decision_node_impl,
    _supersede_decision_node_impl,
    _update_decision_node_impl,
    get_all_analyzer_tools,
    get_available_tools,
)


@pytest.fixture
def graph_on(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)


@pytest.fixture
def graph_off(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", False)


def _names(state):
    return {t.name for t in get_available_tools(state)}


@pytest.mark.asyncio
async def test_create_writes_node_to_state(graph_on):
    state = {"messages": [], "user_confirmed": True, "turn_count": 2, "decision_nodes": {}}

    command = await _create_decision_node_impl("decision", "v1 = operations-first", [], "5_whys", state, "tc1")

    nodes = command.update["decision_nodes"]
    assert len(nodes) == 1
    node = next(iter(nodes.values()))
    assert node["kind"] == "decision"
    assert node["status"] == "proposed"
    assert node["origin"]["technique"] == "5_whys"
    assert node["origin"]["turn"] == 2


@pytest.mark.asyncio
async def test_create_records_section_and_fields(graph_on):
    state = {"messages": [], "user_confirmed": True, "turn_count": 2, "decision_nodes": {}}

    command = await _create_decision_node_impl(
        "objective",
        "Measure successful group scheduling rate.",
        [],
        "moscow",
        state,
        "tc1",
        status="needs_confirmation",
        section="## Success Metrics",
        fields={
            "goal": "Schedule study groups",
            "metric": "Successful scheduling rate",
            "target": "80%",
        },
    )

    node = next(iter(command.update["decision_nodes"].values()))
    assert node["section"] == "## Success Metrics"
    assert node["fields"]["metric"] == "Successful scheduling rate"


@pytest.mark.asyncio
async def test_create_records_section_finding_same_turn_without_blocking(graph_on):
    """A weasel-worded write still succeeds (node written) AND records a section finding this turn."""
    state = {
        "messages": [],
        "user_confirmed": True,
        "turn_count": 1,
        "artifact_type": "brd",
        "decision_nodes": {},
        "section_findings": {},
    }

    command = await _create_decision_node_impl(
        "fact", "The system is fast and flexible.", [], None, state, "tc1", section="## Business Rules"
    )

    # Write is not blocked: the node is present in state.
    assert len(command.update["decision_nodes"]) == 1
    findings = command.update["section_findings"]["## Business Rules"]
    assert any(f["severity"] == "violation" for f in findings)  # business rule missing condition/outcome


@pytest.mark.asyncio
async def test_clean_section_write_stores_empty_findings(graph_on):
    state = {
        "messages": [],
        "user_confirmed": True,
        "turn_count": 1,
        "artifact_type": "brd",
        "decision_nodes": {},
        "section_findings": {"## Stakeholders": [{"severity": "violation", "message": "old"}]},
    }

    command = await _create_decision_node_impl(
        "fact", "The product owner signs off scope.", [], None, state, "tc1", section="## Stakeholders"
    )

    # Re-validated clean -> [] (not absent) so the merge reducer clears the prior defect.
    assert command.update["section_findings"]["## Stakeholders"] == []


@pytest.mark.asyncio
async def test_sectionless_write_records_no_findings(graph_on):
    state = {"messages": [], "user_confirmed": True, "turn_count": 1, "artifact_type": "brd", "decision_nodes": {}}

    command = await _create_decision_node_impl("decision", "v1 = operations-first", [], None, state, "tc1")

    assert "section_findings" not in command.update


@pytest.mark.asyncio
async def test_create_noop_when_flag_off(graph_off):
    state = {"messages": [], "decision_nodes": {}}

    command = await _create_decision_node_impl("decision", "x", [], None, state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"


@pytest.mark.asyncio
async def test_create_rejects_invalid_kind(graph_on):
    state = {"messages": [], "decision_nodes": {}}

    command = await _create_decision_node_impl("bogus", "x", [], None, state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"


@pytest.mark.asyncio
async def test_create_rejects_duplicate_node_id(graph_on, decision_graph_factory):
    state = {"messages": [], "decision_nodes": decision_graph_factory({"id": "N1", "status": "confirmed"})}

    command = await _create_decision_node_impl("decision", "ghi de", [], None, state, "tc1", node_id="N1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"


@pytest.mark.asyncio
async def test_create_can_record_parked_question_blocks(graph_on):
    state = {"messages": [], "turn_count": 4, "decision_nodes": {}}

    command = await _create_decision_node_impl(
        "open_question",
        "Define multi-channel points accrual",
        [],
        None,
        state,
        "tc1",
        node_id="Q8",
        status="parked",
        blocks=["R1", "S1"],
    )

    node = command.update["decision_nodes"]["Q8"]
    assert node["status"] == "parked"
    assert node["blocks"] == ["R1", "S1"]


@pytest.mark.asyncio
async def test_update_changes_status_in_state(graph_on, decision_graph_factory):
    state = {"messages": [], "decision_nodes": decision_graph_factory({"id": "N1", "status": "proposed"})}

    command = await _update_decision_node_impl("N1", "confirmed", None, state, "tc1")

    assert command.update["decision_nodes"]["N1"]["status"] == "confirmed"


@pytest.mark.asyncio
async def test_update_records_section_and_fields(graph_on, decision_graph_factory):
    state = {"messages": [], "decision_nodes": decision_graph_factory({"id": "N1", "status": "confirmed"})}

    command = await _update_decision_node_impl(
        "N1",
        None,
        None,
        state,
        "tc1",
        section="## Success Metrics",
        fields={
            "goal": "Schedule study groups",
            "metric": "Successful scheduling rate",
            "target": "80%",
        },
    )

    node = command.update["decision_nodes"]["N1"]
    assert node["section"] == "## Success Metrics"
    assert node["fields"]["target"] == "80%"


@pytest.mark.asyncio
async def test_update_rejects_invalid_status(graph_on, decision_graph_factory):
    state = {"messages": [], "decision_nodes": decision_graph_factory({"id": "N1", "status": "proposed"})}

    command = await _update_decision_node_impl("N1", "done", None, state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"


@pytest.mark.asyncio
async def test_update_rejects_superseded_node(graph_on, decision_graph_factory):
    state = {"messages": [], "decision_nodes": decision_graph_factory({"id": "N1", "status": "superseded"})}

    command = await _update_decision_node_impl("N1", "confirmed", None, state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"


@pytest.mark.asyncio
async def test_supersede_ripples_to_dependents(graph_on, decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
    )
    state = {"messages": [], "decision_nodes": nodes}

    command = await _supersede_decision_node_impl("N1", "dao huong", "abandon", state, "tc1")

    updated = command.update["decision_nodes"]
    assert updated["N1"]["status"] == "superseded"
    assert updated["N3"]["status"] == "parked"


@pytest.mark.asyncio
async def test_supersede_missing_node_returns_error(graph_on):
    state = {"messages": [], "decision_nodes": {}}

    command = await _supersede_decision_node_impl("ghost", "x", None, state, "tc1")

    assert "decision_nodes" not in command.update
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"


def test_menu_hides_decision_tools_when_flag_off(graph_off):
    names = _names({"messages": [], "user_confirmed": True, "decision_nodes": {}})
    assert "create_decision_node" not in names


def test_menu_offers_create_when_flag_on(graph_on):
    names = _names({"messages": [], "user_confirmed": True, "decision_nodes": {}})
    assert "create_decision_node" in names
    # update/supersede stay hidden until a node exists.
    assert "supersede_decision_node" not in names


def test_menu_offers_supersede_once_nodes_exist(graph_on, decision_graph_factory):
    state = {
        "messages": [],
        "user_confirmed": True,
        "session_elicit_count": 1,
        "decision_nodes": decision_graph_factory({"id": "N1", "status": "confirmed"}),
    }
    names = _names(state)
    assert {"create_decision_node", "update_decision_node", "supersede_decision_node"} <= names


def test_cross_artifact_tools_in_registry():
    names = {tool.name for tool in get_all_analyzer_tools()}

    assert {"run_impact_analysis", "read_artifact_graph", "create_artifact_link"} <= names
