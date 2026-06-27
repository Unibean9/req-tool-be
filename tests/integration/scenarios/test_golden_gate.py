"""Decision gate — golden P1-P3 end-to-end on vision_objectives.

Runs cold-start → co-creation → reversal through the real graph + HTTP driver + checkpointer with
DECISION_GRAPH_ENABLED on, then asserts the decision_nodes transition and the rendered view.
"""

import uuid

import pytest

from app.graphs.decision_graph import render_view
from tests.integration.scenarios.driver import Scenario, ScenarioDriver
from tests.integration.scenarios.scripted_llm import ScriptedLLM, tool_select

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def graph_flag_on(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)


def _full_sequence_brain() -> ScriptedLLM:
    """Cold-start exploration → create graph → pushback → supersede with abandon cascade."""
    return ScriptedLLM(tool_brain=[
        # Part 1 — cold-start: explore before drafting, create nodes with technique provenance.
        tool_select("elicit_tool", technique="5_whys", seed="inventory shortage"),
        tool_select("create_decision_node", node_id="N1", kind="decision",
                    statement="v1 = operations-first (quantification + stock deduction)", technique="5_whys"),
        tool_select("create_decision_node", node_id="N2", kind="risk",
                    statement="staff avoid entering recipes → still mix manually", technique="reverse"),
        # Part 2 — co-creation: dependents build on N1.
        tool_select("create_decision_node", node_id="N3", kind="objective",
                    statement="Reduce preparation time", depends_on=["N1"]),
        tool_select("create_decision_node", node_id="N4", kind="scope",
                    statement="Realtime kitchen screen", depends_on=["N1"]),
        tool_select("create_decision_node", node_id="N5", kind="assumption",
                    statement="Staff use tablets at the counter", depends_on=["N1"]),
        tool_select("update_decision_node", node_id="N1", status="confirmed"),
        tool_select("ask_user", message="Let's confirm operations-first - does that match your pain?"),
        # Part 3 — reversal: pushback BEFORE mutating, then supersede with abandon.
        tool_select("respond",
                    message="Switching to loyalty-first will park N3/N4/N5 (prep reduction, kitchen screen, tablet). "
                            "That is a root direction change, not a small edit - are you sure?",
                    mode="critique"),
        tool_select("supersede_decision_node", node_id="N1",
                    statement="v1 = loyalty-first (customer retention)",
                    new_statement="v1 = loyalty-first (customer retention)", cascade_mode="abandon"),
    ])


async def _run_full_sequence(client, scenario_env, scenario_project):
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    scenario = Scenario(
        name="golden-p1-p3-gate",
        artifact_type="intent",
        llm=_full_sequence_brain(),
        actions=[
            {"type": "send", "content": "I want to build a coffee shop management app."},
            {"type": "send", "content": "Mainly cannot control inventory and often runs short."},
            {"type": "send", "content": "Doi qua customer retention (loyalty-first) di."},
            {"type": "send", "content": "Yes, change it for me."},
        ],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    await driver.run()
    nodes = await scenario_env.get_checkpoint_field(driver.session_id, "decision_nodes")
    return driver, nodes or {}


async def test_full_p1_to_p3_sequence_on_vision_objectives(
    client, scenario_env, scenario_project, graph_flag_on
):
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)

    assert nodes, "decision_nodes empty — tools did not mutate state (flag off or not dispatched)"
    assert nodes["N1"]["status"] == "superseded"
    new_id = nodes["N1"]["superseded_by"]
    assert new_id and nodes[new_id]["supersedes"] == "N1"
    assert "loyalty" in nodes[new_id]["statement"].lower()
    # Abandon cascade: the operational branch is parked, NOT needs_confirmation, NOT deleted.
    assert [nodes[n]["status"] for n in ("N3", "N4", "N5")] == ["parked"] * 3
    assert {"N1", "N2", "N3", "N4", "N5", new_id} <= set(nodes)


async def test_render_after_p3_hides_abandoned_branch(
    client, scenario_env, scenario_project, graph_flag_on
):
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    out = render_view(nodes, "brd")
    active_body = out.split("Parked")[0]

    # Superseded root hidden entirely; the abandoned branch is folded into Parked, not the active body.
    assert "operations-first" not in active_body
    for parked in ("Reduce preparation time", "Realtime kitchen screen", "Staff use tablets"):
        assert parked not in active_body
    assert "Parked" in out
    # The replacement node lives in the graph (supersedes N1) but is `proposed`, so per the render
    # contract it stays out of the active view until confirmed — a fresh proposal, not yet agreed.
    new_id = nodes["N1"]["superseded_by"]
    assert nodes[new_id]["status"] == "proposed"
    assert "loyalty-first" not in active_body


async def test_gate_go_condition(client, scenario_env, scenario_project, graph_flag_on):
    """GO when the full sequence + render both hold — the keystone composes through the real graph."""
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    new_id = nodes["N1"]["superseded_by"]

    go = (
        nodes["N1"]["status"] == "superseded"
        and nodes[new_id]["supersedes"] == "N1"
        and all(nodes[n]["status"] == "parked" for n in ("N3", "N4", "N5"))
        and "operations-first" not in render_view(nodes, "brd").split("Parked")[0]
    )
    assert go, "DECISION GATE: NO-GO — debug R1/R2 (supersede/cascade/render) truoc when mo rong"


# ---------------------------------------------------------------------------
# Named Part 1-3 behaviors (golden) — read off the same driven sequence.
# ---------------------------------------------------------------------------

async def test_cold_start_explores_before_drafting(client, scenario_env, scenario_project, graph_flag_on):
    """Part 1: the agent explores and records nodes; it never proposes a draft on the cold-start turn."""
    driver, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert await driver.executed_artifacts() == []  # no write_draft was approved → none executed
    assert nodes, "exploration produced no nodes"


async def test_nodes_created_with_technique_provenance(client, scenario_env, scenario_project, graph_flag_on):
    """Part 1: each agent-created node records who/what produced it (origin.by + technique)."""
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert nodes["N1"]["origin"]["by"] == "agent"
    assert nodes["N1"]["origin"]["technique"] == "5_whys"
    assert nodes["N2"]["origin"]["technique"] == "reverse"


async def test_decision_reversal_supersedes_not_deletes(client, scenario_env, scenario_project, graph_flag_on):
    """Part 3: reversing keeps history — old node stays as superseded, nothing is removed."""
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert nodes["N1"]["status"] == "superseded"
    assert {"N1", "N2", "N3", "N4", "N5"} <= set(nodes)


async def test_decision_reversal_uses_abandon_mode(client, scenario_env, scenario_project, graph_flag_on):
    """Part 3: a root-direction reversal parks dependents (abandon), not needs_confirmation."""
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert [nodes[n]["status"] for n in ("N3", "N4", "N5")] == ["parked"] * 3


async def test_agent_pushback_and_reads_dependents(client, scenario_env, scenario_project, graph_flag_on):
    """Part 3: the agent warns about the branch impact before superseding, and the supersede surfaces
    the dependents it rippled to."""
    driver, _ = await _run_full_sequence(client, scenario_env, scenario_project)
    # Pushback assessment is surfaced to the user via the message API payload (respond tool).
    import json

    blob = json.dumps(await driver._list_messages(), ensure_ascii=False)
    assert "doi huong goc" in blob, "no pushback before the reversal"
    # The supersede records which dependents it rippled to (read-before-write) in its ToolMessage.
    raw = await scenario_env.get_checkpoint_field(driver.session_id, "messages")
    texts = [getattr(m, "content", "") or "" for m in (raw or [])]
    assert any("dependents=" in t and "N3" in t for t in texts), "supersede did not surface dependents"


# ---------------------------------------------------------------------------
# Part 2 — co-creation: confirmation updates in place; MoSCoW parks out-of-scope.
# ---------------------------------------------------------------------------

async def test_objective_confirmation_updates_node_not_creates_new(
    client, scenario_env, scenario_project, graph_flag_on
):
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    llm = ScriptedLLM(tool_brain=[
        tool_select("create_decision_node", node_id="N3", kind="objective", statement="Tang doanh thu Y%"),
        tool_select("update_decision_node", node_id="N3", status="confirmed", statement="Tang doanh thu 10%"),
    ])
    scenario = Scenario(
        name="golden-p2-confirm",
        artifact_type="intent",
        llm=llm,
        actions=[{"type": "send", "content": "Goal is probably to increase revenue by 10%."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    await driver.run()
    nodes = await scenario_env.get_checkpoint_field(driver.session_id, "decision_nodes") or {}

    assert len(nodes) == 1
    assert nodes["N3"]["status"] == "confirmed"
    assert "10%" in nodes["N3"]["statement"]
    assert not any(n.get("supersedes") == "N3" for n in nodes.values())


async def test_moscow_pushes_out_of_scope_to_parked(client, scenario_env, scenario_project, graph_flag_on):
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    llm = ScriptedLLM(tool_brain=[
        tool_select("elicit_tool", technique="moscow", seed="pham vi v1"),
        tool_select("create_decision_node", node_id="M1", kind="scope", statement="Tinh khung gio chung (Must)"),
        tool_select("create_decision_node", node_id="O1", kind="open_question",
                    statement="Tich hop thanh toan (Out v1)"),
        tool_select("update_decision_node", node_id="O1", status="parked"),
    ])
    scenario = Scenario(
        name="golden-p2-moscow",
        artifact_type="intent",
        llm=llm,
        actions=[{"type": "send", "content": "Phan loai MoSCoW giup toi pham vi v1."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    await driver.run()
    nodes = await scenario_env.get_checkpoint_field(driver.session_id, "decision_nodes") or {}

    assert nodes["O1"]["status"] == "parked"
    assert nodes["M1"]["status"] != "parked"


# ---------------------------------------------------------------------------
# R7 — Multi-role within a single user turn (harness capability, ScriptedLLM)
# ---------------------------------------------------------------------------

async def test_multi_role_tools_within_single_user_turn(
    client, scenario_env, scenario_project, graph_flag_on
):
    """R7: harness must support ≥4 distinct roles in one user-turn (explore→write→update→critique→ask).

    Uses ScriptedLLM to drive the sequence deterministically; the assertion is about
    harness capability (can all these tools dispatch between two user messages?), not
    model judgment.
    """
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    multi_role_brain = ScriptedLLM(tool_brain=[
        # All 5 calls happen BEFORE the second user message (single drain).
        # respond is interrupt-bearing so it cannot precede ask_user in the same drain;
        # use critique_note (non-interrupt) for the critique role instead.
        tool_select("elicit_tool", technique="5_whys", seed="inventory shortage"),       # khai pha
        tool_select("create_decision_node", node_id="N1", kind="decision",             # ghi
                    statement="operations-first", technique="5_whys"),
        tool_select("update_decision_node", node_id="N1", status="confirmed"),         # cap nhat
        tool_select("critique_note", content="Risk: staff avoid data entry."),    # critique
        tool_select("ask_user", message="Do you agree with the operations-first direction?"),    # orchestration
    ])
    scenario = Scenario(
        name="r7-multi-role",
        artifact_type="intent",
        llm=multi_role_brain,
        actions=[{"type": "send", "content": "I want to build a coffee shop management app."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    await driver.run()

    raw_messages = await scenario_env.get_checkpoint_field(driver.session_id, "messages")
    tool_names_used: set[str] = set()
    for msg in raw_messages or []:
        for tc in getattr(msg, "tool_calls", []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tool_names_used.add(name)

    assert "elicit_tool" in tool_names_used, "Explore role not exercised within single turn"
    assert "create_decision_node" in tool_names_used, "Write role not exercised"
    assert "update_decision_node" in tool_names_used, "Update role not exercised"
    assert "critique_note" in tool_names_used, "Critique role not exercised"
    assert "ask_user" in tool_names_used, "Orchestrate role not exercised"
