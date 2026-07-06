"""Tool-loop scenarios end-to-end through the HTTP driver (flag on).

Proves the shim runs a full conversation: analyze emits tool_calls, the ToolNode dispatches, the
HITL interrupt/resume round-trips, an approved write_draft becomes an artifact, and an exhausted
tool-brain (no tool picked) ends the turn cleanly. The canonical scenario lane covers broad
user journeys; this file keeps focused tool-loop edge cases.
"""

import uuid

import pytest

from tests.integration.scenarios.driver import Scenario, ScenarioDriver
from tests.integration.scenarios.library import _GOAL_BODY
from tests.integration.scenarios.scripted_llm import ScriptedLLM, tool_select

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# T5 — HITL round-trip + artifact creation through the tool-loop
# ---------------------------------------------------------------------------

async def test_tool_loop_ask_then_draft_approve(client, scenario_env, scenario_project):
    """ask_user → confirm_intent → write_draft approval → completed."""
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        tool_select("ask_user", message="Who is the primary user?"),
        tool_select("confirm_intent",
                    summary="Set measurable goals for the student group scheduling tool."),
        tool_select("write_draft", title="Goal: orchestration study scheduling", body=_GOAL_BODY),
    ])
    scenario = Scenario(
        name="tool-loop-ask-draft-approve",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "I want to set goals for the study scheduling product."},
            {"type": "send", "content": "Mainly university students studying in groups."},
            {"type": "send", "content": "Dung roi, tiep tuc giup toi."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    recorder = await driver.run()

    assert recorder.summary["final_status"] == "completed"
    artifacts = await driver.executed_artifacts()
    assert len(artifacts) >= 1
    assert artifacts[0]["body"] == _GOAL_BODY

    # No duplicate agent question on the resume path (R1 idempotency holds through the tool path).
    final_msgs = recorder.steps[-1]["snapshot"]["messages"]
    questions = [m for m in final_msgs if (m.get("payload") or {}).get("kind") == "question"]
    assert len(questions) == 1


async def test_tool_loop_reject_draft(client, scenario_env, scenario_project):
    """Rejecting the proposed write_draft creates no artifact; the turn still completes."""
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        tool_select("confirm_intent",
                    summary="Set goals for the student group scheduling tool."),
        tool_select("write_draft", title="Goal (draft)", body=_GOAL_BODY),
    ])
    scenario = Scenario(
        name="tool-loop-reject-draft",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "Set goals for the group study scheduling product."},
            {"type": "send", "content": "Dung roi, tiep tuc giup toi."},
            {"type": "reject_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 0},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    recorder = await driver.run()

    assert recorder.summary["final_status"] == "completed"
    assert len(await driver.executed_artifacts()) == 0


async def test_tool_loop_composite_two_decision_nodes_both_survive(client, scenario_env, scenario_project):
    """Two create_decision_node calls in ONE turn must both persist (merge reducer, not last-writer-wins).

    Both tools receive the same pre-turn snapshot via InjectedState and each return the full graph; a
    plain-replace channel would drop the first node. The merge reducer keeps both.
    """
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        {
            "tools": [
                {"name": "create_decision_node",
                 "args": {"kind": "decision", "statement": "v1 = operations-first", "node_id": "N1"}},
                {"name": "create_decision_node",
                 "args": {"kind": "risk", "statement": "staff avoid entering recipes", "node_id": "N2"}},
            ],
        },
        tool_select("confirm_intent",
                    summary="Dieu phoi study scheduling cho sinh vien."),
        tool_select("write_draft", title="Goal", body=_GOAL_BODY),
    ])
    scenario = Scenario(
        name="tool-loop-composite-decision-nodes",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "I want to build a coffee shop management app."},
            {"type": "send", "content": "Dung roi, tiep tuc."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    await driver.run()

    nodes = await scenario_env.get_checkpoint_field(driver.session_id, "decision_nodes")
    assert "N1" in (nodes or {}) and "N2" in (nodes or {}), (
        f"merge reducer dropped a same-turn node: got {sorted((nodes or {}).keys())}"
    )


async def test_tool_loop_two_elicits_one_turn_accumulate(client, scenario_env, scenario_project):
    """Two elicit calls in ONE turn must not crash and must accumulate session_elicit_count to 2.

    Without an additive reducer, two concurrent session_elicit_count writes raise
    INVALID_CONCURRENT_GRAPH_UPDATE — the bug that killed live multi-technique cold-start turns.
    """
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        {
            "tools": [
                {"name": "elicit", "args": {"technique": "comparable_products", "seed": "app coffee shop"}},
                {"name": "elicit", "args": {"technique": "5_whys", "seed": "inventory shortage"}},
            ],
        },
        tool_select("confirm_intent", summary="App quantification cho coffee shop."),
        tool_select("write_draft", title="Goal", body=_GOAL_BODY),
    ])
    scenario = Scenario(
        name="tool-loop-two-elicits",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "I want to build a coffee shop management app."},
            {"type": "send", "content": "Dung roi, tiep tuc."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    await driver.run()

    count = await scenario_env.get_checkpoint_field(driver.session_id, "session_elicit_count")
    assert count == 2, f"two elicits in one turn should accumulate to 2, got {count}"


async def test_tool_loop_composite_two_note_tools(client, scenario_env, scenario_project):
    """Composite dispatch: brain returns [explore_note, critique_note] in one turn.

    Proves that a single analyze_node cycle emits an AIMessage with 2 tool_calls and
    the ToolNode dispatches both — resulting in 2 ToolMessages in the checkpoint before
    the next turn. The session must stay ACTIVE after both notes execute (no interrupt).
    """
    from langchain_core.messages import AIMessage, ToolMessage

    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        # Turn 1: composite — two non-interrupt note tools in the same turn.
        {
            "tools": [
                {"name": "explore_note", "args": {"content": "Primary users are students in groups of 4-6."}},
                {"name": "critique_note", "args": {"content": "Need measurement: group session attendance rate."}},
            ],
        },
        # Turn 2: confirm intent after notes.
        tool_select("confirm_intent",
                    summary="Dieu phoi study scheduling cho sinh vien, do bang ti le tham gia."),
        # Turn 3: draft after intent confirmed.
        tool_select("write_draft", title="Goal: orchestration study scheduling", body=_GOAL_BODY),
    ])
    scenario = Scenario(
        name="tool-loop-composite-notes",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "I want to set goals for the study scheduling product."},
            {"type": "send", "content": "Dung roi, tiep tuc."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    recorder = await driver.run()

    assert recorder.summary["final_status"] == "completed"
    assert len(await driver.executed_artifacts()) >= 1

    # Verify composite dispatch in checkpoint: find the AIMessage that carried 2 tool_calls.
    raw_msgs = await scenario_env.get_checkpoint_field(driver.session_id, "messages")
    ai_msgs_with_two_calls = [
        m for m in (raw_msgs or [])
        if isinstance(m, AIMessage) and len(getattr(m, "tool_calls", [])) == 2
    ]
    assert ai_msgs_with_two_calls, "No AIMessage with 2 tool_calls found — composite dispatch did not fire"

    composite_ai = ai_msgs_with_two_calls[0]
    dispatched_names = [tc["name"] for tc in composite_ai.tool_calls]
    assert set(dispatched_names) == {"explore_note", "critique_note"}, (
        f"unexpected composite tools: {dispatched_names}"
    )

    # Both ToolMessages must follow the composite AIMessage.
    call_ids = {tc["id"] for tc in composite_ai.tool_calls}
    tool_msgs = [
        m for m in (raw_msgs or [])
        if isinstance(m, ToolMessage) and m.tool_call_id in call_ids
    ]
    assert len(tool_msgs) == 2, (
        f"expected 2 ToolMessages for composite dispatch, got {len(tool_msgs)}"
    )
