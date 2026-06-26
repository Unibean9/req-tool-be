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
        # Phần 1 — cold-start: explore before drafting, create nodes with technique provenance.
        tool_select("elicit_tool", technique="5_whys", seed="hụt nguyên liệu"),
        tool_select("create_decision_node", node_id="N1", kind="decision",
                    statement="v1 = vận hành-first (định lượng + trừ kho)", technique="5_whys"),
        tool_select("create_decision_node", node_id="N2", kind="risk",
                    statement="nhân viên ngại nhập công thức → vẫn pha tay", technique="reverse"),
        # Phần 2 — co-creation: dependents build on N1.
        tool_select("create_decision_node", node_id="N3", kind="objective",
                    statement="Giảm thời gian pha chế", depends_on=["N1"]),
        tool_select("create_decision_node", node_id="N4", kind="scope",
                    statement="Màn hình bếp realtime", depends_on=["N1"]),
        tool_select("create_decision_node", node_id="N5", kind="assumption",
                    statement="Nhân viên dùng tablet tại quầy", depends_on=["N1"]),
        tool_select("update_decision_node", node_id="N1", status="confirmed"),
        tool_select("ask_user", message="Mình chốt hướng vận hành-first nhé — đúng pain của bạn chứ?"),
        # Phần 3 — reversal: pushback BEFORE mutating, then supersede with abandon.
        tool_select("respond",
                    message="Đảo qua loyalty-first sẽ treo N3/N4/N5 (giảm pha chế, màn bếp, tablet). "
                            "Đó là đổi hướng gốc, không phải chỉnh nhỏ — chắc chứ?",
                    mode="critique"),
        tool_select("supersede_decision_node", node_id="N1",
                    statement="v1 = loyalty-first (giữ chân khách)",
                    new_statement="v1 = loyalty-first (giữ chân khách)", cascade_mode="abandon"),
    ])


async def _run_full_sequence(client, scenario_env, scenario_project):
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    scenario = Scenario(
        name="golden-p1-p3-gate",
        artifact_type="intent",
        llm=_full_sequence_brain(),
        actions=[
            {"type": "send", "content": "Tôi muốn làm app quản lý quán cà phê."},
            {"type": "send", "content": "Chủ yếu không kiểm soát được nguyên liệu, hay bị hụt."},
            {"type": "send", "content": "Đổi qua giữ chân khách (loyalty-first) đi."},
            {"type": "send", "content": "Chắc chắn, đổi giúp tôi."},
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
    assert "vận hành-first" not in active_body
    for parked in ("Giảm thời gian pha chế", "Màn hình bếp realtime", "Nhân viên dùng tablet"):
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
        and "vận hành-first" not in render_view(nodes, "brd").split("Parked")[0]
    )
    assert go, "DECISION GATE: NO-GO — debug R1/R2 (supersede/cascade/render) trước khi mở rộng"


# ---------------------------------------------------------------------------
# Named Phần 1-3 behaviors (golden) — read off the same driven sequence.
# ---------------------------------------------------------------------------

async def test_cold_start_explores_before_drafting(client, scenario_env, scenario_project, graph_flag_on):
    """Phần 1: the agent explores and records nodes; it never proposes a draft on the cold-start turn."""
    driver, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert await driver.executed_artifacts() == []  # no write_draft was approved → none executed
    assert nodes, "exploration produced no nodes"


async def test_nodes_created_with_technique_provenance(client, scenario_env, scenario_project, graph_flag_on):
    """Phần 1: each agent-created node records who/what produced it (origin.by + technique)."""
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert nodes["N1"]["origin"]["by"] == "agent"
    assert nodes["N1"]["origin"]["technique"] == "5_whys"
    assert nodes["N2"]["origin"]["technique"] == "reverse"


async def test_decision_reversal_supersedes_not_deletes(client, scenario_env, scenario_project, graph_flag_on):
    """Phần 3: reversing keeps history — old node stays as superseded, nothing is removed."""
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert nodes["N1"]["status"] == "superseded"
    assert {"N1", "N2", "N3", "N4", "N5"} <= set(nodes)


async def test_decision_reversal_uses_abandon_mode(client, scenario_env, scenario_project, graph_flag_on):
    """Phần 3: a root-direction reversal parks dependents (abandon), not needs_confirmation."""
    _, nodes = await _run_full_sequence(client, scenario_env, scenario_project)
    assert [nodes[n]["status"] for n in ("N3", "N4", "N5")] == ["parked"] * 3


async def test_agent_pushback_and_reads_dependents(client, scenario_env, scenario_project, graph_flag_on):
    """Phần 3: the agent warns about the branch impact before superseding, and the supersede surfaces
    the dependents it rippled to."""
    driver, _ = await _run_full_sequence(client, scenario_env, scenario_project)
    # Pushback assessment is surfaced to the user via the message API payload (respond tool).
    import json

    blob = json.dumps(await driver._list_messages(), ensure_ascii=False)
    assert "đổi hướng gốc" in blob, "no pushback before the reversal"
    # The supersede records which dependents it rippled to (read-before-write) in its ToolMessage.
    raw = await scenario_env.get_checkpoint_field(driver.session_id, "messages")
    texts = [getattr(m, "content", "") or "" for m in (raw or [])]
    assert any("dependents=" in t and "N3" in t for t in texts), "supersede did not surface dependents"


# ---------------------------------------------------------------------------
# Phần 2 — co-creation: confirmation updates in place; MoSCoW parks out-of-scope.
# ---------------------------------------------------------------------------

async def test_objective_confirmation_updates_node_not_creates_new(
    client, scenario_env, scenario_project, graph_flag_on
):
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    llm = ScriptedLLM(tool_brain=[
        tool_select("create_decision_node", node_id="N3", kind="objective", statement="Tăng doanh thu Y%"),
        tool_select("update_decision_node", node_id="N3", status="confirmed", statement="Tăng doanh thu 10%"),
    ])
    scenario = Scenario(
        name="golden-p2-confirm",
        artifact_type="intent",
        llm=llm,
        actions=[{"type": "send", "content": "Mục tiêu chắc tầm tăng doanh thu 10%."}],
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
        tool_select("elicit_tool", technique="moscow", seed="phạm vi v1"),
        tool_select("create_decision_node", node_id="M1", kind="scope", statement="Tính khung giờ chung (Must)"),
        tool_select("create_decision_node", node_id="O1", kind="open_question",
                    statement="Tích hợp thanh toán (Out v1)"),
        tool_select("update_decision_node", node_id="O1", status="parked"),
    ])
    scenario = Scenario(
        name="golden-p2-moscow",
        artifact_type="intent",
        llm=llm,
        actions=[{"type": "send", "content": "Phân loại MoSCoW giúp tôi phạm vi v1."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    await driver.run()
    nodes = await scenario_env.get_checkpoint_field(driver.session_id, "decision_nodes") or {}

    assert nodes["O1"]["status"] == "parked"
    assert nodes["M1"]["status"] != "parked"
