import pytest

from app.config import settings
from app.graphs.agent_tools import DIAGNOSIS_JUDGE_CALLS_MAX
from app.graphs.nodes import orchestrator_node


@pytest.mark.asyncio
async def test_orchestrator_writes_diagnosis_every_turn(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "brd"}, {})

    assert update["thinking_mode"] in {"structuring", "challenging", "synthesizing", "risk_probing"}
    assert update["diagnosis_signal"]["risk_level"] in {"low", "high"}


@pytest.mark.asyncio
async def test_orchestrator_classifies_low_coverage_failed_critique_as_higher_risk(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    risky_state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "section_coverage": {"scope_capabilities": "missing", "vision_objectives": "missing"},
        "quality_report": {"quality_gate_result": "fail"},
        "draft_body": "",
    }
    safe_state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "section_coverage": {"scope_capabilities": "filled", "vision_objectives": "filled"},
        "quality_report": None,
        "draft_body": "a fully drafted section body",
    }

    risky_update = await orchestrator_node(risky_state, {})
    safe_update = await orchestrator_node(safe_state, {})

    assert risky_update["diagnosis_signal"]["risk_level"] == "high"
    assert safe_update["diagnosis_signal"]["risk_level"] == "low"


@pytest.mark.asyncio
async def test_orchestrator_single_weak_signal_does_not_escalate_risk(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        # Low coverage alone, no failed quality gate and no sparse draft -> should stay low risk.
        "section_coverage": {"scope_capabilities": "missing", "vision_objectives": "missing"},
        "quality_report": None,
        "draft_body": "a fully drafted section body",
    }

    update = await orchestrator_node(state, {})

    assert update["diagnosis_signal"]["risk_level"] == "low"


@pytest.mark.asyncio
async def test_orchestrator_validated_coverage_downgrades_sections_with_violations(decision_graph_factory):
    """Phase 4: a section with a violation finding no longer counts as covered, so a nominally-full
    coverage on a sparse draft flips the diagnosis from low to high risk."""
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "section_coverage": {"scope_capabilities": "filled", "vision_objectives": "filled"},
        "section_findings": {
            "scope_capabilities": [{"severity": "violation", "message": "x"}],
            "vision_objectives": [{"severity": "violation", "message": "x"}],
        },
        "quality_report": None,
        "draft_body": "",  # sparse
    }

    update = await orchestrator_node(state, {})

    assert update["diagnosis_signal"]["risk_level"] == "high"
    assert "low_coverage" in update["diagnosis_signal"]["signals"]


@pytest.mark.asyncio
async def test_orchestrator_diagnosis_disabled_is_noop(decision_graph_factory, monkeypatch):
    monkeypatch.setattr(settings, "enable_adaptive_diagnosis", False)
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "section_coverage": {"scope_capabilities": "missing"},
        "quality_report": {"quality_gate_result": "fail"},
        "draft_body": "",
    }

    update = await orchestrator_node(state, {})

    assert update["thinking_mode"] is None
    assert update["diagnosis_signal"] is None


@pytest.mark.asyncio
async def test_orchestrator_low_risk_never_triggers_judge(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "section_coverage": {"scope_capabilities": "filled"},
        "quality_report": None,
        "draft_body": "a fully drafted section body",
        "diagnosis_judge_calls_used": 0,
    }

    update = await orchestrator_node(state, {})

    assert update["diagnosis_signal"]["risk_level"] == "low"
    assert update["diagnosis_signal"]["escalation"] == "not_needed"
    assert "judge_result" not in update["diagnosis_signal"]
    assert update["diagnosis_judge_calls_used"] == 0


@pytest.mark.asyncio
async def test_orchestrator_high_risk_escalates_judge_and_spends_budget(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "section_coverage": {"scope_capabilities": "missing"},
        "quality_report": {"quality_gate_result": "fail"},
        "draft_body": "",
        "diagnosis_judge_calls_used": 0,
    }

    update = await orchestrator_node(state, {})

    assert update["diagnosis_signal"]["risk_level"] == "high"
    assert update["diagnosis_signal"]["escalation"] == "escalated"
    assert "judge_result" in update["diagnosis_signal"]
    assert update["diagnosis_judge_calls_used"] == 1


@pytest.mark.asyncio
async def test_orchestrator_high_risk_skips_judge_when_budget_exhausted(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "confirmed"})

    state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "section_coverage": {"scope_capabilities": "missing"},
        "quality_report": {"quality_gate_result": "fail"},
        "draft_body": "",
        "diagnosis_judge_calls_used": DIAGNOSIS_JUDGE_CALLS_MAX,
    }

    update = await orchestrator_node(state, {})

    assert update["diagnosis_signal"]["risk_level"] == "high"
    assert update["diagnosis_signal"]["escalation"] == "escalation_skipped_budget"
    assert "judge_result" not in update["diagnosis_signal"]
    assert update["diagnosis_judge_calls_used"] == DIAGNOSIS_JUDGE_CALLS_MAX


@pytest.mark.asyncio
async def test_orchestrator_resurfaces_parked_question(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "fact", "status": "confirmed"},
        {"id": "Q4", "kind": "open_question", "status": "parked", "blocks": ["N7"]},
    )

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "brd"}, {})

    assert update["feedback_summary"]["resurfaced_questions"][0]["id"] == "Q4"


@pytest.mark.asyncio
async def test_completeness_sweep_only_on_trigger(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "objective", "status": "confirmed"},
        {"id": "N8", "kind": "assumption", "status": "confirmed"},
    )

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "brd"}, {})

    assert "decision_nodes" not in update
    assert "sweep_gaps" not in update["feedback_summary"]


@pytest.mark.asyncio
async def test_orchestrator_depth_transition_creates_parked_questions(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "objective", "status": "confirmed"},
        {"id": "N8", "kind": "assumption", "status": "confirmed"},
    )

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "prd"}, {})

    created = update["feedback_summary"]["created_parked_questions"]
    assert created
    assert all(update["decision_nodes"][item["id"]]["status"] == "parked" for item in created)
