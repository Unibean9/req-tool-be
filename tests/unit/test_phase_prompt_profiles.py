"""Per-phase prompt profiles.

Each session phase renders only the prompt blocks relevant to its job — no phase renders the full
god-assembly. An unset/unknown phase falls back to including every block (backward compatible with
legacy checkpoints and callers that build a prompt before orchestrator_node assigns a phase).
"""

from app.documents.registry import children_of
from app.graphs.analysis.prompt_assembly import (
    _PHASE_PROFILE_BLOCKS,
    _phase_includes,
    build_system_prompt,
)
from app.graphs.nodes import _build_tool_selection_prompt
from app.graphs.session_phase import DRAFT, ELICIT, FINALIZE, INTENT, PHASES, REVIEW
from app.instructions import load_instructions
from tests.factories import _state

_ARTIFACT_CONTRACT_MARKER = "provenance chain"
_BATCHING_MARKER = "BATCHING:"
_THINKING_MODE_MARKER = "THINKING MODE:"
_SECTION_COVERAGE_MARKER = "Section coverage - aspects"
_DECISION_VIEW_MARKER = "DRAFT IN PROGRESS"
_SECTION_REPAIR_MARKER = "SECTION REPAIR"

_VIOLATION = {"section": "## Business Rules", "severity": "violation", "message": "missing outcome"}
_WARNING = {"section": "## Objectives", "severity": "warning", "message": "weasel word: fast"}


def _phase_state(phase: str | None, **overrides):
    state = _state(artifact_type="brd")
    state["user_confirmed"] = True
    state["session_phase"] = phase
    state.update(overrides)
    return state


# --- profile table (authoritative gating) -----------------------------------


def test_every_known_phase_has_a_profile():
    assert set(_PHASE_PROFILE_BLOCKS) == set(PHASES)


def test_no_phase_includes_every_gated_block():
    gated = {"artifact_contract", "thinking_mode", "section_coverage", "decision_view", "batching"}
    for phase in PHASES:
        assert _PHASE_PROFILE_BLOCKS[phase] != gated, f"{phase} renders the full god-assembly"


def test_unknown_phase_falls_back_to_full_assembly():
    # None (legacy checkpoint) and an unmodeled label both include every block.
    assert _phase_includes({"session_phase": None}, "artifact_contract") is True
    assert _phase_includes({}, "decision_view") is True
    assert _phase_includes({"session_phase": "bogus"}, "batching") is True


def test_profile_gating_matches_table():
    assert _phase_includes({"session_phase": DRAFT}, "artifact_contract") is True
    assert _phase_includes({"session_phase": INTENT}, "artifact_contract") is False
    assert _phase_includes({"session_phase": REVIEW}, "artifact_contract") is True
    assert _phase_includes({"session_phase": ELICIT}, "batching") is True
    assert _phase_includes({"session_phase": DRAFT}, "batching") is False
    assert _phase_includes({"session_phase": REVIEW}, "decision_view") is True
    assert _phase_includes({"session_phase": FINALIZE}, "decision_view") is False


# --- rendered system prompt --------------------------------------------------


def test_draft_phase_renders_artifact_contract_but_not_batching():
    load_instructions()
    prompt = build_system_prompt(_phase_state(DRAFT), None, has_draft=True)
    assert _ARTIFACT_CONTRACT_MARKER in prompt
    assert _BATCHING_MARKER not in prompt


def test_elicit_phase_renders_batching_and_thinking_mode_but_not_contract():
    load_instructions()
    prompt = build_system_prompt(_phase_state(ELICIT, thinking_mode="challenging"), None, has_draft=False)
    assert _BATCHING_MARKER in prompt
    assert _THINKING_MODE_MARKER in prompt
    assert _ARTIFACT_CONTRACT_MARKER not in prompt


def test_intent_phase_renders_neither_contract_nor_batching():
    load_instructions()
    prompt = build_system_prompt(_phase_state(INTENT, user_confirmed=None), None, has_draft=False)
    assert _ARTIFACT_CONTRACT_MARKER not in prompt
    assert _BATCHING_MARKER not in prompt


def test_review_phase_renders_contract_but_suppresses_technique_menu():
    load_instructions()
    prompt = build_system_prompt(_phase_state(REVIEW, thinking_mode="challenging"), None, has_draft=True)
    assert _THINKING_MODE_MARKER not in prompt
    assert _ARTIFACT_CONTRACT_MARKER in prompt


def test_unset_phase_renders_full_assembly():
    load_instructions()
    prompt = build_system_prompt(_phase_state(None), None, has_draft=True)
    assert _ARTIFACT_CONTRACT_MARKER in prompt
    assert _BATCHING_MARKER in prompt


# --- section repair block ----------------------------------------------------


def test_draft_phase_renders_section_repair_with_violation_and_warning():
    load_instructions()
    state = _phase_state(DRAFT, section_findings={"## Business Rules": [_VIOLATION], "## Objectives": [_WARNING]})
    prompt = build_system_prompt(state, None, has_draft=True)
    assert _SECTION_REPAIR_MARKER in prompt
    assert "missing outcome" in prompt
    assert "weasel word: fast" in prompt  # warnings surface from DRAFT onward


def test_elicit_phase_hides_warning_findings_but_shows_violations():
    load_instructions()
    state = _phase_state(ELICIT, section_findings={"## Business Rules": [_VIOLATION], "## Objectives": [_WARNING]})
    prompt = build_system_prompt(state, None, has_draft=False)
    assert "missing outcome" in prompt
    assert "weasel word: fast" not in prompt  # warnings suppressed in ELICIT


def test_finalize_phase_omits_section_repair_entirely():
    load_instructions()
    state = _phase_state(FINALIZE, section_findings={"## Business Rules": [_VIOLATION]})
    prompt = build_system_prompt(state, None, has_draft=True)
    assert _SECTION_REPAIR_MARKER not in prompt


def test_cleared_findings_do_not_render():
    load_instructions()
    state = _phase_state(DRAFT, section_findings={"## Business Rules": []})
    prompt = build_system_prompt(state, None, has_draft=True)
    assert _SECTION_REPAIR_MARKER not in prompt


# --- rendered per-turn payload ----------------------------------------------


def _coverage_gaps():
    return {item_type: "missing" for item_type in children_of("brd")}


def test_elicit_payload_shows_coverage_but_not_decision_view(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "statement": "Reduce cost", "status": "confirmed"},
    )
    state = _phase_state(
        ELICIT,
        coverage_complete=False,
        section_coverage=_coverage_gaps(),
        section_coverage_stall_count=0,
        decision_nodes=nodes,
    )
    prompt = _build_tool_selection_prompt(state, [])
    assert _SECTION_COVERAGE_MARKER in prompt
    assert _DECISION_VIEW_MARKER not in prompt


def test_draft_payload_shows_both_coverage_and_decision_view(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "statement": "Reduce cost", "status": "confirmed"},
    )
    state = _phase_state(
        DRAFT,
        coverage_complete=False,
        section_coverage=_coverage_gaps(),
        section_coverage_stall_count=0,
        decision_nodes=nodes,
    )
    prompt = _build_tool_selection_prompt(state, [])
    assert _SECTION_COVERAGE_MARKER in prompt
    assert _DECISION_VIEW_MARKER in prompt


def test_finalize_payload_hides_coverage_and_decision_view(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "statement": "Reduce cost", "status": "confirmed"},
    )
    state = _phase_state(
        FINALIZE,
        coverage_complete=False,
        section_coverage=_coverage_gaps(),
        section_coverage_stall_count=0,
        decision_nodes=nodes,
    )
    prompt = _build_tool_selection_prompt(state, [])
    assert _SECTION_COVERAGE_MARKER not in prompt
    assert _DECISION_VIEW_MARKER not in prompt


def test_payload_surfaces_high_risk_diagnosis_feedback():
    state = _phase_state(
        DRAFT,
        diagnosis_signal={
            "risk_level": "high",
            "signals": ["low_coverage", "quality_gate_failed"],
            "judge_result": {"score": 0.2, "findings": ["Missing acceptance signals"], "suggestions": []},
        },
    )

    prompt = _build_tool_selection_prompt(state, [])

    assert "diagnosis_risk: high" in prompt
    assert "diagnosis_judge_score: 0.2" in prompt
    assert "Missing acceptance signals" in prompt


def test_payload_tells_model_to_read_named_context_artifact_before_asking():
    state = _phase_state(ELICIT, artifact_type="vision_objectives")
    prompt = _build_tool_selection_prompt(
        state,
        [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "type": "executive_summary",
                "title": "Executive Summary",
                "status": "accepted",
            }
        ],
    )

    assert "Executive Summary" in prompt
    assert "read_artifact" in prompt
    assert "before asking" in prompt


# --- per-artifact-type profile block (Phase 4) ------------------------------

_ELICIT_FOCUS_MARKER = "ELICITATION FOCUS"
_REVIEW_CRITERIA_MARKER = "REVIEW CRITERIA"


def test_elicit_renders_type_profile_for_mapped_type():
    load_instructions()
    prompt = build_system_prompt(_phase_state(ELICIT, artifact_type="constraints_assumptions"), None, has_draft=False)
    assert _ELICIT_FOCUS_MARKER in prompt
    assert "elicit(technique='pre_mortem')" in prompt


def test_review_renders_type_criteria_for_mapped_type():
    load_instructions()
    prompt = build_system_prompt(_phase_state(REVIEW, artifact_type="constraints_assumptions"), None, has_draft=True)
    assert _REVIEW_CRITERIA_MARKER in prompt


def test_unmapped_type_falls_back_to_generic_no_profile():
    """A type with an output contract but no profile fields renders no type-profile block (generic)."""
    load_instructions()
    prompt = build_system_prompt(_phase_state(ELICIT, artifact_type="problem_statement"), None, has_draft=False)
    assert _ELICIT_FOCUS_MARKER not in prompt


def test_type_profile_absent_in_draft_phase():
    """DRAFT carries the section scaffold via the artifact contract, not the type-profile block."""
    load_instructions()
    prompt = build_system_prompt(_phase_state(DRAFT, artifact_type="constraints_assumptions"), None, has_draft=True)
    assert _ELICIT_FOCUS_MARKER not in prompt
    assert _REVIEW_CRITERIA_MARKER not in prompt


def test_payload_omits_artifact_reference_policy_without_source_artifacts():
    state = _phase_state(ELICIT, artifact_type="vision_objectives")
    prompt = _build_tool_selection_prompt(
        state,
        [
            {
                "id": "00000000-0000-0000-0000-000000000002",
                "type": "vision_objectives",
                "title": "Vision and Objectives",
                "status": "draft",
            }
        ],
    )

    assert "ARTIFACT REFERENCE POLICY" not in prompt
