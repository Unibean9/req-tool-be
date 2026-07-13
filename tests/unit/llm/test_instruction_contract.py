"""Tests for the layered instruction contract (spec §5, §6, §13, addendum §9).

Also covers D5 contextual layers: has_draft filtering and cache isolation.
"""

import pytest

from app.instructions import _assembled_cache, _overlay_cache, get_instruction, load_instructions

_SHARED_MARKERS = (
    "Requirements Taxonomy",
    "Decision Policy",
    "Question Policy",
    "Tool Policy",
    "Governance",
    "BMAD",
)


@pytest.fixture(autouse=True)
def _loaded():
    load_instructions()


def _ba():
    return get_instruction(artifact_type="problem", workflow_area="", agent_role="business_analyst")


def _pm():
    return get_instruction(artifact_type="goal", workflow_area="", agent_role="product_manager")


def test_shared_layers_present_for_every_role():
    for instruction in (_ba(), _pm()):
        for marker in _SHARED_MARKERS:
            assert marker in instruction, marker


def test_role_overlay_distinguishes_ba_from_pm():
    ba, pm = _ba(), _pm()
    assert "Business Analyst" in ba and "Business Analyst" not in pm
    assert "Product Manager" in pm and "Product Manager" not in ba


def test_no_shared_layer_duplicated_across_roles():
    ba_shared = _ba().replace(_overlay_cache["business_analyst"], "")
    pm_shared = _pm().replace(_overlay_cache["product_manager"], "")
    assert ba_shared == pm_shared


def test_taxonomy_uses_6_brd_sections_not_9_slots():
    """The taxonomy catalog moved out of the static prompt into the per-turn chain block
    (memory/context holds evidence); the registry remains the 6-section BRD source of truth
    (risks_issues merged into constraints_assumptions; executive_summary promoted)."""
    from app.documents.registry import all_item_types

    types = all_item_types()
    for section in (
        "problem_statement", "vision_objectives", "stakeholder_register", "scope_capabilities",
        "business_rules", "constraints_assumptions",
    ):
        assert section in types, section
    assert "risks_issues" not in types
    assert "why_now" not in types
    # The static prompt no longer dumps the full catalog or legacy mode phrasing.
    assert "qa | critique | explore | draft" not in _ba()


def test_tool_policy_references_current_tools():
    instruction = _ba()
    for tool in (
        "run_critique",
        "write_draft",
        "finalize",
        "recommend_next_workflow",
        "run_readiness_check",
    ):
        assert tool in instruction, tool


def test_decision_tool_layer_matches_current_markdown_and_elicit_contract():
    instruction = _ba()

    assert "When decision-graph tools" not in instruction
    assert "create_decision_node" not in instruction
    assert "State model — Markdown draft" in instruction
    for technique in ("pre_mortem", "tree_of_thought", "socratic_questioning", "challenge_assumptions"):
        assert technique in instruction
    assert "On first contact" not in instruction


def test_output_contract_does_not_restate_json_schema():
    instruction = _ba()
    # No raw JSON schema fragments embedded.
    assert '"properties"' not in instruction
    assert '"enum"' not in instruction
    assert "shape is enforced by the harness" in instruction


def test_bmad_method_layer_present_and_bounded():
    instruction = _ba()
    assert "BMAD Method" in instruction
    assert "brainstorm → brief → prd" in instruction
    # The whole static contract stays high-signal/short — a guard against re-bloating the system prompt.
    # Ceiling re-baselined to 1900 when the deliberate 04-feedback-response layer was added; keep it tight.
    assert len(instruction.split()) < 1900


def test_get_instruction_falls_back_to_workflow_area():
    instruction = get_instruction(artifact_type="zzz_unmapped", workflow_area="prd", agent_role=None)
    assert instruction is not None
    assert "Product Manager" in instruction


def test_get_instruction_never_none_for_any_artifact_type():
    """Every ArtifactType resolves to a contract, even with the default workflow_area and no role —
    so analyze_node always sends a system prompt and the inline payload never carries policy alone."""
    from app.models.artifact import ArtifactType

    for at in ArtifactType:
        assert get_instruction(artifact_type=at.value, workflow_area="analysis", agent_role=None) is not None, at


def test_get_instruction_defaults_when_everything_unmapped():
    instruction = get_instruction(artifact_type="zzz", workflow_area="zzz", agent_role=None)
    assert instruction is not None
    # Default role is the Business Analyst.
    assert "Business Analyst" in instruction


def test_output_contract_carries_content_depth_rule():
    """The synthesis/content-depth rule moved from an inline prompt directive into the output layer."""
    instruction = _ba()
    assert "Content depth" in instruction
    assert "fabricate" in instruction
    assert "evidence" in instruction
    assert "never paste the transcript" in instruction
    assert "needs_confirmation" in instruction


# ---------------------------------------------------------------------------
# Feedback response contract (04-feedback-response.md)
# ---------------------------------------------------------------------------

# Every signal key rendered by _build_feedback_control_block + the diagnosis block must have a
# defined response row in the assembled contract. Kept in sync with prompt_assembly.py.
_FEEDBACK_SIGNAL_MARKERS = (
    "resurfaced_questions",
    "depth_signal",
    "sweep_gaps",
    "created_parked_questions",
    "stale_warning",
    "stale_base_version",
    "lifecycle_persist_rejection",
    "candidate_readiness_rejection",
    "dropped tools",
    "out-of-phase tools",
    "diagnosis_risk",
    "tool_errors",
)


def test_feedback_response_layer_present_for_every_role():
    for instruction in (_ba(), _pm()):
        assert "Feedback Response" in instruction
        assert "FEEDBACK CONTROL:" in instruction


def test_feedback_response_layer_covers_every_rendered_signal():
    instruction = _ba()
    for marker in _FEEDBACK_SIGNAL_MARKERS:
        assert marker in instruction, marker


def test_feedback_response_layer_survives_pre_draft():
    """Feedback signals fire before a draft exists, so the layer is kept when has_draft is False."""
    result = get_instruction("brd", "product_analysis", None, context={"has_draft": False})
    assert result is not None
    assert "Feedback Response" in result


# ---------------------------------------------------------------------------
# D5 — Contextual layers (has_draft filtering)
# ---------------------------------------------------------------------------

def test_has_draft_false_omits_output_section():
    """Pre-draft, the whole ## Output section (critique/governance/output) is dropped to save tokens."""
    result = get_instruction("brd", "product_analysis", None, context={"has_draft": False})
    assert result is not None
    assert "## Output" not in result
    assert "Governance Policy" not in result
    assert "Critique and Validation Policy" not in result


def test_has_draft_true_includes_output_section():
    result = get_instruction("brd", "product_analysis", None, context={"has_draft": True})
    assert result is not None
    assert "## Output" in result
    assert "Governance Policy" in result
    assert "Critique and Validation Policy" in result


def test_context_none_behaves_same_as_has_draft_true():
    """context=None backward-compat: same full instruction as has_draft=True."""
    r_none = get_instruction("brd", "product_analysis", None, context=None)
    r_true = get_instruction("brd", "product_analysis", None, context={"has_draft": True})
    assert r_none == r_true


def test_cache_entries_for_false_and_true_are_distinct():
    _assembled_cache.clear()
    get_instruction("brd", "product_analysis", None, context={"has_draft": False})
    get_instruction("brd", "product_analysis", None, context={"has_draft": True})
    role = "business_analyst"
    assert (role, False) in _assembled_cache
    assert (role, True) in _assembled_cache
    assert _assembled_cache[(role, False)] != _assembled_cache[(role, True)]
