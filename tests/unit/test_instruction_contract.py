"""Tests for the layered instruction contract (Phase 7B; spec §5, §6, §13, addendum §9).

Also covers D5 contextual layers: has_draft filtering and cache isolation.
"""

from pathlib import Path

import pytest

import app.instructions as instr_module
from app.instructions import _assembled_cache, get_instruction, load_instructions, role_overlay

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
    ba_shared = _ba().replace(role_overlay("business_analyst"), "")
    pm_shared = _pm().replace(role_overlay("product_manager"), "")
    assert ba_shared == pm_shared


def test_taxonomy_uses_7_sections_not_9_slots():
    """The taxonomy catalog moved out of the static prompt into the per-turn chain block
    (memory/context holds evidence); the registry remains the 7-section source of truth."""
    from app.documents.registry import all_item_types

    types = all_item_types()
    for section in (
        "vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities",
        "business_rules", "constraints_assumptions", "risks_issues",
    ):
        assert section in types, section
    assert "why_now" not in types
    # The static prompt no longer dumps the full catalog or legacy mode phrasing.
    assert "qa | critique | explore | draft" not in _ba()


def test_tool_policy_references_current_tools():
    instruction = _ba()
    for tool in (
        "analysis_frame",
        "run_critique",
        "write_draft",
        "finalize",
        "recommend_next_workflow",
        "run_readiness_check",
    ):
        assert tool in instruction, tool


def test_output_contract_does_not_restate_json_schema():
    instruction = _ba()
    # No raw JSON schema fragments embedded.
    assert '"properties"' not in instruction
    assert '"enum"' not in instruction
    assert "shape is enforced by the harness" in instruction


def test_bmad_method_layer_present_and_bounded():
    assert "BMAD Method" in _ba()
    bmad = Path("app/instructions/layers/04-bmad-method.md").read_text(encoding="utf-8")
    assert len(bmad.split()) < 200


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
    assert "evidence and context" in instruction
    assert "Do not paste the full transcript" in instruction
    assert "(agent suy diễn, cần xác nhận)" in instruction


# ---------------------------------------------------------------------------
# D5 — Contextual layers (has_draft filtering)
# ---------------------------------------------------------------------------

def test_has_draft_false_omits_critique_policy():
    layer_path = Path(instr_module.__file__).parent / "layers" / "08-critique-policy.md"
    if not layer_path.exists():
        pytest.skip("layer 08 not present in this env")
    marker = layer_path.read_text(encoding="utf-8").strip()[:60]
    result = get_instruction("brd", "product_analysis", None, context={"has_draft": False})
    assert result is not None
    assert marker not in result


def test_has_draft_true_includes_critique_policy():
    layer_path = Path(instr_module.__file__).parent / "layers" / "08-critique-policy.md"
    if not layer_path.exists():
        pytest.skip("layer 08 not present in this env")
    marker = layer_path.read_text(encoding="utf-8").strip()[:60]
    result = get_instruction("brd", "product_analysis", None, context={"has_draft": True})
    assert result is not None
    assert marker in result


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
