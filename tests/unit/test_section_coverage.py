"""Document coverage contract after the parent-child migration."""

from app.graphs.nodes import _build_tool_selection_prompt
from app.instructions import get_instruction, load_instructions
from tests.integration.test_graph_nodes import _state


def test_taxonomy_contract_uses_database_coverage():
    load_instructions()
    contract = get_instruction(
        artifact_type="problem_statement",
        workflow_area="analysis",
        agent_role=None,
    )
    assert "accepted child artifacts" in contract
    assert "section_assessment" not in contract


def test_prompt_has_no_legacy_slot_or_assessment_fields():
    prompt = _build_tool_selection_prompt(
        _state(artifact_type="problem_statement"),
        [],
    )
    assert "slot_assessment" not in prompt
    assert "section_assessment" not in prompt
