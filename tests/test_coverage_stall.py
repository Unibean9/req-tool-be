"""Tests for the section-coverage directive, hint, and stall counter (tool-loop world).

The section taxonomy replaced the BRD slot model: analyze_node reads `section_assessment`,
computes coverage over 7 sections, and maintains a stall counter so the deterministic hint
stops re-asking once coverage stops advancing. The directive carries section descriptions plus
a grading rubric, and the hint lists every weak section as a gap inventory rather than pinning
one scripted question.

The agent is a PURE LangGraph tool-loop: analyze_node always emits a tool selection
(TOOL_SELECTION_SCHEMA) and route_node no longer vetoes on coverage.
"""

import uuid

import pytest

from app.graphs.section_schema import COVERAGE_STALL_LIMIT, SECTION_DESCRIPTIONS
from tests.helpers import create_org, create_project, make_auth_headers
from tests.test_graph_nodes import _config, _make_agent_session, _session_factory, _state

# ---------------------------------------------------------------------------
# Fix 1 — the taxonomy instruction layer carries the grading rubric + emit mandate
# ---------------------------------------------------------------------------
# The section grading policy moved from an inline prompt directive into the taxonomy layer (the
# system prompt). These guard that the policy still exists where the harness now expects it.

def _contract(artifact_type: str = "intent") -> str:
    from app.instructions import get_instruction, load_instructions

    load_instructions()
    return get_instruction(artifact_type=artifact_type, workflow_area="analysis", agent_role=None)


def test_contract_lists_section_descriptions():
    contract = _contract()

    # Each section appears with its human description (rendered from section_schema), not just the key.
    assert SECTION_DESCRIPTIONS["vision_objectives"] in contract
    assert SECTION_DESCRIPTIONS["problem_statement"] in contract


def test_contract_carries_grading_rubric_and_credit_latest():
    contract = _contract()

    # Explicit grading rubric so the model stops defaulting everything to 'missing'.
    assert "filled" in contract and "partial" in contract and "missing" in contract
    # Must credit the latest user answer instead of re-grading it 'missing'.
    assert "latest" in contract.lower()


def test_contract_is_rubric_not_script():
    """The taxonomy is a reference set of completeness angles, not a sequential march."""
    contract = _contract()

    assert "NOT a checklist to interrogate in order" in contract


def test_contract_carries_section_assessment_mandate():
    """compute_section_coverage depends on the LLM emitting section_assessment every turn."""
    contract = _contract()

    assert "section_assessment" in contract


# ---------------------------------------------------------------------------
# Fix 2 — stall detection rewrites the coverage hint
# ---------------------------------------------------------------------------

def test_section_hint_breaks_loop_on_stall():
    from app.graphs.nodes import _build_section_coverage_hint

    state = _state(artifact_type="intent")
    state["coverage_complete"] = False
    state["section_coverage"] = {"vision_objectives": "missing", "problem_statement": "missing"}
    state["section_coverage_stall_count"] = COVERAGE_STALL_LIMIT

    hint = _build_section_coverage_hint(state)

    assert "không tăng" in hint
    assert "propose" in hint


def test_section_hint_lists_all_weak_sections():
    """Below the stall limit the hint is a gap inventory listing every weak section."""
    from app.graphs.nodes import _build_section_coverage_hint

    state = _state(artifact_type="intent")
    state["coverage_complete"] = False
    state["section_coverage_stall_count"] = 0
    state["section_coverage"] = {
        "vision_objectives": "missing",
        "problem_statement": "partial",
        "stakeholder_register": "needs_review",
    }

    hint = _build_section_coverage_hint(state)

    assert "không tăng" not in hint
    assert SECTION_DESCRIPTIONS["vision_objectives"] in hint
    assert SECTION_DESCRIPTIONS["problem_statement"] in hint
    assert SECTION_DESCRIPTIONS["stakeholder_register"] in hint
    # LLM is invited to choose the angle, not marched through a checklist.
    assert "angle" in hint or "góc độ" in hint


# ---------------------------------------------------------------------------
# Fix 2 — analyze_node maintains the stall counter
# ---------------------------------------------------------------------------

async def _run_analyze(client, db_session, section_assessment, prev_ratio, prev_stall):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = _AsyncLLM(section_assessment)
    state = _state(artifact_type="intent")
    state["coverage_ratio"] = prev_ratio
    state["section_coverage_stall_count"] = prev_stall
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)
    return result, agent_session


class _AsyncLLM:
    """Tool-loop analyst stub: scripts an ask_user selection that reports section_assessment.

    analyze_node reads section_assessment off the returned dict to compute coverage and maintain
    the stall counter; the named tool only drives the emitted AIMessage (irrelevant here).
    """

    def __init__(self, section_assessment):
        self._section_assessment = section_assessment

    async def generate(self, **kwargs):
        return (
            {
                "tool": "ask_user",
                "message": "Câu hỏi tiếp theo?",
                "confidence": 0.3,
                "gaps": [],
                "section_assessment": self._section_assessment,
            },
            None,
        )


@pytest.mark.asyncio
async def test_analyze_increments_stall_when_coverage_flat(client, db_session):
    all_missing = {section: "missing" for section in SECTION_DESCRIPTIONS}

    result, _ = await _run_analyze(client, db_session, all_missing, prev_ratio=0.0, prev_stall=0)

    assert result["coverage_ratio"] == 0.0
    assert result["section_coverage_stall_count"] == 1


@pytest.mark.asyncio
async def test_analyze_resets_stall_when_coverage_improves(client, db_session):
    one_filled = {section: "missing" for section in SECTION_DESCRIPTIONS}
    one_filled["vision_objectives"] = "filled"

    result, _ = await _run_analyze(client, db_session, one_filled, prev_ratio=0.0, prev_stall=2)

    assert result["coverage_ratio"] > 0.0
    assert result["section_coverage_stall_count"] == 0


@pytest.mark.asyncio
async def test_analyze_stall_zero_when_no_section_assessment(client, db_session):
    """No section_assessment -> fail-open coverage -> stall counter resets to 0."""
    result, _ = await _run_analyze(client, db_session, None, prev_ratio=0.0, prev_stall=2)

    assert result["coverage_complete"] is None
    assert result["section_coverage_stall_count"] == 0
