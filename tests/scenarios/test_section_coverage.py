"""Section-coverage wiring end-to-end (spec §3, §7.4) — replaces the legacy slot-coverage scenarios.

The analyst reports `section_assessment` over 7 sections; analyze_node computes coverage with the
single-arg `compute_section_coverage` and persists `section_coverage` to the checkpoint. These
scenarios drive the full HTTP loop via ScriptedLLM, which injects the assessment directly.
"""

import uuid

import pytest

from tests.scenarios.driver import Scenario, ScenarioDriver
from tests.scenarios.scripted_llm import ScriptedLLM, tool_select
from tests.test_graph_nodes import _state

pytestmark = pytest.mark.asyncio

_FULL_SECTIONS = {
    "vision_objectives": "filled",
    "problem_statement": "filled",
    "stakeholder_register": "filled",
    "scope_capabilities": "filled",
    "business_rules": "filled",
    "constraints_assumptions": "filled",
    "risks_issues": "filled",
}


def _proposed_tool_calls(snapshot: dict) -> list[dict]:
    return [tc for tc in snapshot["tool_calls"] if tc["status"] == "proposed"]


async def test_section_proposes_after_coverage_met(client, scenario_env, scenario_project):
    """write_draft with all 7 sections filled -> coverage complete -> proposes to the approval gate."""
    headers, project = scenario_project
    scenario = Scenario(
        name="section-proposes-after-coverage-met",
        artifact_type="problem",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("write_draft", title="Vấn đề hoàn chỉnh",
                            body="Nội dung đầy đủ cho vấn đề đăng ký lớp.",
                            active_mode="structuring", section_assessment=_FULL_SECTIONS),
            ]
        ),
        actions=[{"type": "send", "content": "Đây là thông tin đầy đủ cho vấn đề."}],
        expect={"min_coverage": 1.0},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "propose_artifacts"
    assert _proposed_tool_calls(recorder.steps[-1]["snapshot"])


async def test_business_rules_section_tracked(client, scenario_env, scenario_project):
    """business_rules is a section key (not an artifact_type): a 'filled' grade lands in section_coverage."""
    headers, project = scenario_project
    scenario = Scenario(
        name="business-rules-section-tracked",
        artifact_type="problem",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("write_draft", title="Vấn đề với quy tắc nghiệp vụ",
                            body="Có quy tắc nghiệp vụ rõ ràng.",
                            active_mode="structuring", section_assessment=_FULL_SECTIONS),
            ]
        ),
        actions=[{"type": "send", "content": "Quy tắc: nếu trùng lịch thì chặn đăng ký."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    await driver.run()
    section_coverage = await scenario_env.get_checkpoint_field(driver.session_id, "section_coverage")

    assert section_coverage["business_rules"] == "filled"


async def test_section_coverage_does_not_gate_non_section_artifacts(client, scenario_env, scenario_project):
    """Non-section artifact: write_draft without section_assessment -> coverage fail-open (None)."""
    headers, project = scenario_project
    scenario = Scenario(
        name="non-section-fail-open",
        artifact_type="functional_requirement",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("write_draft", title="FR-1: Đăng ký lớp",
                            body="Hệ thống phải cho phép sinh viên đăng ký lớp.",
                            active_mode="structuring"),
            ]
        ),
        actions=[{"type": "send", "content": "Tôi cần một yêu cầu chức năng cho đăng ký lớp."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()
    coverage_ratio = await scenario_env.get_checkpoint_field(driver.session_id, "coverage_ratio")

    assert coverage_ratio is None
    assert recorder.summary["final_interrupt"] == "propose_artifacts"
    assert _proposed_tool_calls(recorder.steps[-1]["snapshot"])


async def test_no_slot_assessment_in_prompt():
    """The tool-selection prompt references section_assessment, never the legacy slot_assessment."""
    from app.graphs.nodes import _build_tool_selection_prompt

    prompt = _build_tool_selection_prompt(_state(artifact_type="problem"), [])

    assert "slot_assessment" not in prompt
    assert "section_assessment" in prompt
