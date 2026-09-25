"""Deterministic coverage gate (app/graphs/analysis/coverage.py) and its write_draft wiring."""

from unittest.mock import patch

import pytest

from app.graphs.agent_tools import _write_draft_impl
from app.graphs.analysis.coverage import (
    CoverageItem,
    coverage_instruction,
    coverage_items,
    missing_coverage,
)
from app.graphs.analysis.prompt_assembly import _build_coverage_rule_block
from app.models.artifact import ArtifactType
from tests.factories import (
    _accept_predecessor,
    _config,
    _focused_items,
    _make_agent_run,
    _make_agent_session,
    _project,
    _session_factory,
    _state,
)

SCOPE = """## Scope
Phạm vi MVP.

## Capabilities

### Phase 1

| Capability | Mức ưu tiên | Lý do | Phụ thuộc |
|---|---|---|---|
| **C1** Tạo task hàng loạt | Must Have | RC1 | — |
| **C2** Auto-assign | Must Have | RC2 | C1 |
| **C3** Chat nội bộ | Won't | Ngoài MVP | — |

### Won't have (this release)

| # | Capability | Ưu tiên |
|---|---|---|
| C4 | Mobile app | Could |

### Assumptions

| # | Giả định |
|---|---|
| A1 | Nhóm dưới 10 người |

## Out of Scope
- C9 Tích hợp ERP
"""

CONSTRAINTS = """## Constraints

### C-LEG — Pháp lý

| Ràng buộc | Tác động |
|---|---|
| **C-LEG-01** Nghị định 13/2023 | Cao |
| **C-TECH-01** Phụ thuộc AI provider | Trung bình |
"""


def test_items_read_ids_embedded_in_the_first_cell_and_skip_wont_and_side_tables():
    items = coverage_items(SCOPE, "Capabilities")

    # C3 is prioritised Won't, C4 sits under a "Won't have" heading, A1 is a side table's family,
    # C9 is outside the section.
    assert items == [CoverageItem("C1", "Tạo task hàng loạt"), CoverageItem("C2", "Auto-assign")]


def test_items_handle_hyphenated_group_ids():
    assert [item.id for item in coverage_items(CONSTRAINTS, "Constraints")] == ["C-LEG-01", "C-TECH-01"]


def test_items_are_empty_when_the_section_has_no_ids():
    assert coverage_items("## Capabilities\nFree text only.", "Capabilities") == []
    assert coverage_items("", "Capabilities") == []


def test_citation_needs_the_exact_id_not_a_longer_or_prefixed_one():
    items = [CoverageItem("C1", ""), CoverageItem("BC1", ""), CoverageItem("C-LEG-01", "")]

    # C10, BC1 and BR-C1 must not count as citing C1.
    assert missing_coverage("Rules for C10, BC1 and BR-C1; see C-LEG-1.", items) == [CoverageItem("C1", "")]
    # Zero padding and an optional hyphen are the same ID.
    assert missing_coverage("(C1) | BC-01 | CLEG01", items) == []


def test_instruction_lists_the_ids_to_cover():
    text = coverage_instruction("business_rules", coverage_items(SCOPE, "Capabilities"))

    assert "scope_capabilities > Capabilities: C1, C2" in text
    assert coverage_instruction("vision_objectives", [CoverageItem("C1", "")]) is None


def test_prompt_block_uses_the_preloaded_source_excerpt():
    block = _build_coverage_rule_block(
        "non_functional_requirement",
        [{"artifact_type": "constraints_assumptions", "body": CONSTRAINTS}],
    )

    assert "C-LEG-01, C-TECH-01" in block
    assert _build_coverage_rule_block("non_functional_requirement", []) == ""


RULES_ONLY_C1 = """## Business Rules

| Rule ID | Condition | Trigger | Outcome | Scope | Exception |
|---|---|---|---|---|---|
| BR-R1 | Task có tiêu đề | Tạo task (C1) | Task được lưu | Trưởng nhóm | Không |
"""

RULES_ALL = RULES_ONLY_C1 + (
    "| BR-R2 | Thành viên có workload | Auto-assign (C2) | Gợi ý người | Hệ thống | Nhóm 1 người |\n"
)


async def _seed_business_rules(client, db_session, *, scope_body: str | None):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.BUSINESS_RULES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)
    if scope_body is not None:
        await _accept_predecessor(db_session, project_id, "scope_capabilities", body=scope_body)

    state = _state(artifact_type="business_rules")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()
    return state, config


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_is_blocked_until_every_capability_is_covered(mock_interrupt, client, db_session):
    state, config = await _seed_business_rules(client, db_session, scope_body=SCOPE)

    command = await _write_draft_impl("Rules", RULES_ONLY_C1, state, config, "call_1")

    mock_interrupt.assert_not_called()
    [error] = command.update["tool_errors"]
    assert error["code"] == "coverage_incomplete"
    message = command.update["messages"][0].content
    assert "C2 (Auto-assign)" in message
    assert "C3" not in message  # Won't items are exempt
    assert command.update["readiness_reject_streak"] == 1


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_second_coverage_rejection_hands_the_gap_to_the_user(mock_interrupt, client, db_session):
    state, config = await _seed_business_rules(client, db_session, scope_body=SCOPE)
    state["readiness_reject_streak"] = 1

    command = await _write_draft_impl("Rules", RULES_ONLY_C1, state, config, "call_1")

    assert "ask_user" in command.update["messages"][0].content
    assert command.update["readiness_reject_streak"] == 2


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_complete_draft_passes_the_coverage_gate(mock_interrupt, client, db_session):
    state, config = await _seed_business_rules(client, db_session, scope_body=SCOPE)

    command = await _write_draft_impl("Rules", RULES_ALL, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_no_accepted_source_means_nothing_to_gate(mock_interrupt, client, db_session):
    state, config = await _seed_business_rules(client, db_session, scope_body=None)

    command = await _write_draft_impl("Rules", RULES_ONLY_C1, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
