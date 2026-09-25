"""Selective predecessor context (app/graphs/analysis/context_excerpts.py)."""

from app.graphs.analysis.context_excerpts import (
    CONTEXT_PICKS,
    SectionPick,
    excerpt_body,
    select_sections,
    unknown_reference_warnings,
)
from app.graphs.analysis.prompt_assembly import _build_predecessor_content_block

STAKEHOLDERS = """## Stakeholders

### Nhóm 1 — Người dùng trực tiếp

| Vai trò | Trách nhiệm | Quyền quyết định | Nhu cầu / Mối quan tâm | Mức độ tham gia |
|---|---|---|---|---|
| **Trưởng nhóm** | Tạo task | **Cao** | Tiết kiệm thời gian | Hàng ngày |
| **Thành viên** | Cập nhật trạng thái | Thấp | Rõ việc | Hàng ngày |

## Ma trận Ảnh hưởng / Mối quan tâm (Influence / Interest Matrix)

| Stakeholder | Mức độ ảnh hưởng |
|---|---|
| Trưởng nhóm | Cao |

## Assumptions (cần validate)

| # | Giả định | Độ tin cậy |
|---|---|---|
| A1 | Nhóm dưới 10 người | Cao |
"""

SCOPE = """## Scope
Phạm vi MVP.

## Capabilities

| # | Capability | Ưu tiên | Lý do / Liên kết | Phụ thuộc |
|---|---|---|---|---|
| C1 | Tạo task | **Must** | Root cause | — |
| C2 | Auto-assign | **Must** | Giảm thời gian | C1 |

## Out-of-Scope (Giai đoạn đầu)
- Chat nội bộ
"""


def test_excerpt_keeps_only_picked_sections_and_columns_matched_in_vietnamese():
    excerpt = excerpt_body(STAKEHOLDERS, (SectionPick("Stakeholders", ("role", "decision")),))

    assert excerpt.included == ["Stakeholders"]
    assert excerpt.omitted == [
        "Ma trận Ảnh hưởng / Mối quan tâm (Influence / Interest Matrix)",
        "Assumptions (cần validate)",
    ]
    assert "| Vai trò | Quyền quyết định |" in excerpt.text
    assert "| **Thành viên** | Thấp |" in excerpt.text
    assert "Trách nhiệm" not in excerpt.text
    assert "Giả định" not in excerpt.text


def test_excerpt_always_keeps_the_identifier_column_and_every_row():
    excerpt = excerpt_body(SCOPE, (SectionPick("Capabilities", ("priority",)),))

    assert "| # | Ưu tiên |" in excerpt.text
    assert "| C1 | **Must** |" in excerpt.text
    assert "| C2 | **Must** |" in excerpt.text


def test_a_table_without_any_wanted_column_is_kept_whole_rather_than_guessed():
    excerpt = excerpt_body(STAKEHOLDERS, (SectionPick("Assumptions", ("capability",)),))

    assert "| A1 | Nhóm dưới 10 người | Cao |" in excerpt.text


def test_heading_match_tolerates_hyphens_and_suffixes():
    excerpt = excerpt_body(SCOPE, (SectionPick("Out of Scope"),))

    assert excerpt.included == ["Out-of-Scope (Giai đoạn đầu)"]
    assert "Chat nội bộ" in excerpt.text


def test_a_body_without_any_picked_heading_is_used_whole():
    body = "Free text without the contract headings."

    excerpt = excerpt_body(body, (SectionPick("Capabilities"),))

    assert excerpt.text == body
    assert excerpt.included == ["(whole artifact)"]


def test_select_sections_returns_named_sections_or_none():
    assert select_sections(SCOPE, ["Capabilities"]).startswith("## Capabilities")
    assert select_sections(SCOPE, ["Nonexistent"]) is None


def test_every_pick_names_another_real_artifact_type_and_known_columns():
    from app.graphs.analysis.context_excerpts import COLUMN_KEYWORDS
    from app.graphs.analysis.context_loader import _KNOWN_ARTIFACT_TYPES

    for artifact_type, sources in CONTEXT_PICKS.items():
        assert artifact_type in _KNOWN_ARTIFACT_TYPES
        for source, picks in sources.items():
            assert source in _KNOWN_ARTIFACT_TYPES and source != artifact_type
            for pick in picks:
                assert set(pick.columns) <= set(COLUMN_KEYWORDS), (artifact_type, source, pick)


def test_unknown_reference_flags_ids_the_predecessor_does_not_define():
    draft = """## Business Rules
| ID | Quy tắc | Liên kết |
|---|---|---|
| BR-R1 | Chỉ trưởng nhóm tạo task | C1, C9 |
| BR-R2 | Ràng buộc CON-1 | C2 |
"""

    warnings = unknown_reference_warnings(draft, {"scope_capabilities": SCOPE})

    # C9 is cited but scope only defines C1/C2; the draft's own BR-R ids and a CON- id from an
    # unrelated scheme are not flagged.
    assert warnings == ["unknown_reference: C9 is not defined in scope_capabilities"]


def test_prompt_marks_an_excerpt_and_what_it_left_out():
    block = _build_predecessor_content_block(
        [
            {
                "artifact_type": "scope_capabilities",
                "artifact_id": "id-1",
                "title": "Scope",
                "body": "## Capabilities\n...",
                "truncated": False,
                "included_sections": ["Capabilities"],
                "omitted_sections": ["Scope", "Assumptions"],
            }
        ]
    )

    assert "Excerpt — sections shown: Capabilities" in block
    assert "Not loaded: Scope, Assumptions" in block


def test_every_pick_source_names_at_least_one_heading_its_contract_requires():
    """Catches a mistyped heading: a pick that matches nothing silently falls back to the whole
    body, which would quietly undo the point of selecting."""
    from app.documents.registry import output_contract
    from app.graphs.analysis.context_excerpts import _matches

    # Optional sections agents routinely add to many artifacts (seen in real BRD bodies), which a
    # pick may legitimately rely on even though no contract requires them.
    optional = ("## Assumptions", "## Out of Scope")
    for artifact_type, sources in CONTEXT_PICKS.items():
        for source, picks in sources.items():
            required = (*output_contract(source).required_headings, *optional)
            assert any(_matches(heading.removeprefix("## "), pick.heading) for pick in picks for heading in required), (
                artifact_type,
                source,
            )


def _rendered_domain_events() -> str:
    from app.graphs.decision_graph import _render_entries

    nodes = [
        {
            "statement": "Task Created",
            "fields": {"trigger": "CMD-01", "triggered by": "Team lead", "downstream effects": "Notify assignee"},
        },
        {
            "statement": "Task Assigned",
            "fields": {"trigger": "CMD-02", "triggered by": "System", "downstream effects": "Update workload"},
        },
    ]
    return "## Domain Events\n" + _render_entries(nodes, ("trigger", "triggered by", "downstream effects"), "EVT")


def test_entry_style_items_keep_only_the_wanted_fields():
    excerpt = excerpt_body(_rendered_domain_events(), (SectionPick("Domain Events", ("condition",)),))

    # "condition" matches the "trigger"/"triggered by" field labels; ids and titles always stay.
    assert "### EVT-01: Task Created" in excerpt.text
    assert "### EVT-02: Task Assigned" in excerpt.text
    assert "- **trigger:** CMD-02" in excerpt.text
    assert "downstream effects" not in excerpt.text


def test_entry_style_items_without_a_matching_field_are_kept_whole():
    body = _rendered_domain_events()

    excerpt = excerpt_body(body, (SectionPick("Domain Events", ("capability",)),))

    assert excerpt.text == body


def _read(call_id: str, artifact_id: str, sections=None):
    from langchain_core.messages import AIMessage, ToolMessage

    args = {"id": artifact_id, **({"sections": sections} if sections else {})}
    return [
        AIMessage(content="", tool_calls=[{"id": call_id, "name": "read_artifact", "args": args}]),
        ToolMessage(content=f"BODY of {artifact_id} {sections or 'all'}", tool_call_id=call_id, name="read_artifact"),
    ]


def _tool_results(messages):
    return [
        block["content"]
        for message in messages
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if block.get("type") == "tool_result"
    ]


def test_an_artifact_read_again_later_keeps_only_the_latest_copy_in_history():
    from langchain_core.messages import HumanMessage

    from app.graphs.analysis.prompt_assembly import _build_analyzer_messages

    history = [
        HumanMessage(content="viết NFR"),
        *_read("r1", "fr", ["Functional Requirements"]),
        *_read("r2", "scope"),
        *_read("r3", "fr"),  # whole body: repeats r1
        *_read("r4", "scope", ["Capabilities"]),  # fewer sections than r2: does not repeat it
    ]

    results = _tool_results(
        _build_analyzer_messages({"messages": history, "artifact_type": "non_functional_requirement"}, "")
    )

    assert results[0].startswith("(Content omitted")  # r1
    assert results[1] == "BODY of scope all"
    assert results[2] == "BODY of fr all"
    assert results[3] == "BODY of scope ['Capabilities']"
