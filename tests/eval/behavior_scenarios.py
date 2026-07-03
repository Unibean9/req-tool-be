"""Behavior-baseline scenarios: 4 scripted sessions covering the symptom axes.

Scenario set:
- brd-happy-path       — clear brief, cooperative answers, one Q&A round before drafting.
- brd-ambiguous        — vague brief, contradictory answers, rejected first draft, failing
                          critique, clarify, redraft.
- prd-from-brd         — functional requirement derived on top of seeded ACCEPTED BRD
                          predecessors (the driver seeds them).
- sad-tech-decision    — ADR-style technical decision with a risk_review critique after a
                          rejected first draft (the axis expected to trigger diagnosis
                          escalation on live runs).

Stub mode scripts the analyst brain deterministically (CI guard for harness mechanics only);
live mode replaces the ScriptedLLM with a real client — all before/after behavior claims come
from live runs.
"""

from tests.integration.scenarios.driver import Scenario
from tests.integration.scenarios.scripted_llm import ScriptedLLM, tool_select

_CONTINUE = {"type": "send", "content": "Dung roi, tiep tuc giup toi."}

_VISION_BODY = (
    "## Vision\n"
    "The product targets student groups that need a shared study scheduling tool to reduce schedule conflicts.\n\n"
    "## Objectives\n"
    "- Create study groups and sync personal calendars.\n"
    "- Suggest common free time slots for the whole group.\n"
    "- Increase group session attendance.\n\n"
    "## Success Metrics\n"
    "- Reduce coordination time from 30 minutes to under 10 minutes per week within 3 months."
)

_VISION_BODY_CLARIFIED = (
    "## Vision\n"
    "The MVP serves university study groups of 4-6 members who lose sessions to schedule conflicts.\n\n"
    "## Objectives\n"
    "- Sync each member's personal calendar into a shared free/busy view.\n"
    "- Suggest at least 3 common free slots per week per group.\n\n"
    "## Success Metrics\n"
    "- Group session attendance rises from 60% to 80% within one semester."
)

_FR_BODY = (
    "## Functional Requirements\n"
    "| id | requirement | behavior | inputs/outputs | acceptance signal | priority |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
    "| FR-01 | Members connect personal calendars | System extracts busy slots and computes overlaps "
    "when the group leader requests it | Input: calendars, member list, target range; "
    "Output: common slots with available-member counts | Results update within 5 seconds after a "
    "calendar change | Must |"
)

_TECH_DECISION_BODY = (
    "## Technical Decision\n"
    "Use PostgreSQL row-level locking for concurrent schedule-slot booking.\n\n"
    "## Context\n"
    "Multiple group leaders can confirm overlapping slots at the same time; double-booking corrupts "
    "the shared calendar.\n\n"
    "## Options Considered\n"
    "- Application-level mutex per group (single-worker only).\n"
    "- Redis distributed lock (new infrastructure dependency).\n"
    "- PostgreSQL SELECT ... FOR UPDATE on the slot row.\n\n"
    "## Decision\n"
    "PostgreSQL row locks: no new dependency, correct under multi-worker deployment.\n\n"
    "## Consequences\n"
    "- Lock contention is bounded by group size (max 8 members).\n"
    "- Requires transaction-scoped session handling in the booking service."
)

_TECH_DECISION_BODY_REVISED = (
    _TECH_DECISION_BODY
    + "\n- Rollback path: locks degrade to last-writer-wins if the transaction wrapper is removed."
)


def _confirm(summary: str) -> dict:
    return tool_select("confirm_intent", summary=summary, active_mode="discovery")


def brd_happy_path() -> Scenario:
    """Clear brief, one cooperative Q&A round, first draft approved."""
    llm = ScriptedLLM(tool_brain=[
        tool_select("ask_user", message="Who is the primary user of the tool?", active_mode="discovery"),
        _confirm("Study-scheduling coordination tool for university student groups."),
        tool_select("write_draft", title="Vision: study scheduling coordination", body=_VISION_BODY,
                    active_mode="structuring"),
    ])
    return Scenario(
        name="behavior-brd-happy-path",
        artifact_type="intent",
        llm=llm,
        actions=[
            {"type": "send", "content": "Toi muon xay cong cu giup sinh vien orchestration study scheduling."},
            {"type": "send", "content": "Mainly university students studying in groups."},
            _CONTINUE,
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def brd_ambiguous() -> Scenario:
    """Vague brief + contradictory answers: rejected draft, failing critique, clarify, redraft."""
    llm = ScriptedLLM(
        tool_brain=[
            tool_select("ask_user", message="What problem should the product solve first?",
                        active_mode="discovery"),
            tool_select("ask_user",
                        message="You mentioned both lecturers and students as primary users — which one is it for the MVP?",
                        active_mode="discovery"),
            _confirm("Scheduling tool; primary user still ambiguous between students and lecturers."),
            # A confirmed decision node so the critique target (the graph-rendered view) is non-empty.
            tool_select("create_decision_node", node_id="D1", kind="objective",
                        statement="Reduce schedule conflicts for study groups"),
            tool_select("update_decision_node", node_id="D1", status="confirmed"),
            tool_select("write_draft", title="Vision (draft)", body=_VISION_BODY, active_mode="structuring"),
            tool_select("run_critique", target="draft", mode="consistency", active_mode="critique"),
            tool_select("ask_user",
                        message="The draft contradicts itself on the primary user. Confirm: students only for MVP?",
                        active_mode="structuring"),
            tool_select("write_draft", title="Vision: study scheduling (clarified MVP)",
                        body=_VISION_BODY_CLARIFIED, active_mode="structuring"),
        ],
        critique=[{
            "score": 0.55,
            "findings": ["Objectives contradict the stated primary user"],
            "suggestions": ["Pick one primary user for the MVP and align objectives"],
        }],
    )
    return Scenario(
        name="behavior-brd-ambiguous",
        artifact_type="intent",
        llm=llm,
        actions=[
            {"type": "send", "content": "Toi co mot y tuong app nhung chua ro lam."},
            {"type": "send", "content": "Chac la giup giang vien theo doi sinh vien... hay la giup sinh vien tu hoc?"},
            {"type": "send", "content": "Ca hai deu can, nhung sinh vien truoc di."},
            _CONTINUE,
            {"type": "reject_all"},
            {"type": "send", "content": "Dung, MVP chi cho sinh vien thoi."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def prd_from_brd() -> Scenario:
    """Functional requirement derived from an existing accepted BRD (driver seeds predecessors)."""
    llm = ScriptedLLM(tool_brain=[
        _confirm("Derive functional requirements from the accepted BRD for common-slot computation."),
        tool_select("write_draft", title="FR: compute common free slots", body=_FR_BODY,
                    active_mode="structuring"),
    ])
    return Scenario(
        name="behavior-prd-from-brd",
        artifact_type="functional_requirement",
        llm=llm,
        actions=[
            {"type": "send", "content": "BRD da chot roi, viet functional requirement cho tinh khung gio chung."},
            _CONTINUE,
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def sad_tech_decision() -> Scenario:
    """High-risk technical decision: rejected first draft, risk_review critique, revised ADR."""
    llm = ScriptedLLM(
        tool_brain=[
            _confirm("Record the locking strategy decision for concurrent slot booking."),
            # A confirmed decision node so the critique target (the graph-rendered view) is non-empty.
            tool_select("create_decision_node", node_id="T1", kind="decision",
                        statement="Use PostgreSQL row locks for concurrent slot booking"),
            tool_select("update_decision_node", node_id="T1", status="confirmed"),
            tool_select("write_draft", title="ADR: slot booking concurrency", body=_TECH_DECISION_BODY,
                        active_mode="structuring"),
            tool_select("run_critique", target="draft", mode="risk_review", active_mode="critique"),
            tool_select("write_draft", title="ADR: slot booking concurrency (risk-reviewed)",
                        body=_TECH_DECISION_BODY_REVISED, active_mode="structuring"),
        ],
        critique=[{
            "score": 0.8,
            "findings": ["No stated rollback path if row locking causes contention"],
            "suggestions": ["Document the rollback path"],
        }],
    )
    return Scenario(
        name="behavior-sad-tech-decision",
        artifact_type="tech_decision",
        llm=llm,
        actions=[
            {"type": "send", "content": "Can chot quyet dinh ky thuat ve locking khi dat lich dong thoi."},
            _CONTINUE,
            {"type": "reject_all"},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


BEHAVIOR_SCENARIOS = [brd_happy_path, brd_ambiguous, prd_from_brd, sad_tech_decision]
