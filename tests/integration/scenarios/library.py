"""Behavior-scenario definitions.

Each scenario scripts the analyst tool-loop "brain" (ordered tool-selection turns) and the user's
side of the conversation (the actions). Names are kebab-cased and used as the transcript filename.
User-facing strings are Vietnamese on purpose — they are the real conversational content the agent
sees.

Tool-loop flow: confirm_intent opens artifact work, then write_draft proposes to the approval gate.
ask_user pauses for the human between turns; reject_all declines a proposed draft.
"""

from tests.integration.scenarios.driver import Scenario
from tests.integration.scenarios.scripted_llm import ScriptedLLM, tool_select

# A realistic intent artifact body — gives the judge something substantive to score.
_INTENT_BODY = (
    "## Vision\n"
    "The product targets student groups that need a shared study scheduling tool to reduce schedule conflicts.\n\n"
    "## Objectives\n"
    "- Create study group va dong bo lich ca nhan.\n"
    "- Goi y khung gio ranh chung cho ca nhom.\n"
    "- Tang ti le tham gia buoi hoc nhom.\n\n"
    "## Success Metrics\n"
    "- Reduce coordination time from 30 minutes to under 10 minutes per week within 3 months."
)

_INTENT_BODY_CLARIFIED = (
    "## Vision\n"
    "The MVP serves university study groups of 4-6 members who lose study sessions to scheduling conflicts.\n\n"
    "## Objectives\n"
    "- Sync each member's calendar into a shared free/busy view.\n"
    "- Suggest at least 3 common free slots per group each week.\n"
    "- Keep the first release focused on students, not lecturers.\n\n"
    "## Success Metrics\n"
    "- Group session attendance rises from 60% to 80% within one semester."
)

_PROBLEM_BODY = (
    "## Problem Statement\n"
    "Sinh vien hien sap study scheduling thu cong qua chat, dan toi trung lich va bo buoi.\n\n"
    "## Affected Users\n"
    "Student groups of 4-6 people and group leaders responsible for finalizing schedules.\n\n"
    "## Impact\n"
    "Moi tuan moi nhom mat khoang 30 phut orchestration; ti le tham gia buoi nhom duoi 60%.\n\n"
    "## Root Cause / Contributing Factors\n"
    "Personal calendars are fragmented, with no way to compute overlapping free slots and no centralized confirmation."
)

_STAKEHOLDER_BODY = (
    "## Stakeholders\n"
    "| role | responsibility | decision authority | needs/concerns | involvement |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| Truong nhom | Tao nhom va chot buoi hoc | Cao | Can lich chung nhanh | Weekly |\n"
    "| Member | Sync personal calendar | Medium | Only share free/busy status | Weekly |\n"
    "| Giang vien | Theo doi tien do nhom | Thap | Muon nhom duy tri nhip hoc | Theo dot |"
)

# Goal must carry a metric + time-bound to satisfy the SMART validator heuristic.
_GOAL_BODY = (
    "## Scope\n"
    "MVP focuses on student groups that need to find shared study times during the week.\n\n"
    "## Capabilities\n"
    "| capability | priority | rationale | dependency |\n"
    "| --- | --- | --- | --- |\n"
    "| Create study group | Must | Has a member list for calendar comparison | User account |\n"
    "| Sync personal calendar | Must | Identify busy/free slots | Google Calendar integration |\n"
    "| Suggest common time slots | Must | Reduce coordination time tu 30 phut xuong duoi 10 phut | Calendar data |\n\n"
    "## Out of Scope\n"
    "- Payments, advanced attendance management, and long-term learning analytics."
)

# Functional requirement — concrete, measurable behavior in the id-tagged table form.
_FR_BODY = (
    "## Functional Requirements\n"
    "| id | requirement | behavior | inputs/outputs | acceptance signal | priority |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
    "| FR-01 | Members connect personal calendars | System extracts busy slots and computes overlaps "
    "when the group leader requests it | Input: calendars, member list, target range; "
    "Output: khung gio chung kem so thanh vien ranh | Ket qua cap nhat trong vong 5 giay sau khi doi lich | Must |"
)

# Non-functional requirement — quality attributes with thresholds, id-tagged table form.
_NFR_BODY = (
    "## Non-Functional Requirements\n"
    "| id | quality attribute | requirement | measurement | scope/tradeoff |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| NFR-01 | Performance | Compute common slots for up to 8 members with p95 under 2s | "
    "Load test 500 concurrent active groups with CPU under 70% | Prioritize fast responses for small groups |\n"
    "| NFR-02 | Security | Encrypt personal calendar data at rest | Verify encryption in a storage audit | "
    "Clear errors when a group exceeds MVP limits |"
)

# Epic maps to use_case (driver), now framed as a Business Capability — brief, no flow detail.
_EPIC_BODY = (
    "## Business Capabilities\n"
    "### BC-01: Dong bo va doi chieu lich nhom\n"
    "- **goal:** Tim khung gio ranh chung de nhom chot buoi hoc nhanh hon\n"
    "- **users:** Truong nhom va member nhom\n"
    "- **value:** Giam thoi gian orchestration tu 30 phut xuong duoi 10 phut moi tuan\n"
    "- **scope:** Dong bo lich ca nhan va goi y khung gio chung; chi tiet luong de o user stories"
)

# Story must carry INVEST "testable" signals such as acceptance criteria / Given-When-Then.
_STORY_BODY = (
    "## Acceptance Criteria\n"
    "As a group leader, I want to view common free slots for the whole group so I can finalize a session without asking each person.\n\n"
    "- When all members sync calendars, then the system shows at least 3 overlapping free slots in the week.\n"
    "- When there is no common slot, then the system suggests the nearest slot with the number of absent members."
)


# Every session opens in the intent phase (user_confirmed is None); confirm_intent opens the
# artifact phase before the first draft.
def _confirm_turn():
    return tool_select(
        "confirm_intent",
        summary="Xay cong cu orchestration study scheduling cho sinh vien, uu tien tim khung gio ranh chung.",
    )


_CONTINUE = {"type": "send", "content": "Dung roi, tiep tuc giup toi."}


def _scenario(name: str, artifact_type: str, llm: ScriptedLLM, actions, expect) -> Scenario:
    return Scenario(name=name, artifact_type=artifact_type, llm=llm, actions=actions, expect=expect)


def _draft_approve(
    name: str, artifact_type: str, title: str, body: str, opening: str
) -> Scenario:
    """Build a happy-path scenario: open → write_draft proposes → approve → completed.

    Used to give every artifact type one self-contained behavior scenario whose
    produced artifact the judge can score.
    """
    llm = ScriptedLLM(tool_brain=[
        _confirm_turn(),
        tool_select("write_draft", title=title, body=body),
    ])
    return _scenario(
        name,
        artifact_type,
        llm,
        actions=[
            {"type": "send", "content": opening},
            _CONTINUE,
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def clarify_draft_approve() -> Scenario:
    """Canonical journey: clarify context, draft once, then approve."""
    llm = ScriptedLLM(
        tool_brain=[
            tool_select("ask_user", message="Who is the primary user of the tool?"),
            _confirm_turn(),
            tool_select(
                "write_draft",
                title="Intent: study scheduling coordination",
                body=_INTENT_BODY,
            ),
        ]
    )
    return _scenario(
        "canonical-clarify-draft-approve",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Toi muon xay cong cu giup sinh vien orchestration study scheduling."},
            {"type": "send", "content": "Mainly university students studying in groups."},
            _CONTINUE,
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def reject_critique_redraft() -> Scenario:
    """Canonical journey: ambiguous brief, rejected draft, critique, clarify, redraft."""
    llm = ScriptedLLM(
        tool_brain=[
            tool_select("ask_user", message="What problem should the product solve first?"),
            tool_select(
                "ask_user",
                message="You mentioned both lecturers and students as primary users - which one is it for the MVP?",
            ),
            _confirm_turn(),
            tool_select(
                "create_decision_node",
                node_id="D1",
                kind="objective",
                statement="Reduce schedule conflicts for study groups",
            ),
            tool_select("update_decision_node", node_id="D1", status="confirmed"),
            tool_select("write_draft", title="Intent: study scheduling draft", body=_INTENT_BODY),
            tool_select("run_critique", target="draft", mode="consistency"),
            tool_select(
                "ask_user",
                message="The draft still mixes primary users. Confirm: students only for the MVP?",
            ),
            tool_select(
                "write_draft",
                title="Intent: study scheduling for student MVP",
                body=_INTENT_BODY_CLARIFIED,
            ),
        ],
        critique=[
            {
                "score": 0.55,
                "findings": ["Objectives contradict the stated primary user"],
                "suggestions": ["Pick one primary user for the MVP and align objectives"],
            }
        ],
    )
    return _scenario(
        "canonical-reject-critique-redraft",
        "intent",
        llm,
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


def context_artifact_from_predecessors() -> Scenario:
    """Canonical journey: derive a downstream artifact from accepted predecessor context."""
    llm = ScriptedLLM(
        tool_brain=[
            _confirm_turn(),
            tool_select("write_draft", title="FR: compute common free slots", body=_FR_BODY),
        ]
    )
    return _scenario(
        "canonical-context-artifact-from-predecessors",
        "functional_requirement",
        llm,
        actions=[
            {"type": "send", "content": "BRD da chot roi, viet functional requirement cho tinh khung gio chung."},
            _CONTINUE,
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def intent_propose_approve() -> Scenario:
    """Happy path: first message → write_draft proposes → approve → completed."""
    return _draft_approve(
        "intent-propose-approve",
        "intent",
        "Intent: Cong cu orchestration study scheduling",
        _INTENT_BODY,
        "Toi muon xay mot cong cu giup sinh vien orchestration study scheduling.",
    )


def multi_turn_qna() -> Scenario:
    """Multi-turn Q&A (one-question rhythm) before drafting."""
    llm = ScriptedLLM(
        tool_brain=[
            tool_select("ask_user", message="Who is the primary user of the tool?"),
            tool_select("ask_user", message="What is their biggest scheduling coordination problem?",
                        acknowledgment="Ro roi, la sinh vien."),
            _confirm_turn(),
            tool_select("write_draft", title="Intent: Dieu phoi study scheduling cho sinh vien",
                        body=_INTENT_BODY),
        ]
    )
    return _scenario(
        "multi-turn-qna",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Toi co mot y tuong product nhung chua ro rang lam."},
            {"type": "send", "content": "Mainly university students studying in groups."},
            {"type": "send", "content": "Ho bi trung lich va hay bo buoi hoc nhom."},
            _CONTINUE,
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def reject_then_explore() -> Scenario:
    """User rejects the first proposed draft (explore more), agent asks, then drafts again."""
    llm = ScriptedLLM(
        tool_brain=[
            _confirm_turn(),
            tool_select("write_draft", title="Intent (draft)", body=_INTENT_BODY),
            tool_select("ask_user",
                        message="Ban muon kham pha them khia canh nao — doi tuong, pham vi hay gia tri mang lai?"),
            tool_select("write_draft", title="Intent: Dieu phoi study scheduling (da lam ro pham vi)",
                        body=_INTENT_BODY),
        ]
    )
    return _scenario(
        "reject-then-explore",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Toi muon tao intent cho product orchestration study scheduling."},
            _CONTINUE,
            {"type": "reject_all"},
            {"type": "send", "content": "Hay lam ro pham vi MVP giup toi."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def reject_proposal() -> Scenario:
    """User rejects the proposed draft at the approval gate — no artifacts created."""
    llm = ScriptedLLM(
        tool_brain=[
            _confirm_turn(),
            tool_select("write_draft", title="Intent: proposal", body=_INTENT_BODY),
        ]
    )
    return _scenario(
        "reject-proposal",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Toi muon tao intent cho product orchestration lich."},
            _CONTINUE,
            {"type": "reject_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 0},
    )


def problem_propose_approve() -> Scenario:
    """Happy path for a `problem` artifact — used by the aggregated-document test."""
    llm = ScriptedLLM(
        tool_brain=[
            _confirm_turn(),
            tool_select(
                "write_draft",
                title="Van de: Dieu phoi study scheduling thu cong",
                body=_PROBLEM_BODY,
            ),
        ]
    )
    return _scenario(
        "problem-propose-approve",
        "problem",
        llm,
        actions=[
            {"type": "send", "content": "Van de la sinh vien sap study scheduling thu cong nen hay trung lich."},
            _CONTINUE,
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def stakeholder_propose_approve() -> Scenario:
    return _draft_approve(
        "stakeholder-propose-approve",
        "stakeholder",
        "Stakeholders: study group, group leader, lecturer",
        _STAKEHOLDER_BODY,
        "List stakeholders for the study scheduling tool.",
    )


def goal_propose_approve() -> Scenario:
    return _draft_approve(
        "goal-propose-approve",
        "goal",
        "Goal: reduce coordination time and increase attendance rate",
        _GOAL_BODY,
        "Set measurable goals for the study scheduling product.",
    )


def functional_requirement_propose_approve() -> Scenario:
    return _draft_approve(
        "functional-requirement-propose-approve",
        "functional_requirement",
        "Requirement chuc nang: tinh khung gio ranh chung cua nhom",
        _FR_BODY,
        "Write functional requirements for finding common free slots.",
    )


def non_functional_requirement_propose_approve() -> Scenario:
    return _draft_approve(
        "non-functional-requirement-propose-approve",
        "non_functional_requirement",
        "Requirement phi chuc nang: hieu nang va bao mat lich nhom",
        _NFR_BODY,
        "State non-functional performance and security requirements for the system.",
    )


def epic_propose_approve() -> Scenario:
    return _draft_approve(
        "epic-propose-approve",
        "epic",
        "Epic: Dong bo va doi chieu lich nhom",
        _EPIC_BODY,
        "Gom cac tinh nang lich nhom thanh mot epic giup toi.",
    )


def story_propose_approve() -> Scenario:
    return _draft_approve(
        "story-propose-approve",
        "story",
        "Story: truong nhom xem khung gio ranh chung",
        _STORY_BODY,
        "Viet user story cho viec truong nhom xem khung gio ranh chung.",
    )


# Ordered registry used by integration, eval, live smoke, and benchmark lanes.
# Keep this small: high-level tests are canaries, not an artifact-type matrix.
CANONICAL_SCENARIOS = [
    clarify_draft_approve,
    reject_critique_redraft,
    context_artifact_from_predecessors,
]

ALL_SCENARIOS = CANONICAL_SCENARIOS

# Reduced topological document smoke. The exhaustive artifact-type matrix was
# high-maintenance and overlapped with lower-level schema/validator tests.
DOCUMENT_PIPELINE = [
    clarify_draft_approve,
    problem_propose_approve,
    context_artifact_from_predecessors,
]
