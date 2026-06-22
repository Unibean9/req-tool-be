"""Behavior-scenario definitions.

Each scenario scripts the analyst tool-loop "brain" (ordered tool-selection turns) and the user's
side of the conversation (the actions). Names are kebab-cased and used as the transcript filename.
User-facing strings are Vietnamese on purpose — they are the real conversational content the agent
sees.

Tool-loop flow: write_draft proposes a draft straight to the approval gate (no separate confirm
step), so a happy path is simply `send opening → approve_all`. ask_user pauses for the human between
turns; reject_all declines a proposed draft.
"""

from tests.scenarios.driver import Scenario
from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

# A realistic intent artifact body — gives the judge something substantive to score.
_INTENT_BODY = (
    "Sản phẩm nhắm tới các nhóm sinh viên cần một công cụ quản lý lịch học chung. "
    "Mục đích: giảm xung đột lịch và tăng tỉ lệ tham gia buổi học nhóm. "
    "Phạm vi MVP: tạo nhóm, đồng bộ lịch cá nhân, gợi ý khung giờ rảnh chung."
)

_PROBLEM_BODY = (
    "Sinh viên hiện sắp lịch học nhóm thủ công qua chat, dẫn tới trùng lịch và bỏ buổi. "
    "Tần suất: mỗi tuần mỗi nhóm mất ~30 phút điều phối. "
    "Tác động: tỉ lệ tham gia buổi nhóm dưới 60%."
)

_PROBLEM_FULL_SECTION_ASSESSMENT = {
    "vision_objectives": "filled",
    "problem_statement": "filled",
    "stakeholder_register": "filled",
    "scope_capabilities": "filled",
    "business_rules": "filled",
    "constraints_assumptions": "filled",
    "risks_issues": "filled",
}

_STAKEHOLDER_BODY = (
    "Các bên liên quan chính: (1) Trưởng nhóm — tạo nhóm và chốt buổi học; "
    "(2) Thành viên — đồng bộ lịch cá nhân và xác nhận khung giờ; "
    "(3) Giảng viên — theo dõi tiến độ buổi nhóm. "
    "Mỗi nhóm có 4–6 thành viên. Trưởng nhóm cần quyền chỉnh lịch chung, "
    "thành viên chỉ xem và xác nhận khung giờ rảnh của mình."
)

# Goal must carry a metric + time-bound to satisfy the SMART validator heuristic.
_GOAL_BODY = (
    "Trong vòng 3 tháng sau khi ra mắt MVP, giảm thời gian điều phối lịch học nhóm "
    "từ 30 phút xuống dưới 10 phút mỗi tuần, và nâng tỉ lệ tham gia buổi nhóm "
    "từ 60% lên 80%. Đo bằng log thao tác trong hệ thống và khảo sát cuối kỳ."
)

# Functional requirement — concrete, measurable behavior.
_FR_BODY = (
    "Hệ thống phải cho phép thành viên kết nối lịch cá nhân (Google Calendar) và "
    "trích xuất các khung giờ bận. Khi trưởng nhóm yêu cầu, hệ thống tính giao của "
    "các khung rảnh và trả về danh sách khung giờ chung kèm số thành viên rảnh tương ứng. "
    "Kết quả cập nhật lại trong vòng 5 giây sau khi có thay đổi lịch."
)

# Non-functional requirement — quality attributes with thresholds.
_NFR_BODY = (
    "Thời gian phản hồi khi tính khung giờ chung cho nhóm tối đa 8 thành viên phải dưới "
    "2 giây ở phân vị 95. Hệ thống phục vụ đồng thời 500 nhóm hoạt động mà không vượt 70% CPU. "
    "Dữ liệu lịch cá nhân được mã hoá khi lưu và chỉ trưởng nhóm xem được bản tổng hợp."
)

# Epic must carry INVEST "testable" signals such as acceptance criteria / Given-When-Then.
_EPIC_BODY = (
    "Đồng bộ và đối chiếu lịch nhóm: kết nối lịch cá nhân, phát hiện khung giờ trùng rảnh, "
    "gợi ý buổi học. Tiêu chí hoàn thành: khi một nhóm 5 thành viên đồng bộ lịch, "
    "thì hệ thống tính được khung giờ chung trong dưới 5 giây và cho phép trưởng nhóm chốt buổi."
)

# Story must carry INVEST "testable" signals such as acceptance criteria / Given-When-Then.
_STORY_BODY = (
    "Là trưởng nhóm, tôi muốn xem khung giờ rảnh chung của cả nhóm để chốt buổi học "
    "mà không phải hỏi từng người.\n"
    "Tiêu chí chấp nhận (acceptance):\n"
    "- Khi tất cả thành viên đã đồng bộ lịch, thì hệ thống hiển thị tối thiểu 3 khung giờ trùng rảnh trong tuần.\n"
    "- Khi không có khung giờ chung, thì hệ thống gợi ý khung gần nhất kèm số thành viên vắng."
)


def _scenario(name: str, artifact_type: str, llm: ScriptedLLM, actions, expect) -> Scenario:
    return Scenario(name=name, artifact_type=artifact_type, llm=llm, actions=actions, expect=expect)


def _draft_approve(
    name: str, artifact_type: str, title: str, body: str, opening: str
) -> Scenario:
    """Build a happy-path scenario: open → write_draft proposes → approve → completed.

    Used to give every artifact type one self-contained behavior scenario whose
    produced artifact the judge can score.
    """
    llm = ScriptedLLM(tool_brain=[tool_select("write_draft", title=title, body=body, active_mode="draft")])
    return _scenario(
        name,
        artifact_type,
        llm,
        actions=[
            {"type": "send", "content": opening},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def intent_propose_approve() -> Scenario:
    """Happy path: first message → write_draft proposes → approve → completed."""
    return _draft_approve(
        "intent-propose-approve",
        "intent",
        "Intent: Công cụ điều phối lịch học nhóm",
        _INTENT_BODY,
        "Tôi muốn xây một công cụ giúp sinh viên điều phối lịch học nhóm.",
    )


def multi_turn_qna() -> Scenario:
    """Multi-turn Q&A (one-question rhythm) before drafting."""
    llm = ScriptedLLM(
        tool_brain=[
            tool_select("ask_user", message="Đối tượng người dùng chính của công cụ là ai?", active_mode="qa"),
            tool_select("ask_user", message="Vấn đề lớn nhất họ đang gặp khi điều phối lịch là gì?",
                        active_mode="qa", acknowledgment="Rõ rồi, là sinh viên."),
            tool_select("write_draft", title="Intent: Điều phối lịch học nhóm cho sinh viên",
                        body=_INTENT_BODY, active_mode="draft"),
        ]
    )
    return _scenario(
        "multi-turn-qna",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Tôi có một ý tưởng sản phẩm nhưng chưa rõ ràng lắm."},
            {"type": "send", "content": "Chủ yếu là sinh viên đại học học theo nhóm."},
            {"type": "send", "content": "Họ bị trùng lịch và hay bỏ buổi học nhóm."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def reject_then_explore() -> Scenario:
    """User rejects the first proposed draft (explore more), agent asks, then drafts again."""
    llm = ScriptedLLM(
        tool_brain=[
            tool_select("write_draft", title="Intent (bản nháp)", body=_INTENT_BODY, active_mode="draft"),
            tool_select("ask_user", active_mode="explore",
                        message="Bạn muốn khám phá thêm khía cạnh nào — đối tượng, phạm vi hay giá trị mang lại?"),
            tool_select("write_draft", title="Intent: Điều phối lịch học nhóm (đã làm rõ phạm vi)",
                        body=_INTENT_BODY, active_mode="draft"),
        ]
    )
    return _scenario(
        "reject-then-explore",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Tôi muốn tạo intent cho sản phẩm điều phối lịch học nhóm."},
            {"type": "reject_all"},
            {"type": "send", "content": "Hãy làm rõ phạm vi MVP giúp tôi."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def reject_proposal() -> Scenario:
    """User rejects the proposed draft at the approval gate — no artifacts created."""
    llm = ScriptedLLM(
        tool_brain=[tool_select("write_draft", title="Intent: bản đề xuất", body=_INTENT_BODY, active_mode="draft")]
    )
    return _scenario(
        "reject-proposal",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Tôi muốn tạo intent cho sản phẩm điều phối lịch."},
            {"type": "reject_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 0},
    )


def problem_propose_approve() -> Scenario:
    """Happy path for a `problem` artifact — used by the aggregated-document test."""
    llm = ScriptedLLM(
        tool_brain=[
            tool_select(
                "write_draft",
                title="Vấn đề: Điều phối lịch học nhóm thủ công",
                body=_PROBLEM_BODY,
                active_mode="draft",
                section_assessment=_PROBLEM_FULL_SECTION_ASSESSMENT,
            )
        ]
    )
    return _scenario(
        "problem-propose-approve",
        "problem",
        llm,
        actions=[
            {"type": "send", "content": "Vấn đề là sinh viên sắp lịch học nhóm thủ công nên hay trùng lịch."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def stakeholder_propose_approve() -> Scenario:
    return _draft_approve(
        "stakeholder-propose-approve",
        "stakeholder",
        "Các bên liên quan: nhóm học, trưởng nhóm, giảng viên",
        _STAKEHOLDER_BODY,
        "Liệt kê các bên liên quan cho công cụ điều phối lịch học nhóm giúp tôi.",
    )


def goal_propose_approve() -> Scenario:
    return _draft_approve(
        "goal-propose-approve",
        "goal",
        "Mục tiêu: giảm thời gian điều phối và tăng tỉ lệ tham gia",
        _GOAL_BODY,
        "Đặt mục tiêu đo lường được cho sản phẩm điều phối lịch học nhóm.",
    )


def functional_requirement_propose_approve() -> Scenario:
    return _draft_approve(
        "functional-requirement-propose-approve",
        "functional_requirement",
        "Yêu cầu chức năng: tính khung giờ rảnh chung của nhóm",
        _FR_BODY,
        "Viết yêu cầu chức năng cho tính năng tìm khung giờ rảnh chung.",
    )


def non_functional_requirement_propose_approve() -> Scenario:
    return _draft_approve(
        "non-functional-requirement-propose-approve",
        "non_functional_requirement",
        "Yêu cầu phi chức năng: hiệu năng và bảo mật lịch nhóm",
        _NFR_BODY,
        "Nêu các yêu cầu phi chức năng về hiệu năng và bảo mật cho hệ thống.",
    )


def epic_propose_approve() -> Scenario:
    return _draft_approve(
        "epic-propose-approve",
        "epic",
        "Epic: Đồng bộ và đối chiếu lịch nhóm",
        _EPIC_BODY,
        "Gom các tính năng lịch nhóm thành một epic giúp tôi.",
    )


def story_propose_approve() -> Scenario:
    return _draft_approve(
        "story-propose-approve",
        "story",
        "Story: trưởng nhóm xem khung giờ rảnh chung",
        _STORY_BODY,
        "Viết user story cho việc trưởng nhóm xem khung giờ rảnh chung.",
    )


# Ordered registry — used to parametrize the scenario test. Behavior scenarios
# (intent flows) first, then one happy-path per artifact type for output scoring.
ALL_SCENARIOS = [
    intent_propose_approve,
    multi_turn_qna,
    reject_then_explore,
    reject_proposal,
    problem_propose_approve,
    stakeholder_propose_approve,
    goal_propose_approve,
    functional_requirement_propose_approve,
    non_functional_requirement_propose_approve,
    epic_propose_approve,
    story_propose_approve,
]

# Topological pipeline order for the aggregated document (BA → PM). Predecessors
# are soft (recorded as missing_context, non-blocking); `capability` is omitted by
# design, so functional/non_functional requirements carry that soft warning.
DOCUMENT_PIPELINE = [
    intent_propose_approve,
    problem_propose_approve,
    stakeholder_propose_approve,
    goal_propose_approve,
    functional_requirement_propose_approve,
    non_functional_requirement_propose_approve,
    epic_propose_approve,
    story_propose_approve,
]
