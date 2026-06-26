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
    "Sản phẩm nhắm tới các nhóm sinh viên cần một công cụ quản lý lịch học chung để giảm xung đột lịch.\n\n"
    "## Objectives\n"
    "- Tạo nhóm học và đồng bộ lịch cá nhân.\n"
    "- Gợi ý khung giờ rảnh chung cho cả nhóm.\n"
    "- Tăng tỉ lệ tham gia buổi học nhóm.\n\n"
    "## Success Metrics\n"
    "- Giảm thời gian điều phối từ 30 phút xuống dưới 10 phút mỗi tuần trong 3 tháng."
)

_PROBLEM_BODY = (
    "## Problem Statement\n"
    "Sinh viên hiện sắp lịch học nhóm thủ công qua chat, dẫn tới trùng lịch và bỏ buổi.\n\n"
    "## Affected Users\n"
    "Nhóm sinh viên 4-6 người và trưởng nhóm chịu trách nhiệm chốt lịch.\n\n"
    "## Impact\n"
    "Mỗi tuần mỗi nhóm mất khoảng 30 phút điều phối; tỉ lệ tham gia buổi nhóm dưới 60%.\n\n"
    "## Root Cause / Contributing Factors\n"
    "Lịch cá nhân phân tán, thiếu cách tính giao khung giờ rảnh và thiếu xác nhận tập trung."
)

_STAKEHOLDER_BODY = (
    "## Stakeholders\n"
    "| role | responsibility | decision authority | needs/concerns | involvement |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| Trưởng nhóm | Tạo nhóm và chốt buổi học | Cao | Cần lịch chung nhanh | Hằng tuần |\n"
    "| Thành viên | Đồng bộ lịch cá nhân | Trung bình | Chỉ chia sẻ trạng thái rảnh/bận | Hằng tuần |\n"
    "| Giảng viên | Theo dõi tiến độ nhóm | Thấp | Muốn nhóm duy trì nhịp học | Theo đợt |"
)

# Goal must carry a metric + time-bound to satisfy the SMART validator heuristic.
_GOAL_BODY = (
    "## Scope\n"
    "MVP tập trung vào nhóm sinh viên cần tìm khung giờ học chung trong tuần.\n\n"
    "## Capabilities\n"
    "| capability | priority | rationale | dependency |\n"
    "| --- | --- | --- | --- |\n"
    "| Tạo nhóm học | Must | Có danh sách thành viên để đối chiếu lịch | Tài khoản người dùng |\n"
    "| Đồng bộ lịch cá nhân | Must | Xác định khung bận/rảnh | Tích hợp Google Calendar |\n"
    "| Gợi ý khung giờ chung | Must | Giảm thời gian điều phối từ 30 phút xuống dưới 10 phút | Dữ liệu lịch |\n\n"
    "## Out of Scope\n"
    "- Thanh toán, quản lý điểm danh nâng cao và phân tích học tập dài hạn."
)

# Functional requirement — concrete, measurable behavior.
_FR_BODY = (
    "## Functional Requirement\n"
    "Hệ thống phải cho phép thành viên kết nối lịch cá nhân và trích xuất các khung giờ bận.\n\n"
    "## Behavior\n"
    "Khi trưởng nhóm yêu cầu, hệ thống tính giao các khung rảnh và trả về danh sách khung giờ chung.\n\n"
    "## Inputs and Outputs\n"
    "- Input: lịch cá nhân, danh sách thành viên, khoảng thời gian cần tìm.\n"
    "- Output: khung giờ chung kèm số thành viên rảnh tương ứng.\n\n"
    "## Acceptance Signals\n"
    "- Kết quả cập nhật trong vòng 5 giây sau khi có thay đổi lịch."
)

# Non-functional requirement — quality attributes with thresholds.
_NFR_BODY = (
    "## Quality Attribute\n"
    "Hiệu năng và bảo mật dữ liệu lịch cá nhân.\n\n"
    "## Requirement\n"
    "Tính khung giờ chung cho nhóm tối đa 8 thành viên phải phản hồi dưới 2 giây ở p95.\n\n"
    "## Measurement\n"
    "- Load test 500 nhóm hoạt động đồng thời, CPU không vượt 70%.\n"
    "- Kiểm tra dữ liệu lịch được mã hóa khi lưu.\n\n"
    "## Scope and Tradeoffs\n"
    "Ưu tiên phản hồi nhanh cho nhóm nhỏ; báo lỗi rõ nếu nhóm vượt giới hạn MVP."
)

# Epic must carry INVEST "testable" signals such as acceptance criteria / Given-When-Then.
_EPIC_BODY = (
    "## Use Case\n"
    "Đồng bộ và đối chiếu lịch nhóm để gợi ý buổi học.\n\n"
    "## Actors\n"
    "- Trưởng nhóm\n"
    "- Thành viên nhóm\n\n"
    "## Preconditions\n"
    "Các thành viên đã tham gia nhóm và có thể kết nối lịch cá nhân.\n\n"
    "## Main Flow\n"
    "1. Thành viên đồng bộ lịch.\n"
    "2. Trưởng nhóm yêu cầu tìm khung giờ chung.\n"
    "3. Hệ thống trả danh sách khung giờ khả dụng.\n"
    "4. Trưởng nhóm chốt buổi học.\n\n"
    "## Alternate / Exception Flows\n"
    "- Nếu không có khung giờ chung, hệ thống gợi ý khung gần nhất kèm thành viên vắng.\n\n"
    "## Postconditions\n"
    "Buổi học được chốt hoặc có lý do rõ vì sao chưa thể chốt."
)

# Story must carry INVEST "testable" signals such as acceptance criteria / Given-When-Then.
_STORY_BODY = (
    "## Acceptance Criteria\n"
    "Là trưởng nhóm, tôi muốn xem khung giờ rảnh chung của cả nhóm để chốt buổi học mà không phải hỏi từng người.\n\n"
    "- Khi tất cả thành viên đã đồng bộ lịch, thì hệ thống hiển thị tối thiểu 3 khung giờ trùng rảnh trong tuần.\n"
    "- Khi không có khung giờ chung, thì hệ thống gợi ý khung gần nhất kèm số thành viên vắng."
)


# Every session opens in the intent phase (user_confirmed is None); confirm_intent opens the
# artifact phase before the first draft.
def _confirm_turn():
    return tool_select(
        "confirm_intent",
        summary="Xây công cụ điều phối lịch học nhóm cho sinh viên, ưu tiên tìm khung giờ rảnh chung.",
        active_mode="discovery",
    )


_CONTINUE = {"type": "send", "content": "Đúng rồi, tiếp tục giúp tôi."}


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
        tool_select("write_draft", title=title, body=body, active_mode="structuring"),
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
            tool_select("ask_user", message="Đối tượng người dùng chính của công cụ là ai?", active_mode="discovery"),
            tool_select("ask_user", message="Vấn đề lớn nhất họ đang gặp khi điều phối lịch là gì?",
                        active_mode="discovery", acknowledgment="Rõ rồi, là sinh viên."),
            _confirm_turn(),
            tool_select("write_draft", title="Intent: Điều phối lịch học nhóm cho sinh viên",
                        body=_INTENT_BODY, active_mode="structuring"),
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
            tool_select("write_draft", title="Intent (bản nháp)", body=_INTENT_BODY, active_mode="structuring"),
            tool_select("ask_user", active_mode="structuring",
                        message="Bạn muốn khám phá thêm khía cạnh nào — đối tượng, phạm vi hay giá trị mang lại?"),
            tool_select("write_draft", title="Intent: Điều phối lịch học nhóm (đã làm rõ phạm vi)",
                        body=_INTENT_BODY, active_mode="structuring"),
        ]
    )
    return _scenario(
        "reject-then-explore",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Tôi muốn tạo intent cho sản phẩm điều phối lịch học nhóm."},
            _CONTINUE,
            {"type": "reject_all"},
            {"type": "send", "content": "Hãy làm rõ phạm vi MVP giúp tôi."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )


def reject_proposal() -> Scenario:
    """User rejects the proposed draft at the approval gate — no artifacts created."""
    llm = ScriptedLLM(
        tool_brain=[
            _confirm_turn(),
            tool_select("write_draft", title="Intent: bản đề xuất", body=_INTENT_BODY, active_mode="structuring"),
        ]
    )
    return _scenario(
        "reject-proposal",
        "intent",
        llm,
        actions=[
            {"type": "send", "content": "Tôi muốn tạo intent cho sản phẩm điều phối lịch."},
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
                title="Vấn đề: Điều phối lịch học nhóm thủ công",
                body=_PROBLEM_BODY,
                active_mode="structuring",
            ),
        ]
    )
    return _scenario(
        "problem-propose-approve",
        "problem",
        llm,
        actions=[
            {"type": "send", "content": "Vấn đề là sinh viên sắp lịch học nhóm thủ công nên hay trùng lịch."},
            _CONTINUE,
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
