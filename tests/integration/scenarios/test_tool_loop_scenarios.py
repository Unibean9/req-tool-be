"""Tool-loop scenarios end-to-end through the HTTP driver (flag on).

Proves the shim runs a full conversation: analyze emits tool_calls, the ToolNode dispatches, the
HITL interrupt/resume round-trips, an approved write_draft becomes an artifact, and an exhausted
tool-brain (no tool picked) ends the turn cleanly. The enum ALL_SCENARIOS keep running the enum
path (flag off); these run only with tool_loop_only monkeypatched on.
"""

import uuid

import pytest

from tests.integration.scenarios.driver import Scenario, ScenarioDriver
from tests.integration.scenarios.scripted_llm import ScriptedLLM, tool_select

pytestmark = pytest.mark.asyncio

_GOAL_BODY = (
    "## Scope\n"
    "MVP tập trung vào nhóm sinh viên cần tìm khung giờ học chung trong tuần.\n\n"
    "## Capabilities\n"
    "| capability | priority | rationale | dependency |\n"
    "| --- | --- | --- | --- |\n"
    "| Tạo nhóm học | Must | Có danh sách thành viên để đối chiếu lịch | Tài khoản người dùng |\n"
    "| Đồng bộ lịch cá nhân | Must | Xác định khung bận/rảnh | Tích hợp Google Calendar |\n"
    "| Gợi ý khung giờ chung | Must | Giảm thời gian điều phối | Dữ liệu lịch |\n\n"
    "## Out of Scope\n"
    "- Thanh toán, quản lý điểm danh nâng cao và phân tích học tập dài hạn."
)


def _analysis_frame_turn():
    return tool_select(
        "analysis_frame",
        interpreted_intent="Đặt mục tiêu cho công cụ điều phối lịch học nhóm của sinh viên.",
        evidence=[
            "Đối tượng chính là sinh viên đại học học theo nhóm.",
            "Pain chính là trùng lịch và quên buổi học nhóm.",
        ],
        gaps=["Chưa có target định lượng cho success metric."],
        analysis_angles=["Đối tượng", "Pain", "MVP scope", "Success metric"],
        assumptions=["Có thể draft mục tiêu với metric cần xác nhận."],
        recommended_next_move="Tạo draft mục tiêu và đánh dấu metric cần user xác nhận.",
        active_mode="structuring",
    )


# ---------------------------------------------------------------------------
# T5 — HITL round-trip + artifact creation through the tool-loop
# ---------------------------------------------------------------------------

async def test_tool_loop_ask_then_draft_approve(client, scenario_env, scenario_project):
    """ask_user → confirm_intent → analysis_frame → write_draft approval → completed."""
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        tool_select("ask_user", message="Đối tượng người dùng chính là ai?", active_mode="discovery"),
        tool_select("confirm_intent",
                    summary="Đặt mục tiêu đo lường được cho công cụ điều phối lịch học nhóm của sinh viên.",
                    active_mode="discovery"),
        _analysis_frame_turn(),
        tool_select("write_draft", title="Mục tiêu: điều phối lịch học nhóm", body=_GOAL_BODY, active_mode="structuring"),
    ])
    scenario = Scenario(
        name="tool-loop-ask-draft-approve",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "Tôi muốn đặt mục tiêu cho sản phẩm điều phối lịch học nhóm."},
            {"type": "send", "content": "Chủ yếu là sinh viên đại học học theo nhóm."},
            {"type": "send", "content": "Đúng rồi, tiếp tục giúp tôi."},
            {"type": "send", "content": "Frame ổn, draft đi."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    recorder = await driver.run()

    assert recorder.summary["final_status"] == "completed"
    artifacts = await driver.executed_artifacts()
    assert len(artifacts) >= 1
    assert artifacts[0]["body"] == _GOAL_BODY

    # No duplicate agent question on the resume path (R1 idempotency holds through the tool path).
    final_msgs = recorder.steps[-1]["snapshot"]["messages"]
    questions = [m for m in final_msgs if (m.get("payload") or {}).get("kind") == "question"]
    assert len(questions) == 1


async def test_tool_loop_reject_draft(client, scenario_env, scenario_project):
    """Rejecting the proposed write_draft creates no artifact; the turn still completes."""
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        tool_select("confirm_intent",
                    summary="Đặt mục tiêu cho công cụ điều phối lịch học nhóm của sinh viên.",
                    active_mode="discovery"),
        _analysis_frame_turn(),
        tool_select("write_draft", title="Mục tiêu (bản nháp)", body=_GOAL_BODY, active_mode="structuring"),
    ])
    scenario = Scenario(
        name="tool-loop-reject-draft",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "Đặt mục tiêu cho sản phẩm điều phối lịch học nhóm."},
            {"type": "send", "content": "Đúng rồi, tiếp tục giúp tôi."},
            {"type": "send", "content": "Frame ổn, gửi draft."},
            {"type": "reject_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 0},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    recorder = await driver.run()

    assert recorder.summary["final_status"] == "completed"
    assert len(await driver.executed_artifacts()) == 0


async def test_tool_loop_composite_two_note_tools(client, scenario_env, scenario_project):
    """Composite dispatch: brain returns [explore_note, critique_note] in one turn.

    Proves that a single analyze_node cycle emits an AIMessage with 2 tool_calls and
    the ToolNode dispatches both — resulting in 2 ToolMessages in the checkpoint before
    the next turn. The session must stay ACTIVE after both notes execute (no interrupt).
    """
    from langchain_core.messages import AIMessage, ToolMessage

    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        # Turn 1: composite — two non-interrupt note tools in the same turn.
        {
            "tools": [
                {"name": "explore_note", "args": {"content": "Người dùng chính là sinh viên nhóm 4-6 người."}},
                {"name": "critique_note", "args": {"content": "Cần đo lường: tỉ lệ tham gia buổi nhóm."}},
            ],
            "active_mode": "discovery",
        },
        # Turn 2: confirm intent after notes.
        tool_select("confirm_intent",
                    summary="Điều phối lịch học nhóm cho sinh viên, đo bằng tỉ lệ tham gia.",
                    active_mode="discovery"),
        # Turn 3: visible analysis frame, then draft.
        _analysis_frame_turn(),
        tool_select("write_draft", title="Mục tiêu: điều phối lịch học nhóm", body=_GOAL_BODY,
                    active_mode="structuring"),
    ])
    scenario = Scenario(
        name="tool-loop-composite-notes",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "Tôi muốn đặt mục tiêu cho sản phẩm điều phối lịch học nhóm."},
            {"type": "send", "content": "Đúng rồi, tiếp tục."},
            {"type": "send", "content": "Frame đúng, viết draft."},
            {"type": "approve_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 1},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    recorder = await driver.run()

    assert recorder.summary["final_status"] == "completed"
    assert len(await driver.executed_artifacts()) >= 1

    # Verify composite dispatch in checkpoint: find the AIMessage that carried 2 tool_calls.
    raw_msgs = await scenario_env.get_checkpoint_field(driver.session_id, "messages")
    ai_msgs_with_two_calls = [
        m for m in (raw_msgs or [])
        if isinstance(m, AIMessage) and len(getattr(m, "tool_calls", [])) == 2
    ]
    assert ai_msgs_with_two_calls, "No AIMessage with 2 tool_calls found — composite dispatch did not fire"

    composite_ai = ai_msgs_with_two_calls[0]
    dispatched_names = [tc["name"] for tc in composite_ai.tool_calls]
    assert set(dispatched_names) == {"explore_note", "critique_note"}, (
        f"unexpected composite tools: {dispatched_names}"
    )

    # Both ToolMessages must follow the composite AIMessage.
    call_ids = {tc["id"] for tc in composite_ai.tool_calls}
    tool_msgs = [
        m for m in (raw_msgs or [])
        if isinstance(m, ToolMessage) and m.tool_call_id in call_ids
    ]
    assert len(tool_msgs) == 2, (
        f"expected 2 ToolMessages for composite dispatch, got {len(tool_msgs)}"
    )
