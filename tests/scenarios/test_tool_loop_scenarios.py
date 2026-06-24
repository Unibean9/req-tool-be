"""Phase 5 Slice B — tool-loop scenarios end-to-end through the HTTP driver (flag on).

Proves the shim runs a full conversation: analyze emits tool_calls, the ToolNode dispatches, the
HITL interrupt/resume round-trips, an approved write_draft becomes an artifact, and an exhausted
tool-brain (no tool picked) ends the turn cleanly. The enum ALL_SCENARIOS keep running the enum
path (flag off); these run only with tool_loop_only monkeypatched on.
"""

import uuid

import pytest

from tests.scenarios.driver import Scenario, ScenarioDriver
from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

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


# ---------------------------------------------------------------------------
# T5 — HITL round-trip + artifact creation through the tool-loop
# ---------------------------------------------------------------------------

async def test_tool_loop_ask_then_draft_approve(client, scenario_env, scenario_project):
    """ask_user (interrupt/resume) → write_draft (approve → artifact) → exhausted brain ends turn."""
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])

    llm = ScriptedLLM(tool_brain=[
        tool_select("ask_user", message="Đối tượng người dùng chính là ai?", active_mode="discovery"),
        tool_select("write_draft", title="Mục tiêu: điều phối lịch học nhóm", body=_GOAL_BODY, active_mode="structuring"),
    ])
    scenario = Scenario(
        name="tool-loop-ask-draft-approve",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "Tôi muốn đặt mục tiêu cho sản phẩm điều phối lịch học nhóm."},
            {"type": "send", "content": "Chủ yếu là sinh viên đại học học theo nhóm."},
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
        tool_select("write_draft", title="Mục tiêu (bản nháp)", body=_GOAL_BODY, active_mode="structuring"),
    ])
    scenario = Scenario(
        name="tool-loop-reject-draft",
        artifact_type="goal",
        llm=llm,
        actions=[
            {"type": "send", "content": "Đặt mục tiêu cho sản phẩm điều phối lịch học nhóm."},
            {"type": "reject_all"},
        ],
        expect={"final_status": "completed", "min_artifacts": 0},
    )
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
    recorder = await driver.run()

    assert recorder.summary["final_status"] == "completed"
    assert len(await driver.executed_artifacts()) == 0
