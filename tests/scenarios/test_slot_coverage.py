import uuid

import pytest

from app.graphs.slot_schema import BRD_SLOTS, COVERAGE_THRESHOLD
from tests.scenarios.driver import Scenario, ScenarioDriver
from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

pytestmark = pytest.mark.asyncio


def _full(required: list[str]) -> dict[str, str]:
    """All required slots filled -> coverage complete."""
    return {slot: "filled" for slot in required}


_PARTIAL_PROBLEM = {
    "who": "filled",
    "obstacle": "filled",
    "root_cause": "partial",
    "frequency": "empty",
    "impact": "empty",
}

_FULL_PROBLEM = {
    "who": "filled",
    "obstacle": "filled",
    "root_cause": "filled",
    "frequency": "filled",
    "impact": "filled",
}


def _proposed_tool_calls(snapshot: dict) -> list[dict]:
    """write_draft proposals waiting at the approval gate in this snapshot."""
    return [tc for tc in snapshot["tool_calls"] if tc["status"] == "proposed"]


def _agent_messages(snapshot: dict) -> list[dict]:
    return [m for m in snapshot["messages"] if m["role"] == "agent"]


async def test_problem_proposes_after_coverage_met(client, scenario_env, scenario_project):
    """Ask clarifying questions via ask_user until coverage is met, then write_draft proposes to the approval gate.

    Coverage is still computed and saved to the checkpoint; write_draft carries a full slot_assessment so
    coverage_ratio meets the problem threshold by the time it proposes.
    """
    headers, project = scenario_project
    scenario = Scenario(
        name="problem-proposes-after-coverage-met",
        artifact_type="problem",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("ask_user", message="Ai là người bị ảnh hưởng trực tiếp?",
                            active_mode="qa", slot_assessment={}),
                tool_select("ask_user", message="Vấn đề xảy ra thường xuyên như thế nào?",
                            active_mode="qa", acknowledgment="Đã rõ đối tượng và trở ngại.",
                            slot_assessment=_PARTIAL_PROBLEM),
                tool_select("write_draft", title="Vấn đề: đăng ký lớp bị kẹt",
                            body="Sinh viên bị kẹt khi đăng ký lớp do rule lớp tiên quyết không rõ, "
                                 "xảy ra hằng tuần và làm trễ kế hoạch học.",
                            active_mode="draft", slot_assessment=_FULL_PROBLEM),
            ]
        ),
        actions=[
            {"type": "send", "content": "Tôi cần làm rõ vấn đề đăng ký lớp."},
            {"type": "send", "content": "Sinh viên năm nhất bị kẹt ở bước chọn lớp."},
            {"type": "send", "content": "Xảy ra mỗi tuần, ảnh hưởng tiến độ học và do rule tiên quyết không rõ."},
        ],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()
    coverage_ratio = await scenario_env.get_checkpoint_field(driver.session_id, "coverage_ratio")

    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "propose_artifacts"
    assert coverage_ratio >= COVERAGE_THRESHOLD["problem"]
    assert _proposed_tool_calls(recorder.steps[-1]["snapshot"])


async def test_one_question_per_turn_preserved(client, scenario_env, scenario_project):
    """ask_user still keeps the pace of one question per turn."""
    headers, project = scenario_project
    scenario = Scenario(
        name="problem-one-question-per-turn-preserved",
        artifact_type="problem",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("ask_user", message="Ai là người bị ảnh hưởng trực tiếp?",
                            active_mode="qa", slot_assessment={}),
                tool_select("ask_user", message="Tần suất xảy ra như thế nào?",
                            active_mode="qa", acknowledgment="Đã rõ đối tượng.",
                            slot_assessment=_PARTIAL_PROBLEM),
            ]
        ),
        actions=[
            {"type": "send", "content": "Tôi muốn làm rõ vấn đề đăng ký lớp."},
            {"type": "send", "content": "Sinh viên năm nhất bị ảnh hưởng trực tiếp."},
        ],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()
    agent_messages = _agent_messages(recorder.steps[-1]["snapshot"])

    assert agent_messages
    for message in agent_messages:
        if message["payload"]["kind"] == "question":
            assert message["content"].count("?") == 1


# ---------------------------------------------------------------------------
# Generalized per-key BRD coverage. `problem` is covered above; here we exercise the
# other eight keys: full slot_assessment -> coverage met -> write_draft proposes.
# ---------------------------------------------------------------------------

_OTHER_BRD_KEYS = [
    "intent",
    "goal",
    "stakeholder",
    "capability",
    "constraint",
    "assumption",
    "risk",
    "open_question",
]


@pytest.mark.parametrize("key", _OTHER_BRD_KEYS)
async def test_brd_key_coverage_then_propose(client, scenario_env, scenario_project, key):
    """Each BRD key: write_draft with full slots -> coverage meets threshold -> proposes to the approval gate."""
    headers, project = scenario_project
    required = BRD_SLOTS[key]["required"]

    scenario = Scenario(
        name=f"{key}-coverage-then-propose",
        artifact_type=key,
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("write_draft", title=f"{key} hoàn chỉnh", body=f"Nội dung đầy đủ cho {key}.",
                            active_mode="draft", slot_assessment=_full(required)),
            ]
        ),
        actions=[
            {"type": "send", "content": f"Đây là thông tin đầy đủ cho {key}."},
        ],
        expect={"min_coverage": COVERAGE_THRESHOLD[key]},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    # min_coverage is asserted inside driver.run; here we confirm it reached the proposal gate.
    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "propose_artifacts"
    assert _proposed_tool_calls(recorder.steps[-1]["snapshot"])


async def test_slot_coverage_does_not_gate_non_brd(client, scenario_env, scenario_project):
    """Non-BRD artifact: write_draft without slot_assessment -> coverage fail-open (None)."""
    headers, project = scenario_project
    scenario = Scenario(
        name="non-brd-fail-open",
        artifact_type="functional_requirement",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select(
                    "write_draft",
                    title="FR-1: Đăng ký lớp",
                    body="Hệ thống phải cho phép sinh viên đăng ký lớp trong thời gian mở đăng ký.",
                    active_mode="draft",
                ),
            ]
        ),
        actions=[{"type": "send", "content": "Tôi cần một yêu cầu chức năng cho đăng ký lớp."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()
    coverage_ratio = await scenario_env.get_checkpoint_field(driver.session_id, "coverage_ratio")

    # No slot_assessment reported -> coverage fail-open (None); the proposal still goes straight to the approval gate.
    assert coverage_ratio is None
    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "propose_artifacts"
    assert _proposed_tool_calls(recorder.steps[-1]["snapshot"])


# ---------------------------------------------------------------------------
# On-topic judge infrastructure (test-harness only schema).
# ---------------------------------------------------------------------------

async def test_m2_question_maps_to_slot():
    """Only exercises the judge infrastructure.

    The scripted judge defaults to on_topic=True regardless of the question, so this test only confirms the 'judge'
    route is wired correctly, NOT that the question is truly on-topic. A real check needs a live LLM judge outside CI.
    """
    from tests.scenarios.scripted_llm import ON_TOPIC_SCHEMA, ScriptedLLM

    llm = ScriptedLLM(tool_brain=[])
    result, _usage = await llm.generate(
        messages=[{"role": "user", "content": "Ai là người tài trợ sáng kiến này?"}],
        response_format=ON_TOPIC_SCHEMA,
    )

    assert result["on_topic"] is True
    assert llm.calls[-1]["route"] == "judge"


# ---------------------------------------------------------------------------
# Proposes when coverage is sufficient.
# ---------------------------------------------------------------------------

async def test_m3_proposes_when_coverage_sufficient(client, scenario_env, scenario_project):
    """When coverage meets the threshold, the agent write_drafts to the proposal gate (proposes at the right time)."""
    headers, project = scenario_project
    required = BRD_SLOTS["intent"]["required"]
    scenario = Scenario(
        name="m3-proposes-when-sufficient",
        artifact_type="intent",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("write_draft", title="Intent hoàn chỉnh", body="Nội dung đầy đủ cho intent.",
                            active_mode="draft", slot_assessment=_full(required)),
            ]
        ),
        actions=[{"type": "send", "content": "Đây là toàn bộ thông tin đầy đủ cho intent."}],
        expect={"min_coverage": COVERAGE_THRESHOLD["intent"]},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    assert recorder.summary["final_interrupt"] == "propose_artifacts"
    assert _proposed_tool_calls(recorder.steps[-1]["snapshot"])


# ---------------------------------------------------------------------------
# last_asked_slot does not repeat on two consecutive turns.
# ---------------------------------------------------------------------------

async def test_m1_last_asked_slot_no_consecutive_repeat(client, scenario_env, scenario_project):
    """Run a multi-turn HTTP conversation under chronic under-grading and assert the stored
    last_asked_slot does not repeat on two consecutive turns.

    Anti-repeat is best-effort (a hint that steers the LLM, not a hard constraint); here the brain credits
    no slot, isolating the deterministic rotation of the exclusion target.
    """
    headers, project = scenario_project
    scenario = Scenario(
        name="m1-no-consecutive-repeat",
        artifact_type="intent",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("ask_user", message="Câu hỏi khai thác 1?", active_mode="qa", slot_assessment={}),
                tool_select("ask_user", message="Câu hỏi khai thác 2?", active_mode="qa", slot_assessment={}),
                tool_select("ask_user", message="Câu hỏi khai thác 3?", active_mode="qa", slot_assessment={}),
            ]
        ),
        actions=[],
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)
    scenario_env.set_llm(scenario.llm)
    data = await driver._create_session()
    driver.session_id = uuid.UUID(data["session_id"])

    sequence: list[str | None] = []
    for content in ["Tôi cần làm rõ intent.", "Trả lời thứ nhất.", "Trả lời thứ hai."]:
        await driver._send_message(content)
        await scenario_env.drain(driver.session_id)
        sequence.append(await scenario_env.get_checkpoint_field(driver.session_id, "last_asked_slot"))

    asked = [slot for slot in sequence if slot is not None]
    assert asked, sequence
    assert all(a != b for a, b in zip(asked, asked[1:], strict=False)), sequence


async def test_get_checkpoint_field_reads_coverage_ratio(client, scenario_env, scenario_project):
    """Directly test the harness helper used to assert min_coverage."""
    headers, project = scenario_project
    scenario = Scenario(
        name="helper-reads-coverage",
        artifact_type="problem",
        llm=ScriptedLLM(
            tool_brain=[
                tool_select("write_draft", title="Vấn đề đăng ký lớp",
                            body="Sinh viên bị kẹt khi đăng ký lớp.",
                            active_mode="draft", slot_assessment=_FULL_PROBLEM),
            ]
        ),
        actions=[{"type": "send", "content": "Sinh viên bị kẹt khi đăng ký lớp, mỗi tuần, do rule tiên quyết."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    await driver.run()

    coverage_ratio = await scenario_env.get_checkpoint_field(driver.session_id, "coverage_ratio")
    assert coverage_ratio == 1.0
    missing = await scenario_env.get_checkpoint_field(driver.session_id, "nonexistent_field")
    assert missing is None
