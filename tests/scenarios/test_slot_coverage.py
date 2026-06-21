import uuid

import pytest

from app.graphs.slot_schema import BRD_SLOTS, COVERAGE_THRESHOLD
from tests.scenarios.driver import Scenario, ScenarioDriver
from tests.scenarios.scripted_llm import ScriptedLLM, artifact, ask, propose

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

# Phase 0: after tier-2 removal only the tier-1 floor (0 slot filled) blocks. Scenarios that
# previously relied on tier-2 to gate a past-floor propose now drive the floor with 0 filled.
_EMPTY_PROBLEM = {
    "who": "empty",
    "obstacle": "empty",
    "root_cause": "empty",
    "frequency": "empty",
    "impact": "empty",
}


async def test_problem_gate_blocks_premature_propose(client, scenario_env, scenario_project):
    """Phase 0: the tier-1 floor blocks propose-from-greeting (0 slot filled)."""
    headers, project = scenario_project
    scenario = Scenario(
        name="problem-gate-blocks-premature-propose",
        artifact_type="problem",
        llm=ScriptedLLM(
            brain=[
                {
                    **propose(
                        artifact(
                            "problem",
                            "Vấn đề: đăng ký lớp bị kẹt",
                            "Người dùng bị kẹt khi đăng ký lớp.",
                        )
                    ),
                    "message": "Bạn mô tả thêm nguyên nhân gốc rễ của vấn đề này được không?",
                    "slot_assessment": _EMPTY_PROBLEM,
                }
            ]
        ),
        actions=[{"type": "send", "content": "Sinh viên bị kẹt khi đăng ký lớp."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "ask_human"
    final_snapshot = recorder.steps[-1]["snapshot"]
    assert final_snapshot["tool_calls"] == []
    agent_messages = [m for m in final_snapshot["messages"] if m["role"] == "agent"]
    assert agent_messages[-1]["payload"]["kind"] == "question"


async def test_problem_proposes_after_coverage_met(client, scenario_env, scenario_project):
    headers, project = scenario_project
    final_turn = propose(
        artifact(
            "problem",
            "Vấn đề: đăng ký lớp bị kẹt",
            "Sinh viên bị kẹt khi đăng ký lớp do rule lớp tiên quyết không rõ, xảy ra hằng tuần và làm trễ kế hoạch học.",
        )
    )
    final_turn["slot_assessment"] = _FULL_PROBLEM
    scenario = Scenario(
        name="problem-proposes-after-coverage-met",
        artifact_type="problem",
        llm=ScriptedLLM(
            brain=[
                ask("Ai là người bị ảnh hưởng trực tiếp?", slot_assessment={}),
                ask(
                    "Vấn đề xảy ra thường xuyên như thế nào?",
                    acknowledgment="Đã rõ đối tượng và trở ngại.",
                    slot_assessment=_PARTIAL_PROBLEM,
                ),
                final_turn,
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
    assert recorder.summary["final_interrupt"] == "ask_human"
    assert coverage_ratio >= COVERAGE_THRESHOLD["problem"]
    final_snapshot = recorder.steps[-1]["snapshot"]
    agent_messages = [m for m in final_snapshot["messages"] if m["role"] == "agent"]
    assert agent_messages[-1]["payload"]["kind"] == "confirm"


async def test_one_question_per_turn_preserved(client, scenario_env, scenario_project):
    headers, project = scenario_project
    scenario = Scenario(
        name="problem-one-question-per-turn-preserved",
        artifact_type="problem",
        llm=ScriptedLLM(
            brain=[
                ask("Ai là người bị ảnh hưởng trực tiếp?", slot_assessment={}),
                ask("Tần suất xảy ra như thế nào?", acknowledgment="Đã rõ đối tượng.", slot_assessment=_PARTIAL_PROBLEM),
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
    final_snapshot = recorder.steps[-1]["snapshot"]
    agent_messages = [m for m in final_snapshot["messages"] if m["role"] == "agent"]

    assert agent_messages
    for message in agent_messages:
        if message["payload"]["kind"] == "question":
            assert message["content"].count("?") == 1


# ---------------------------------------------------------------------------
# Generalized per-key BRD scenarios (Phase 5/6). `problem` is covered above;
# here we exercise the other eight keys through the same gate -> confirm path.
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
async def test_brd_key_gate_and_coverage(client, scenario_env, scenario_project, key):
    """Each BRD key: premature propose is gated, then proceeds once coverage is met."""
    headers, project = scenario_project
    required = BRD_SLOTS[key]["required"]

    # Turn 1: brain attempts to propose from greeting (0 slot filled), but carries a fallback
    # question so the tier-1 floor can convert it into a real ask_human turn. (Phase 0: only the
    # floor blocks now; a past-floor partial would be honoured.)
    block_turn = {
        **propose(artifact(key, f"{key} nháp", f"Nội dung sơ bộ cho {key}.")),
        "message": "Bạn bổ sung thêm thông tin còn thiếu được không?",
        "slot_assessment": {slot: "empty" for slot in required},
    }
    # Turn 2: full slots -> coverage complete -> proceed to confirm.
    final_turn = propose(artifact(key, f"{key} hoàn chỉnh", f"Nội dung đầy đủ cho {key}."))
    final_turn["slot_assessment"] = _full(required)

    scenario = Scenario(
        name=f"{key}-gate-and-coverage",
        artifact_type=key,
        llm=ScriptedLLM(brain=[block_turn, final_turn]),
        actions=[
            {"type": "send", "content": f"Tôi cần làm rõ {key}."},
            {"type": "send", "content": "Đây là thông tin đầy đủ tôi cung cấp."},
        ],
        expect={"min_coverage": COVERAGE_THRESHOLD[key]},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    # Turn 1 (steps[1]) — gate blocked the premature propose: agent asked a question.
    first_send_msgs = [m for m in recorder.steps[1]["snapshot"]["messages"] if m["role"] == "agent"]
    assert first_send_msgs[-1]["payload"]["kind"] == "question"

    # Turn 2 — coverage met: agent reaches confirm. (min_coverage asserted inside driver.run.)
    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "ask_human"
    final_msgs = [m for m in recorder.steps[-1]["snapshot"]["messages"] if m["role"] == "agent"]
    assert final_msgs[-1]["payload"]["kind"] == "confirm"


async def test_slot_coverage_does_not_gate_non_brd(client, scenario_env, scenario_project):
    """A non-BRD artifact proposes on turn 1 with no slot_assessment -> gate never blocks."""
    headers, project = scenario_project
    scenario = Scenario(
        name="non-brd-fail-open",
        artifact_type="functional_requirement",
        llm=ScriptedLLM(
            brain=[
                propose(
                    artifact(
                        "functional_requirement",
                        "FR-1: Đăng ký lớp",
                        "Hệ thống phải cho phép sinh viên đăng ký lớp trong thời gian mở đăng ký.",
                    )
                )
            ]
        ),
        actions=[{"type": "send", "content": "Tôi cần một yêu cầu chức năng cho đăng ký lớp."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "ask_human"
    final_msgs = [m for m in recorder.steps[-1]["snapshot"]["messages"] if m["role"] == "agent"]
    assert final_msgs[-1]["payload"]["kind"] == "confirm"


async def test_gate_redirects_done_without_message_does_not_crash(client, scenario_env, scenario_project):
    """A `done` turn with incomplete coverage is gated to ask_human; it carries no message,
    so ask_human_node must repair it into a conversational question instead of raising."""
    headers, project = scenario_project
    done_turn = {
        "next_action": "done",
        "confidence": 1.0,
        "gaps": [],
        "message": "",
        "proposals": [],
        "slot_assessment": _EMPTY_PROBLEM,
    }
    scenario = Scenario(
        name="gate-redirects-done-no-message",
        artifact_type="problem",
        llm=ScriptedLLM(
            brain=[done_turn],
            followup={
                "message": (
                    "Mình thấy phần vấn đề vẫn còn thiếu bối cảnh để viết chắc hơn. "
                    "Ai đang bị ảnh hưởng trực tiếp bởi tình huống này?"
                )
            },
        ),
        actions=[{"type": "send", "content": "Vấn đề đăng ký lớp, chưa rõ chi tiết."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    assert recorder.summary["final_status"] == "waiting_for_human"
    assert recorder.summary["final_interrupt"] == "ask_human"
    agent_messages = [m for m in recorder.steps[-1]["snapshot"]["messages"] if m["role"] == "agent"]
    assert agent_messages[-1]["payload"]["kind"] == "question"
    assert agent_messages[-1]["content"].startswith("Mình thấy phần vấn đề")
    assert agent_messages[-1]["content"].count("?") == 1


# ---------------------------------------------------------------------------
# Eval scenarios for the LLM-led rebalance — M1 (no consecutive repeat), M2 (on-topic judge
# infra), M3 (ask->propose timing incl. user override).
# ---------------------------------------------------------------------------

async def test_m2_question_maps_to_slot():
    """M2 — judge infrastructure only.

    The scripted judge defaults on_topic=True regardless of the question, so this confirms the
    judge route is wired, NOT that questions are actually on-topic. Real M2 needs a live LLM
    judge outside CI (see plan.md M2). Asserting more here would be false confidence.
    """
    from tests.scenarios.scripted_llm import ON_TOPIC_SCHEMA, ScriptedLLM

    llm = ScriptedLLM(brain=[])
    result, _usage = await llm.generate(
        messages=[{"role": "user", "content": "Ai là người tài trợ sáng kiến này?"}],
        response_format=ON_TOPIC_SCHEMA,
    )

    assert result["on_topic"] is True
    assert llm.calls[-1]["route"] == "judge"


async def test_m3_proposes_when_coverage_sufficient(client, scenario_env, scenario_project):
    """M3 — once coverage is met the agent reaches confirm (proposes on time)."""
    headers, project = scenario_project
    required = BRD_SLOTS["intent"]["required"]
    final_turn = propose(artifact("intent", "Intent hoàn chỉnh", "Nội dung đầy đủ cho intent."))
    final_turn["slot_assessment"] = _full(required)
    scenario = Scenario(
        name="m3-proposes-when-sufficient",
        artifact_type="intent",
        llm=ScriptedLLM(brain=[final_turn]),
        actions=[{"type": "send", "content": "Đây là toàn bộ thông tin đầy đủ cho intent."}],
        expect={"min_coverage": COVERAGE_THRESHOLD["intent"]},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    assert recorder.summary["final_interrupt"] == "ask_human"
    final_msgs = [m for m in recorder.steps[-1]["snapshot"]["messages"] if m["role"] == "agent"]
    assert final_msgs[-1]["payload"]["kind"] == "confirm"


async def test_m3_user_override_respected(client, scenario_env, scenario_project):
    """M3 — coverage incomplete but past the tier-1 floor (1 slot filled) -> agent proceeds to
    confirm. Phase 0 removed the tier-2 soft gate, so propose past the floor is now honoured
    regardless of the user override; the "cứ tạo đi" message is incidental here.
    """
    headers, project = scenario_project
    block_turn = ask("Vì sao sáng kiến này cần làm bây giờ?", slot_assessment={"why_now": "filled"})
    override_turn = {
        **propose(artifact("intent", "Intent nháp", "Nội dung sơ bộ cho intent.")),
        "message": "Bạn bổ sung thêm thông tin còn thiếu được không?",
        "slot_assessment": {"why_now": "filled"},
    }
    scenario = Scenario(
        name="m3-user-override-respected",
        artifact_type="intent",
        llm=ScriptedLLM(brain=[block_turn, override_turn]),
        actions=[
            {"type": "send", "content": "Tôi cần làm rõ intent."},
            {"type": "send", "content": "cứ tạo đi"},
        ],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    recorder = await driver.run()

    final_msgs = [m for m in recorder.steps[-1]["snapshot"]["messages"] if m["role"] == "agent"]
    assert final_msgs[-1]["payload"]["kind"] == "confirm"


async def test_m1_last_asked_slot_no_consecutive_repeat(client, scenario_env, scenario_project):
    """M1 — drive a real multi-turn HTTP conversation under chronic under-grading and assert the
    persisted last_asked_slot never repeats two turns running.

    Unlike the unit test (which calls analyze_node directly), this exercises the full
    HTTP -> graph -> checkpointer path. Anti-repeat is best-effort by design (the hint steers the
    LLM but does not bind it); here the scripted brain emits no slot credit, isolating the
    deterministic rotation of the exclusion target.
    """
    headers, project = scenario_project
    scenario = Scenario(
        name="m1-no-consecutive-repeat",
        artifact_type="intent",
        llm=ScriptedLLM(
            brain=[
                ask("Câu hỏi khai thác 1?", slot_assessment={}),
                ask("Câu hỏi khai thác 2?", slot_assessment={}),
                ask("Câu hỏi khai thác 3?", slot_assessment={}),
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
    assert all(a != b for a, b in zip(asked, asked[1:])), sequence


async def test_get_checkpoint_field_reads_coverage_ratio(client, scenario_env, scenario_project):
    """Direct test of the harness helper used to assert min_coverage."""
    headers, project = scenario_project
    final_turn = propose(artifact("problem", "Vấn đề đăng ký lớp", "Sinh viên bị kẹt khi đăng ký lớp."))
    final_turn["slot_assessment"] = _FULL_PROBLEM
    scenario = Scenario(
        name="helper-reads-coverage",
        artifact_type="problem",
        llm=ScriptedLLM(brain=[final_turn]),
        actions=[{"type": "send", "content": "Sinh viên bị kẹt khi đăng ký lớp, mỗi tuần, do rule tiên quyết."}],
        expect={},
    )
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)

    await driver.run()

    coverage_ratio = await scenario_env.get_checkpoint_field(driver.session_id, "coverage_ratio")
    assert coverage_ratio == 1.0
    missing = await scenario_env.get_checkpoint_field(driver.session_id, "nonexistent_field")
    assert missing is None
