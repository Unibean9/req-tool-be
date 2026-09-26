"""Deterministic next steps and the "claims done without a tool" guard (app/graphs/analysis/auto_steps.py)."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graphs.analysis.auto_steps import (
    AUTO_BRIEF,
    claims_completion,
    honest_fallback,
    needs_tool_retry,
    scripted_tool_call,
    with_nudge,
)
from tests.factories import _config, _make_agent_session, _project, _session_factory, _state

ALL_TOOLS = {"draft_in_parallel", "write_draft", "write_draft_section", "ask_user", "respond", "note"}


def _confirmed(reply: str) -> list:
    return [
        {"role": "user", "content": "viết FR"},
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "confirm_intent", "args": {"summary": "Viết FR"}}]),
        ToolMessage(content=reply, tool_call_id="c1"),
        {"role": "user", "content": reply},
    ]


def _fr_state(messages, **extra):
    state = _state(artifact_type="functional_requirement")
    state.update({"user_confirmed": True, "messages": messages, **extra})
    return state


@pytest.mark.parametrize("reply", ["ok", "Xác nhận", "đồng ý nhé", "làm đi", "OK!"])
def test_an_approved_intent_is_drafted_in_parallel_without_asking_the_model(reply):
    call = scripted_tool_call(_fr_state(_confirmed(reply)), ALL_TOOLS)

    assert call == {"name": "draft_in_parallel", "args": {"brief": AUTO_BRIEF}}


@pytest.mark.parametrize(
    ("state_change", "tools"),
    [
        ({"messages": _confirmed("không, bỏ phần thanh toán")}, ALL_TOOLS),  # a correction: the model handles it
        ({"artifact_type": "vision_objectives"}, ALL_TOOLS),  # no parallel plan
        ({"draft_body": "## Functional Requirements\n..."}, ALL_TOOLS),  # already drafted
        ({}, ALL_TOOLS - {"draft_in_parallel"}),  # not on this phase's menu
        ({"lifecycle_reports": [{"state": "current", "focused": True}]}, ALL_TOOLS),  # accepted: no re-proposal
    ],
)
def test_the_model_decides_when_the_step_is_not_fixed(state_change, tools, monkeypatch):
    state = _fr_state(_confirmed("ok"))
    state.update(state_change)
    if "lifecycle_reports" in state_change:
        monkeypatch.setattr(
            "app.graphs.analysis.auto_steps.lifecycle_tool_block_reason",
            lambda *_: "current_artifact_reproposal_blocked",
        )

    assert scripted_tool_call(state, tools) is None


def _after_parallel(status: str = "success"):
    return [
        *_confirmed("ok"),
        AIMessage(content="", tool_calls=[{"id": "p1", "name": "draft_in_parallel", "args": {"brief": "x"}}]),
        ToolMessage(content="Draft written", tool_call_id="p1", status=status),
    ]


def test_a_finished_parallel_draft_is_proposed_right_away():
    sections = {"## Functional Requirements": {"content": "## Functional Requirements\n| FR-01 |", "done": True}}

    call = scripted_tool_call(_fr_state(_after_parallel(), draft_sections=sections), ALL_TOOLS)

    assert call["name"] == "write_draft"
    assert call["args"]["title"]


def test_a_failed_parallel_draft_goes_back_to_the_model():
    # Even with complete sections left over from earlier, a failed run is not proposed.
    sections = {"## Functional Requirements": {"content": "## Functional Requirements\n| FR-01 |", "done": True}}

    assert scripted_tool_call(_fr_state(_after_parallel("error"), draft_sections=sections), ALL_TOOLS) is None


@pytest.mark.parametrize(
    "text",
    [
        "Đã tạo Constraints, Assumptions, and Risks như yêu cầu.",
        "Mình vừa viết xong bảng FR",
        "I have created the draft.",
    ],
)
def test_completion_claims_are_recognised(text):
    assert claims_completion(text)


def test_ordinary_replies_are_not_claims():
    assert not claims_completion("Mình sẽ tạo bảng FR sau khi bạn xác nhận phạm vi.")


def test_claiming_work_no_tool_did_needs_a_retry():
    state = _fr_state([{"role": "user", "content": "tạo artifact cho tôi ấy"}])

    assert needs_tool_retry(state, AIMessage(content="Đã tạo artifact như yêu cầu."))
    respond = {"id": "r1", "name": "respond", "args": {"message": "Đã tạo artifact như yêu cầu."}}
    assert needs_tool_retry(state, AIMessage(content="", tool_calls=[respond]))


def test_a_claim_after_a_real_write_this_turn_is_fine():
    messages = [
        {"role": "user", "content": "sửa FR-03"},
        AIMessage(content="", tool_calls=[{"id": "w1", "name": "write_draft_section", "args": {}}]),
        ToolMessage(content="Saved", tool_call_id="w1"),
    ]
    state = _fr_state(messages, draft_body="## Functional Requirements\n...")

    assert not needs_tool_retry(state, AIMessage(content="Đã cập nhật FR-03."))


def test_plain_text_before_a_draft_needs_a_retry_but_a_question_does_not():
    state = _fr_state([{"role": "user", "content": "viết FR"}])

    assert needs_tool_retry(state, AIMessage(content="Mình sẽ bắt đầu từ các capability."))
    assert not needs_tool_retry(state, AIMessage(content="Bạn muốn ưu tiên capability nào trước?"))
    # Once a draft exists, a plain answer (without a false claim) is a normal reply.
    assert not needs_tool_retry(
        _fr_state([{"role": "user", "content": "giải thích FR-02"}], draft_body="## Functional Requirements\n..."),
        AIMessage(content="FR-02 mô tả luồng đăng ký."),
    )


def test_an_unbacked_claim_is_replaced_with_the_truth():
    state = _fr_state([{"role": "user", "content": "tạo đi"}])

    fixed = honest_fallback(state, AIMessage(content="Đã tạo xong như yêu cầu."), "vi")

    assert fixed.content.startswith("Mình chưa tạo được nội dung nào")
    kept = AIMessage(content="", tool_calls=[{"id": "a1", "name": "draft_in_parallel", "args": {}}])
    assert honest_fallback(state, kept, "vi") is kept


def test_the_nudge_keeps_roles_alternating():
    messages = [{"role": "assistant", "content": "x"}, {"role": "user", "content": [{"type": "text", "text": "p"}]}]

    nudged = with_nudge(messages, "call a tool")

    assert [message["role"] for message in nudged] == ["assistant", "user"]
    assert nudged[-1]["content"][-1] == {"type": "text", "text": "call a tool"}
    assert messages[-1]["content"] == [{"type": "text", "text": "p"}]  # the original is untouched


# --- through analyze_node ----------------------------------------------------------------------


async def _analyze(client, db_session, state, replies):
    from app.graphs.nodes import analyze_node

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=[(reply, None) for reply in replies])
    config = _config(str(agent_session.id), str(project_id), llm)
    config["configurable"]["session_factory"] = _session_factory()
    out = await analyze_node(state, config)
    return out, llm


@pytest.mark.asyncio
async def test_analyze_takes_the_scripted_step_without_an_llm_call(client, db_session, monkeypatch):
    # A fresh project has no accepted predecessors, so FR's lifecycle would block drafting.
    monkeypatch.setattr("app.graphs.analysis.auto_steps.lifecycle_tool_block_reason", lambda *_: None)

    out, llm = await _analyze(client, db_session, _fr_state(_confirmed("ok")), [])

    llm.generate.assert_not_called()
    assert [call["name"] for call in out["messages"][-1].tool_calls] == ["draft_in_parallel"]


@pytest.mark.asyncio
async def test_analyze_retries_a_false_claim_and_uses_the_real_tool_call(client, db_session):
    state = _fr_state([{"role": "user", "content": "tạo artifact cho tôi ấy"}])
    real = AIMessage(content="", tool_calls=[{"id": "x", "name": "ask_user", "args": {"message": "Phạm vi?"}}])

    out, llm = await _analyze(client, db_session, state, [AIMessage(content="Đã tạo artifact như yêu cầu."), real])

    assert llm.generate.await_count == 2
    assert llm.generate.await_args_list[1].kwargs["tool_choice"] == "required"
    assert [call["name"] for call in out["messages"][-1].tool_calls] == ["ask_user"]


@pytest.mark.asyncio
async def test_analyze_never_shows_a_claim_the_retry_still_makes(client, db_session):
    state = _fr_state([{"role": "user", "content": "tạo artifact cho tôi ấy"}], locale="vi")
    claim = AIMessage(content="Đã tạo artifact như yêu cầu.")

    out, _llm = await _analyze(client, db_session, state, [claim, claim])

    [call] = out["messages"][-1].tool_calls
    assert call["name"] == "respond"
    assert call["args"]["message"].startswith("Mình chưa tạo được nội dung nào")


@pytest.mark.asyncio
async def test_analyze_does_not_retry_a_plain_question(client, db_session):
    state = _fr_state([{"role": "user", "content": "viết FR"}])

    out, llm = await _analyze(client, db_session, state, [AIMessage(content="Bạn muốn ưu tiên capability nào?")])

    assert llm.generate.await_count == 1
    assert [call["name"] for call in out["messages"][-1].tool_calls] == ["ask_user"]
