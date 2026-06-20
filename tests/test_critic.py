"""Critic Node (edge-loop). Tests written before implementation."""

from unittest.mock import AsyncMock

import pytest

from app.graphs.critic import (
    ANALYSIS_SCHEMA,
    CRITIC_SCHEMA,
    quality_gate_node,
    route_after_gate,
)


def _config(llm_client):
    return {"configurable": {"llm_client": llm_client}}


def _config_with_strong(default_client, strong_client):
    return {"configurable": {"llm_client": default_client, "strong_llm_client": strong_client}}


def _critic_result(overall: float) -> dict:
    return {
        "scores": {"unambiguous": overall, "verifiable": overall},
        "overall": overall,
        "rationale": "lý do",
        "suggestions": ["thêm tiêu chí đo lường"],
    }


def _state(proposals, *, rounds=0, messages=None, next_action="propose"):
    return {
        "artifact_type": "story",
        "messages": messages or [],
        "analysis_result": {"next_action": next_action, "confidence": 0.8, "proposals": proposals},
        "critique_rounds": rounds,
    }


def _clean_proposal():
    return {
        "artifact_type": "story",
        "title": "Đăng nhập",
        "body": "Given đã đăng ký, when nhập đúng mật khẩu, then vào hệ thống",
    }


def _critic_calls(llm_client):
    return [c for c in llm_client.generate.call_args_list if c.kwargs.get("response_format") is CRITIC_SCHEMA]


def _regen_calls(llm_client):
    return [c for c in llm_client.generate.call_args_list if c.kwargs.get("response_format") is ANALYSIS_SCHEMA]


# --- Group 1: Happy path ---

@pytest.mark.asyncio
async def test_gate_passes_clean_proposal():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    out = await quality_gate_node(_state([_clean_proposal()]), _config(llm))

    assert out["quality_report"]["passed"] is True
    assert out["critique_rounds"] == 1
    assert route_after_gate({**out, "quality_report": out["quality_report"]}) == "propose_artifacts"


@pytest.mark.asyncio
async def test_quality_gate_uses_strong_client_when_present():
    default_llm = AsyncMock()
    default_llm.generate = AsyncMock()
    strong_llm = AsyncMock()
    strong_llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    out = await quality_gate_node(_state([_clean_proposal()]), _config_with_strong(default_llm, strong_llm))

    assert out["quality_report"]["passed"] is True
    strong_llm.generate.assert_called_once()
    default_llm.generate.assert_not_called()


# --- Group 2: Missing required fields (hard block, critic not called) ---

@pytest.mark.asyncio
async def test_gate_blocks_missing_title():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    out = await quality_gate_node(_state([{"body": "Thiếu title"}]), _config(llm))

    llm.generate.assert_not_called()
    assert out["quality_report"]["passed"] is False


@pytest.mark.asyncio
async def test_gate_blocks_missing_body():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    out = await quality_gate_node(_state([{"title": "Thiếu body"}]), _config(llm))

    llm.generate.assert_not_called()
    assert out["quality_report"]["passed"] is False


# --- Group 3: Edge-loop / round cap ---

def test_route_loops_back_when_below_threshold():
    state = {"quality_report": {"passed": False}, "critique_rounds": 1}
    assert route_after_gate(state) == "quality_gate"


def test_route_forwards_when_passed():
    state = {"quality_report": {"passed": True}, "critique_rounds": 1}
    assert route_after_gate(state) == "propose_artifacts"


def test_route_forwards_at_max_rounds():
    state = {"quality_report": {"passed": False}, "critique_rounds": 2}
    assert route_after_gate(state) == "propose_artifacts"


@pytest.mark.asyncio
async def test_node_calls_critic_once_per_invocation():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    await quality_gate_node(_state([_clean_proposal()]), _config(llm))

    assert len(_critic_calls(llm)) == 1


# --- Group 4: Feedback from request_edit ---

@pytest.mark.asyncio
async def test_request_edit_note_in_regenerate_prompt():
    note = "hãy thêm tiêu chí đo lường"
    regen_result = {"next_action": "propose", "proposals": [_clean_proposal()]}
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=[(_critic_result(0.4), None), (regen_result, None)])

    messages = [{"type": "request_edit", "note": note}]
    await quality_gate_node(_state([_clean_proposal()], messages=messages), _config(llm))

    regen = _regen_calls(llm)
    assert len(regen) == 1
    assert note in regen[0].kwargs["messages"][0]["content"]


# --- Group 5: Weasel words (warning, non-blocking) ---

@pytest.mark.asyncio
async def test_weasel_word_triggers_regenerate_not_block():
    weak = {"artifact_type": "story", "title": "Story", "body": "Given A, when B, then C — chạy nhanh"}
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.85), None))

    out = await quality_gate_node(_state([weak]), _config(llm))

    assert len(_critic_calls(llm)) == 1
    assert out["analysis_result"]["proposals"]


# --- Group 6: State output (no messages write, proposals-only replacement) ---

@pytest.mark.asyncio
async def test_gate_increments_critique_rounds():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    out = await quality_gate_node(_state([_clean_proposal()], rounds=1), _config(llm))

    assert out["critique_rounds"] == 2


@pytest.mark.asyncio
async def test_gate_writes_quality_report():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    out = await quality_gate_node(_state([_clean_proposal()]), _config(llm))
    report = out["quality_report"]

    assert "scores" in report
    assert "overall" in report
    assert "violations" in report
    assert "warnings" in report


@pytest.mark.asyncio
async def test_gate_does_not_write_messages():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(_critic_result(0.8), None))

    out = await quality_gate_node(_state([_clean_proposal()]), _config(llm))

    assert "messages" not in out


@pytest.mark.asyncio
async def test_regenerate_only_replaces_proposals():
    old_proposals = [_clean_proposal()]
    new_proposal = {"artifact_type": "story", "title": "Mới", "body": "Given X, when Y, then Z"}
    regen_result = {"next_action": "ask", "proposals": [new_proposal]}
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=[(_critic_result(0.4), None), (regen_result, None)])

    out = await quality_gate_node(_state(old_proposals, next_action="propose"), _config(llm))

    assert out["analysis_result"]["next_action"] == "propose"
    assert out["analysis_result"]["proposals"] == [new_proposal]
