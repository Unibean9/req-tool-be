"""Tests for thinking-mode suffix and stuck-loop warning blocks."""

from app.graphs.agent_tools import ELICIT_TECHNIQUES
from app.graphs.nodes import (
    _THINKING_MODE_TECHNIQUE_HINTS,
    _build_stuck_escalation_block,
    _build_thinking_mode_block,
)


def test_high_risk_thinking_mode_produces_guidance_block():
    block = _build_thinking_mode_block({"thinking_mode": "challenging"})
    assert "THINKING MODE: challenging" in block
    assert "reverse" in block
    assert "first_principles" in block
    assert "challenge_assumptions" in block


def test_low_risk_or_unset_thinking_mode_produces_no_block():
    assert _build_thinking_mode_block({"thinking_mode": "structuring"}) == ""
    assert _build_thinking_mode_block({"thinking_mode": "synthesizing"}) == ""
    assert _build_thinking_mode_block({"thinking_mode": None}) == ""
    assert _build_thinking_mode_block({}) == ""


def test_thinking_mode_block_only_names_techniques_elicit_tool_can_call():
    for techniques in _THINKING_MODE_TECHNIQUE_HINTS.values():
        for technique in techniques:
            assert technique in ELICIT_TECHNIQUES


def test_stuck_escalation_block_fires_one_repeat_before_hard_stop():
    # route_node's hard stop needs 3 identical fingerprints; the warning should fire at 2.
    near_stuck_calls = ["write_draft:[]", "write_draft:[]"]
    block = _build_stuck_escalation_block({"recent_tool_calls": near_stuck_calls})
    assert "LOOP WARNING" in block


def test_stuck_escalation_block_empty_when_not_near_stuck():
    assert _build_stuck_escalation_block({"recent_tool_calls": []}) == ""
    assert _build_stuck_escalation_block({"recent_tool_calls": ["write_draft:[]"]}) == ""
    assert _build_stuck_escalation_block({"recent_tool_calls": ["a:[]", "b:[]"]}) == ""
    assert _build_stuck_escalation_block({}) == ""


def test_stuck_escalation_block_disabled_by_kill_switch(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "enable_adaptive_diagnosis", False)
    near_stuck_calls = ["write_draft:[]", "write_draft:[]"]
    assert _build_stuck_escalation_block({"recent_tool_calls": near_stuck_calls}) == ""
