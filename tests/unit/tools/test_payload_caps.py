"""Payload caps: key_facts / situation report / artifact history render caps, and confirms
_build_feedback_control_block is already idempotent with enough signal (no refactor, tests only).
"""

from app.graphs.nodes import (
    _build_artifact_history_block,
    _build_feedback_control_block,
    _build_key_facts_block,
    _build_situation_report_block,
)
from tests.factories import _state

# ---------------------------------------------------------------------------
# key_facts cap (_MAX_KEY_FACTS_RENDERED = 20)
# ---------------------------------------------------------------------------

def test_key_facts_block_renders_all_when_under_cap():
    state = _state()
    state["key_facts"] = [{"statement": f"fact {i}", "source": "user"} for i in range(5)]

    block = _build_key_facts_block(state)

    for i in range(5):
        assert f"fact {i}" in block


def test_key_facts_block_caps_at_20_and_keeps_newest():
    state = _state()
    # 25 facts, append order: fact 0 (oldest) -> fact 24 (newest).
    state["key_facts"] = [{"statement": f"fact {i}", "source": "user"} for i in range(25)]

    block = _build_key_facts_block(state)

    # The 20 MOST RECENT entries (fact 5..24) must remain intact, in the same order.
    for i in range(5, 25):
        assert f"fact {i}" in block
    # The 5 OLDEST entries (fact 0..4) are dropped. Append " (source" to avoid a false positive
    # from "fact 1" being a substring of "fact 10".."fact 19".
    for i in range(5):
        assert f"fact {i} (source" not in block
    assert "fact 24" in block  # the newest fact is never dropped


# ---------------------------------------------------------------------------
# situation report / artifact history cap (_MAX_LIFECYCLE_ITEMS_RENDERED = 8)
# ---------------------------------------------------------------------------

def _lifecycle_reports(n: int) -> list[dict]:
    return [
        {
            "artifact_type": "goal",
            "artifact_id": f"art-{i}",
            "state": "draft",
            "allowed_actions": [],
            "reason": f"reason {i}",
        }
        for i in range(n)
    ]


def _artifact_history(n: int) -> list[dict]:
    return [
        {
            "artifact_type": "goal",
            "version_number": i,
            "change_source": "user",
            "version_id": f"v-{i}",
        }
        for i in range(n)
    ]


def test_situation_report_block_renders_all_when_under_cap():
    state = _state()
    state["lifecycle_reports"] = _lifecycle_reports(3)

    block = _build_situation_report_block(state)

    for i in range(3):
        assert f"art-{i}" in block


def test_situation_report_block_caps_at_8_and_keeps_last_8():
    state = _state()
    state["lifecycle_reports"] = _lifecycle_reports(12)

    block = _build_situation_report_block(state)

    for i in range(4, 12):
        assert f"art-{i} " in block
    # Append " state=" to avoid a false positive from "art-1" being a substring of "art-10".."art-19".
    for i in range(4):
        assert f"art-{i} state=" not in block


def test_artifact_history_block_caps_at_8_and_keeps_last_8():
    state = _state()
    state["artifact_history"] = _artifact_history(10)

    block = _build_artifact_history_block(state)

    for i in range(2, 10):
        assert f"v-{i}" in block
    for i in range(2):
        assert f"v-{i}" not in block


# ---------------------------------------------------------------------------
# feedback block: idempotency + presence of every signal (no refactor — confirm only)
# ---------------------------------------------------------------------------

def test_feedback_control_block_is_idempotent_for_same_state():
    state = _state()
    state["quality_report"] = {"quality_gate_result": "pass", "blocking_issues": ["b1"]}
    state["candidate_readiness"] = {"state": "ready", "missing": ["m1"]}
    state["diagnosis_signal"] = {"risk_level": "high", "signals": ["s1"]}

    first = _build_feedback_control_block(state)
    second = _build_feedback_control_block(state)

    assert first == second


def test_feedback_control_block_contains_every_main_signal_when_set():
    state = _state()
    state["quality_report"] = {"quality_gate_result": "fail", "blocking_issues": ["missing scope"]}
    state["candidate_readiness"] = {"state": "not_ready", "blocking_reasons": ["needs review"]}
    state["diagnosis_signal"] = {"risk_level": "high", "signals": ["ambiguity"]}

    block = _build_feedback_control_block(state)

    assert "quality_gate" in block
    assert "blockers" in block
    assert "candidate_readiness" in block
    assert "diagnosis_risk" in block
