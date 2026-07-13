"""Prefix stability of the analyst prompts after reordering assembled blocks.

build_system_prompt now appends semi-static blocks (artifact_contract, batching, type_profile —
depend only on artifact_type/phase, stable within a session phase) before dynamic blocks
(thinking_mode, section_repair, stuck_escalation — read per-turn state). This lets a caching
provider reuse the stable prefix across turns of the same phase regardless of how much the
dynamic signals change. _build_tool_selection_prompt keeps conversation_summary out of the
stable context/tool cluster for the same reason.
"""

from app.graphs.analysis.prompt_assembly import build_system_prompt
from app.graphs.nodes import _build_tool_selection_prompt
from app.instructions import load_instructions
from tests.factories import _state

_THINKING_MODE_MARKER = "THINKING MODE:"
_SECTION_REPAIR_MARKER = "SECTION REPAIR"
_BATCHING_MARKER = "BATCHING:"
_CONTRACT_MARKER = "REQUIRED OUTPUT CONTRACT"

_VIOLATION = {"section": "## Business Rules", "severity": "violation", "message": "missing outcome"}


def _no_dynamic_signal_state():
    state = _state(artifact_type="vision_objectives")
    state["session_phase"] = None  # unmodeled -> full block assembly, no phase-specific gating noise
    state["thinking_mode"] = None
    state["section_findings"] = {}
    state["recent_tool_calls"] = []
    return state


def _full_dynamic_signal_state():
    state = _no_dynamic_signal_state()
    state["thinking_mode"] = "challenging"
    state["section_findings"] = {"## Business Rules": [_VIOLATION]}
    # _is_near_stuck fires at (_REPEATED_TOOL_CALL_EXIT_THRESHOLD - 1) identical tail entries.
    state["recent_tool_calls"] = ["ask_user::{}", "ask_user::{}"]
    return state


def test_system_prompt_prefix_is_stable_regardless_of_dynamic_signals():
    load_instructions()
    prompt_none = build_system_prompt(_no_dynamic_signal_state(), None, has_draft=True)
    prompt_with_dynamic = build_system_prompt(_full_dynamic_signal_state(), None, has_draft=True)

    assert prompt_with_dynamic.startswith(prompt_none)
    assert prompt_with_dynamic != prompt_none


def test_system_prompt_still_renders_every_block_after_reorder():
    load_instructions()
    prompt = build_system_prompt(_full_dynamic_signal_state(), None, has_draft=True)

    assert _CONTRACT_MARKER in prompt
    assert _BATCHING_MARKER in prompt
    assert _THINKING_MODE_MARKER in prompt
    assert _SECTION_REPAIR_MARKER in prompt


def test_tool_selection_prompt_prefix_ignores_conversation_summary():
    state_a = _state(artifact_type="brd")
    state_a["conversation_summary"] = "Earlier the user described the checkout flow."
    state_a["key_facts"] = [{"statement": "Channel is web", "source": "user", "turn": "1"}]
    state_a["quality_report"] = {"quality_gate_result": "fail", "blocking_issues": ["missing scope"]}

    state_b = _state(artifact_type="brd")
    state_b["conversation_summary"] = "Later turns pivoted to the mobile app entirely."
    state_b["key_facts"] = [
        {"statement": "Channel is mobile", "source": "user", "turn": "2"},
        {"statement": "Region is APAC", "source": "user", "turn": "3"},
    ]
    state_b["quality_report"] = {"quality_gate_result": "pass"}

    prompt_a = _build_tool_selection_prompt(state_a, [])
    prompt_b = _build_tool_selection_prompt(state_b, [])

    marker = "Tools available this turn:"
    prefix_a = prompt_a[: prompt_a.index(marker)]
    prefix_b = prompt_b[: prompt_b.index(marker)]
    assert prefix_a == prefix_b
