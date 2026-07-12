"""active_mode lock removed.

The per-turn single-mode derivation (_TOOL_ACTIVE_MODE → analysis_result["active_mode"]) is gone:
the analyst may exercise multiple roles in one turn, and tool availability never depends on a mode
label. The real safety gates (finalize-quality, solo-interrupt) survive untouched.
"""

from app.graphs import nodes
from app.graphs.agent_tools import get_available_tools
from app.graphs.nodes import _INTERRUPT_BEARING_TOOLS, _gate_selected_tools
from app.graphs.state import WorkflowState


def _names(state):
    return {t.name for t in get_available_tools(state)}


def test_active_mode_field_removed_from_state():
    """No active_mode channel in state and no derivation table in nodes."""
    assert "active_mode" not in WorkflowState.__annotations__
    assert not hasattr(nodes, "_TOOL_ACTIVE_MODE")


def test_available_tools_not_filtered_by_mode():
    """get_available_tools depends only on phase + real gates, never on a mode label."""
    state = {"messages": [], "user_confirmed": True}
    names = _names(state)
    assert names  # non-empty menu derived purely from state
    # Multiple roles coexist in the same menu: note and read all offered together.
    assert {"note", "read_artifact"} <= names


def test_finalize_safety_gate_still_active():
    """Regression: finalize stays withheld before readiness, independent of any mode."""
    state = {"messages": [], "user_confirmed": True}
    assert "finalize" not in _names(state)


def test_solo_interrupt_still_enforced():
    """Regression: two interrupt-bearing tools in one turn collapse to the first, independent of mode.

    Uses ask_user + respond, both in-phase in ELICIT, so this isolates solo enforcement from the
    per-phase gate (picking two arbitrary interrupt tools could hit finalize/confirm_intent,
    which ELICIT excludes — the drop would then be phase gating, not solo enforcement)."""
    assert {"ask_user", "respond"} <= _INTERRUPT_BEARING_TOOLS
    raw = [{"name": "ask_user", "args": {}}, {"name": "respond", "args": {}}]
    gated = _gate_selected_tools({"messages": [], "user_confirmed": True}, raw)
    assert [t["name"] for t in gated] == ["ask_user"]
