"""Phase 02 — cold-start hard gate.

At cold start (no decision_nodes AND no elicit run this session) write_draft is removed from the
menu even when an analysis_frame is ready: exploration must precede the first draft. Once elicit has
run, or nodes already exist (resumed project), write_draft returns.
"""

from app.graphs.agent_tools import get_available_tools


def _ready_analysis_frame():
    return {
        "interpreted_intent": "Xây app quản lý quán cà phê.",
        "evidence": ["Chủ quán muốn giảm thất thoát."],
        "gaps": ["Chưa rõ quy mô."],
        "analysis_angles": ["Pain", "Scope", "Metric"],
        "assumptions": ["Bắt đầu từ POS cơ bản."],
        "recommended_next_move": "Khai phá trước khi draft.",
    }


def _base_state(**overrides):
    state = {
        "messages": [],
        "user_confirmed": True,
        "analysis_frame": _ready_analysis_frame(),
    }
    state.update(overrides)
    return state


def _names(state):
    return {t.name for t in get_available_tools(state)}


def test_write_draft_blocked_when_no_nodes_and_no_elicit():
    state = _base_state(decision_nodes={}, session_elicit_count=0)

    assert "write_draft" not in _names(state)


def test_write_draft_available_after_elicit_runs():
    state = _base_state(decision_nodes={}, session_elicit_count=1)

    assert "write_draft" in _names(state)


def test_write_draft_available_when_nodes_exist():
    state = _base_state(decision_nodes={"N1": {"id": "N1"}}, session_elicit_count=0)

    assert "write_draft" in _names(state)
