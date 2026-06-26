"""Phase 05 — render_view pure projection from the decision graph to markdown.

The view is derived, never the source: superseded hidden, parked folded into its own section,
active nodes (confirmed/inferred/needs_confirmation) shown. brd and prd use distinct templates.
"""

from app.graphs.decision_graph import render_view


def test_render_view_shows_confirmed_and_inferred(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N3", "kind": "objective", "statement": "Giảm thất thoát", "status": "confirmed"},
        {"id": "N4", "kind": "scope", "statement": "Quản lý công thức", "status": "inferred"},
    )

    out = render_view(nodes, "brd")

    assert "Giảm thất thoát" in out
    assert "Quản lý công thức" in out


def test_render_view_hides_superseded(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "statement": "Hướng cũ", "status": "superseded"},
        {"id": "N3", "kind": "objective", "statement": "Mục tiêu chốt", "status": "confirmed"},
    )

    out = render_view(nodes, "brd")

    assert "Hướng cũ" not in out


def test_render_view_folds_parked_into_separate_section(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "P", "kind": "scope", "statement": "Phạm vi treo lại", "status": "parked"},
        {"id": "C", "kind": "objective", "statement": "Mục tiêu active", "status": "confirmed"},
    )

    out = render_view(nodes, "brd")

    assert "Phạm vi treo lại" in out
    # The parked section heading sits below the active content; the confirmed node precedes it.
    assert out.index("Mục tiêu active") < out.index("Phạm vi treo lại")
    assert "Parked" in out


def test_render_view_marks_needs_confirmation(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N5", "kind": "objective", "statement": "Cần xác nhận lại", "status": "needs_confirmation"},
    )

    out = render_view(nodes, "brd")

    assert "Cần xác nhận lại" in out
    assert "needs_confirmation" in out


def test_render_view_prd_includes_rules_section(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "D", "kind": "decision", "statement": "Trừ kho theo công thức", "status": "confirmed"},
        {"id": "F", "kind": "fact", "statement": "Mỗi ly có công thức cố định", "status": "confirmed"},
    )

    out = render_view(nodes, "prd")

    assert "Business Rules" in out
    assert "Trừ kho theo công thức" in out


def test_render_view_empty_graph_returns_valid_string():
    out = render_view({}, "brd")

    assert isinstance(out, str)
