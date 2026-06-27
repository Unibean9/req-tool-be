"""render_view pure projection from the decision graph to markdown.

The view is derived, never the source: superseded hidden, parked folded into its own section,
active nodes (confirmed/inferred/needs_confirmation) shown. brd and prd use distinct templates.
"""

from app.graphs.decision_graph import create_node, render_view


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


def test_render_view_document_item_uses_contract_headings_and_table():
    nodes = {
        "V1": create_node(
            kind="objective",
            statement="Help students schedule study groups faster.",
            origin={"source": "test"},
            status="confirmed",
            node_id="V1",
            section="## Vision",
        ),
        "O1": create_node(
            kind="objective",
            statement="Reduce schedule agreement time below 10 minutes.",
            origin={"source": "test"},
            status="confirmed",
            node_id="O1",
            section="## Objectives",
        ),
        "M1": create_node(
            kind="objective",
            statement="Measure successful group scheduling rate.",
            origin={"source": "test"},
            status="needs_confirmation",
            node_id="M1",
            section="## Success Metrics",
            fields={
                "goal": "Schedule study groups",
                "user/business value": "Students reduce coordination loops",
                "metric": "Successful scheduling rate",
                "target": "80%",
                "timeframe": "First semester",
            },
        ),
    }

    out = render_view(nodes, "vision_objectives")

    assert "## Vision" in out
    assert "## Objectives" in out
    assert "## Success Metrics" in out
    assert "## Vision & Objectives" not in out
    assert "| goal | user/business value | metric | target | timeframe |" in out
    assert "| Schedule study groups | Students reduce coordination loops | Successful scheduling rate | 80% | First semester ⚠️ needs confirmation |" in out
    assert out.index("## Vision") < out.index("## Objectives") < out.index("## Success Metrics")
