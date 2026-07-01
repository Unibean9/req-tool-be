"""render_view pure projection from the decision graph to markdown.

The view is derived, never the source: superseded hidden, parked folded into its own section,
active nodes (confirmed/inferred/needs_confirmation) shown. brd and prd use distinct templates.
"""

from app.graphs.decision_graph import create_node, render_view


def test_render_view_shows_confirmed_and_inferred(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N3", "kind": "objective", "statement": "Reduce loss", "status": "confirmed"},
        {"id": "N4", "kind": "scope", "statement": "Quan ly cong thuc", "status": "inferred"},
    )

    out = render_view(nodes, "brd")

    assert "Reduce loss" in out
    assert "Quan ly cong thuc" in out


def test_render_view_hides_superseded(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "statement": "Huong cu", "status": "superseded"},
        {"id": "N3", "kind": "objective", "statement": "Goal chot", "status": "confirmed"},
    )

    out = render_view(nodes, "brd")

    assert "Huong cu" not in out


def test_render_view_folds_parked_into_separate_section(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "P", "kind": "scope", "statement": "Pham vi treo lai", "status": "parked"},
        {"id": "C", "kind": "objective", "statement": "Goal active", "status": "confirmed"},
    )

    out = render_view(nodes, "brd")

    assert "Pham vi treo lai" in out
    # The parked section heading sits below the active content; the confirmed node precedes it.
    assert out.index("Goal active") < out.index("Pham vi treo lai")
    assert "Parked" in out


def test_render_view_marks_needs_confirmation(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N5", "kind": "objective", "statement": "Needs reconfirmation", "status": "needs_confirmation"},
    )

    out = render_view(nodes, "brd")

    assert "Needs reconfirmation" in out
    assert "needs_confirmation" in out


def test_render_view_prd_includes_rules_section(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "D", "kind": "decision", "statement": "Tru kho theo cong thuc", "status": "confirmed"},
        {"id": "F", "kind": "fact", "statement": "Moi ly co cong thuc co dinh", "status": "confirmed"},
    )

    out = render_view(nodes, "prd")

    assert "Business Rules" in out
    assert "Tru kho theo cong thuc" in out


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


def test_render_view_functional_requirement_assigns_trace_ids_and_placeholders():
    nodes = {
        "F1": create_node(
            kind="decision",
            statement="Scan QR to check in",
            origin={"source": "test"},
            status="confirmed",
            node_id="F1",
            fields={"requirement": "Scan rotating QR", "behavior": "Validate within 30s", "priority": "Must"},
        ),
        "F2": create_node(
            kind="decision",
            statement="Verify GPS",
            origin={"source": "test"},
            status="confirmed",
            node_id="F2",
            fields={"requirement": "Check geofence"},
        ),
    }

    out = render_view(nodes, "functional_requirement")

    assert "## Functional Requirements" in out
    assert "| id | requirement | behavior | inputs/outputs | acceptance signal | priority |" in out
    # First column is an auto-assigned trace tag the agent never supplies.
    assert "| FR-01 |" in out
    assert "| FR-02 |" in out
    # Columns the node left unfilled render a visible placeholder so the gate can block them.
    assert "_(cần bổ sung)_" in out


def test_render_view_business_capability_renders_id_tagged_entries():
    nodes = {
        "U1": create_node(
            kind="fact",
            statement="Attendance check-in",
            origin={"source": "test"},
            status="confirmed",
            node_id="U1",
            fields={"goal": "Let students check in fast", "user_segment": "Students"},
        ),
    }

    out = render_view(nodes, "use_case")

    assert "## Business Capabilities" in out
    assert "### BC-01: Attendance check-in" in out
    assert "- **goal:** Let students check in fast" in out
    assert "- **user_segment:** Students" in out
    # Unfilled brief fields still placeholder so the gate can flag them.
    assert "- **business_value:** _(cần bổ sung)_" in out
