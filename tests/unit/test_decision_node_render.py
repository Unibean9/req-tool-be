"""render_view projection contract.

render_view projects the decision graph to markdown: superseded hidden, parked folded
into a separate section, active nodes shown. The view is derived, never the source.
"""

import pytest

try:
    from app.graphs.decision_graph import render_view

    _PENDING = None
except ImportError as exc:  # pragma: no cover - resolves once decision_graph lands
    render_view = None
    _PENDING = str(exc)

pytestmark = pytest.mark.xfail(
    _PENDING is not None,
    reason=f"render_view not yet available: {_PENDING}",
    strict=False,
)


def test_view_excludes_superseded_nodes(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "statement": "Huong cu da bo", "status": "superseded"},
        {"id": "N3", "kind": "objective", "statement": "Goal being confirmed", "status": "confirmed"},
        {"id": "N4", "kind": "scope", "statement": "Pham vi treo lai", "status": "parked"},
    )

    out = render_view(nodes, "brd")

    assert "Huong cu da bo" not in out
    assert "Goal being confirmed" in out
    assert "Pham vi treo lai" in out
