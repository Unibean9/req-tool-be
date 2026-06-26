"""Phase 05 FIX 2 — draft_body precedence: the rendered graph view wins over any stored body.

A stale draft_body from a prior session must never shadow the live decision graph. Both the async
canonical reader (critique input) and the sync cached reader (finalize-gate hash) honor this.
"""

import pytest

from app.graphs.agent_tools import _cached_draft_body, current_draft_body
from app.graphs.decision_graph import render_view


@pytest.mark.asyncio
async def test_current_draft_body_returns_rendered_view_when_nodes_present(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N3", "kind": "objective", "statement": "Mục tiêu từ graph", "status": "confirmed"},
    )
    state = {"decision_nodes": nodes, "draft_body": "BODY DB CŨ STALE", "artifact_type": "brd"}

    body = await current_draft_body(state)

    assert body == render_view(nodes, "brd")
    assert "STALE" not in body


@pytest.mark.asyncio
async def test_current_draft_body_falls_back_when_nodes_empty():
    state = {"decision_nodes": {}, "draft_body": "BODY DB", "artifact_type": "brd"}

    body = await current_draft_body(state)

    assert body == "BODY DB"


def test_cached_draft_body_returns_view_when_nodes_present(decision_graph_factory):
    # The finalize-gate hash routes through _cached_draft_body, so it must score the view too.
    nodes = decision_graph_factory(
        {"id": "N3", "kind": "objective", "statement": "View để chấm", "status": "confirmed"},
    )
    state = {"decision_nodes": nodes, "draft_body": "STALE", "artifact_type": "brd"}

    assert _cached_draft_body(state) == render_view(nodes, "brd")


def test_cached_draft_body_falls_back_when_nodes_empty():
    state = {"decision_nodes": {}, "draft_body": "BODY DB"}

    assert _cached_draft_body(state) == "BODY DB"
