"""Pure builders for DecisionNode test fixtures (golden TDD harness).

Build plain dicts matching the DecisionNode schema in
plans/harness-refactor/golden-conversation.md (Phần 0). Deliberately do NOT import
from app.graphs.decision_graph: that module ships in Phase 04, and the harness must
collect and fail red before it exists.
"""

from itertools import count
from typing import Any

VALID_KINDS = {
    "objective",
    "scope",
    "assumption",
    "decision",
    "risk",
    "open_question",
    "fact",
}
VALID_STATUSES = {
    "proposed",
    "confirmed",
    "inferred",
    "needs_confirmation",
    "parked",
    "superseded",
}


def make_decision_node(
    node_id: str,
    *,
    kind: str = "objective",
    statement: str = "",
    status: str = "proposed",
    origin: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
    supersedes: str | None = None,
    superseded_by: str | None = None,
    blocks: list[str] | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    """Construct one schema-complete DecisionNode dict.

    Validates kind/status so a typo in a test surfaces as an error here rather than a
    misleading downstream assertion failure.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"invalid kind {kind!r}; expected one of {sorted(VALID_KINDS)}")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}")
    return {
        "id": node_id,
        "kind": kind,
        "statement": statement,
        "status": status,
        "origin": origin or {"turn": 0, "by": "user", "technique": None, "source": None},
        "depends_on": list(depends_on or []),
        "supersedes": supersedes,
        "superseded_by": superseded_by,
        "blocks": list(blocks or []),
        "answer": answer,
    }


def make_node_factory():
    """Return a callable that mints schema-complete nodes with auto-incremented ids.

    id is overridable so a test can pin a stable name (N1, Q4) when assertions reference it.
    """
    counter = count(1)

    def _make(**overrides: Any) -> dict[str, Any]:
        node_id = overrides.pop("id", None) or f"N{next(counter)}"
        return make_decision_node(node_id, **overrides)

    return _make


def graph_from(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index a list of nodes by id into the decision_nodes dict shape."""
    return {node["id"]: node for node in nodes}
