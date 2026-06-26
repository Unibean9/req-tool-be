"""Decision graph — pure functions over dict[str, DecisionNode].

Two core invariants:
- Non-destructive: changing a decision creates a new node that supersedes the old one; the old node
  transitions to superseded and is never deleted (full history preserved).
- Mandatory ripple: superseding a node marks all transitive dependents stale (reconfirm) or parked (abandon).

All mutating functions return a new dict because LangGraph Command.update replaces decision_nodes
entirely — it does not merge nested dicts. get_dependents uses a visited-set guard so cyclic graphs
always terminate.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.graphs.state import DecisionNode

# Minimum dependents before a decision node is treated as direction-setting and cascade infers abandon.
ABANDON_THRESHOLD = 2
MAX_SWEEP_QUESTIONS = 5

VALID_KINDS = {"objective", "scope", "assumption", "decision", "risk", "open_question", "fact"}
VALID_STATUSES = {"proposed", "confirmed", "inferred", "needs_confirmation", "parked", "superseded"}

_VALID_CASCADE_MODES = {"reconfirm", "abandon"}
_RESOLVED_BLOCKER_STATUSES = {"confirmed", "inferred"}
_BRD_STABLE_STATUSES = {"confirmed", "inferred"}
_INACTIVE_STATUSES = {"parked", "superseded"}


def create_node(
    *,
    kind: str,
    statement: str,
    origin: dict[str, Any],
    depends_on: list[str] | None = None,
    status: str = "proposed",
    blocks: list[str] | None = None,
    supersedes: str | None = None,
    node_id: str | None = None,
) -> DecisionNode:
    """Build a complete DecisionNode; status defaults to proposed.

    node_id lets the caller assign a stable short id for later cross-references; omit to auto-generate uuid.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    return {
        "id": node_id or uuid.uuid4().hex,
        "kind": kind,
        "statement": statement,
        "status": status,
        "origin": dict(origin),
        "depends_on": list(depends_on or []),
        "supersedes": supersedes,
        "superseded_by": None,
        "blocks": list(blocks or []),
        "answer": None,
    }


def _clone(nodes: dict[str, DecisionNode]) -> dict[str, DecisionNode]:
    """Shallow-copy each node so mutations in the returned dict cannot leak back into the caller's input."""
    return {node_id: dict(node) for node_id, node in nodes.items()}


def update_node(nodes: dict[str, DecisionNode], node_id: str, **updates: Any) -> dict[str, DecisionNode]:
    """Patch status/statement/... on an existing node in-place; does not create a new node or supersede."""
    if node_id not in nodes:
        raise KeyError(f"node {node_id!r} not found")
    if nodes[node_id].get("status") == "superseded":
        raise ValueError(f"node {node_id!r} is superseded; history must not be rewritten")
    if "status" in updates and updates["status"] not in VALID_STATUSES:
        raise ValueError(f"invalid status: {updates['status']!r}")
    result = _clone(nodes)
    result[node_id].update(updates)
    return result


def get_dependents(
    nodes: dict[str, DecisionNode], node_id: str, visited: set[str] | None = None
) -> list[str]:
    """Return all nodes that transitively depend on node_id by following depends_on edges.

    visited-set prevents infinite loops: each node is visited at most once even in cyclic graphs.
    """
    if visited is None:
        visited = set()
    visited.add(node_id)
    dependents: list[str] = []
    for candidate_id, node in nodes.items():
        if candidate_id in visited:
            continue
        if node_id in node.get("depends_on", []):
            visited.add(candidate_id)
            dependents.append(candidate_id)
            dependents.extend(get_dependents(nodes, candidate_id, visited))
    return dependents


def infer_cascade_mode(nodes: dict[str, DecisionNode], node_id: str) -> str:
    """Infer cascade mode when the agent does not supply it explicitly.

    abandon when the node is a direction-setting decision (kind=decision, root or >= ABANDON_THRESHOLD
    dependents); reconfirm otherwise (local edit — the branch is still valid but must be re-confirmed).
    """
    node = nodes[node_id]
    dependent_count = len(get_dependents(nodes, node_id))
    is_direction = node.get("kind") == "decision" and (
        not node.get("depends_on") or dependent_count >= ABANDON_THRESHOLD
    )
    return "abandon" if is_direction else "reconfirm"


def supersede_node(
    nodes: dict[str, DecisionNode],
    old_id: str,
    new_statement: str,
    origin: dict[str, Any],
    cascade_mode: str | None = None,
) -> dict[str, DecisionNode]:
    """Reverse a decision non-destructively and ripple the change to dependents.

    Creates a new node that supersedes old_id; old_id transitions to superseded. cascade_mode defaults
    to the result of infer_cascade_mode (agent can override by passing explicitly): reconfirm marks
    dependents needs_confirmation (stale but recoverable); abandon marks them parked (branch suspended).
    """
    if old_id not in nodes:
        raise KeyError(f"node {old_id!r} not found")
    if cascade_mode is None:
        cascade_mode = infer_cascade_mode(nodes, old_id)
    if cascade_mode not in _VALID_CASCADE_MODES:
        raise ValueError(f"invalid cascade_mode: {cascade_mode!r}")

    dependents = get_dependents(nodes, old_id, visited=set())
    result = _clone(nodes)
    new_node = create_node(
        kind=result[old_id]["kind"],
        statement=new_statement,
        origin=origin,
        depends_on=list(result[old_id].get("depends_on", [])),
        supersedes=old_id,
    )
    result[new_node["id"]] = new_node
    result[old_id]["status"] = "superseded"
    result[old_id]["superseded_by"] = new_node["id"]

    dependent_status = "parked" if cascade_mode == "abandon" else "needs_confirmation"
    for dependent_id in dependents:
        result[dependent_id]["status"] = dependent_status
    return result


def scan_parked_questions(decision_nodes: dict[str, DecisionNode]) -> list[DecisionNode]:
    """Return parked open_questions whose every blocker has been resolved.

    A blocker is resolved when its status is confirmed or inferred. A parked question with no blocks
    never resurfaces automatically — there is no objective condition for the orchestrator to check.
    """
    resurfaced: list[DecisionNode] = []
    for node in decision_nodes.values():
        blocks = list(node.get("blocks") or [])
        if node.get("kind") != "open_question" or node.get("status") != "parked" or not blocks:
            continue
        if all(
            (decision_nodes.get(blocker_id) or {}).get("status") in _RESOLVED_BLOCKER_STATUSES
            for blocker_id in blocks
        ):
            resurfaced.append(node)
    return resurfaced


def is_brd_stable(decision_nodes: dict[str, DecisionNode]) -> bool:
    """BRD is stable when every active node is confirmed or inferred; parked and superseded do not count."""
    active = [
        node
        for node in decision_nodes.values()
        if node.get("status") not in _INACTIVE_STATUSES
    ]
    return bool(active) and all(node.get("status") in _BRD_STABLE_STATUSES for node in active)


def _normalize_statement(value: str) -> str:
    return " ".join(str(value or "").lower().strip().split())


_BRD_SWEEP_GAPS: tuple[tuple[str, str], ...] = (
    ("objective", "Cần chốt mục tiêu đo được cho BRD."),
    ("scope", "Cần chốt phạm vi v1 cho BRD."),
    ("assumption", "Cần ghi rõ giả định chính của BRD."),
    ("risk", "Cần ghi rủi ro chính của BRD."),
)

_PRD_SWEEP_GAPS: tuple[tuple[str, str], ...] = (
    ("actor", "Actor: xác định khách hàng và nhân viên thao tác trong luồng."),
    ("flow", "Luồng chính: mô tả từng bước xử lý từ đầu đến cuối."),
    ("rule", "Business rule: chốt quy tắc tích điểm và điều kiện tính điểm."),
    ("edge_case", "Edge-case: khách quên SĐT lúc mua → cộng bù sau được không?"),
    ("edge_case", "Edge-case: khách đổi SĐT → gộp lịch sử thế nào?"),
    ("edge_case", "Edge-case: phiếu free hết hạn không?"),
)


def _gap_present(decision_nodes: dict[str, DecisionNode], marker: str) -> bool:
    active_nodes = [
        node for node in decision_nodes.values()
        if node.get("status") not in {"parked", "superseded"}
    ]
    if marker in VALID_KINDS:
        return any(node.get("kind") == marker for node in active_nodes)
    marker_text = marker.replace("_", " ")
    return any(marker_text in _normalize_statement(node.get("statement", "")) for node in active_nodes)


def completeness_sweep(
    decision_nodes: dict[str, DecisionNode],
    artifact_type: str,
    max_questions: int = MAX_SWEEP_QUESTIONS,
) -> list[str]:
    """Check the minimum coverage checklist and return gap descriptions, deduplicated by exact statement.

    Only produces descriptions; the caller decides whether to create parked nodes or inject into the
    prompt. Dedup is exact (normalized statement match), not LLM similarity.
    """
    template = _PRD_SWEEP_GAPS if artifact_type == "prd" else _BRD_SWEEP_GAPS
    existing_questions = {
        _normalize_statement(node.get("statement", ""))
        for node in decision_nodes.values()
        if node.get("kind") == "open_question"
    }
    gaps: list[str] = []
    for marker, question in template:
        normalized = _normalize_statement(question)
        if normalized in existing_questions:
            continue
        if _gap_present(decision_nodes, marker):
            continue
        gaps.append(question)
        if len(gaps) >= max_questions:
            break
    return gaps


def add_parked_questions_for_gaps(
    decision_nodes: dict[str, DecisionNode],
    gaps: list[str],
    origin: dict[str, Any],
) -> tuple[dict[str, DecisionNode], list[DecisionNode]]:
    """Create parked open_question nodes from completeness_sweep gaps without blocking the main flow."""
    result = _clone(decision_nodes)
    created: list[DecisionNode] = []
    existing = {
        _normalize_statement(node.get("statement", ""))
        for node in result.values()
        if node.get("kind") == "open_question"
    }
    for gap in gaps:
        normalized = _normalize_statement(gap)
        if normalized in existing:
            continue
        node = create_node(
            kind="open_question",
            statement=gap,
            origin=origin,
            status="parked",
            blocks=[],
        )
        result[node["id"]] = node
        created.append(node)
        existing.add(normalized)
    return result, created


def _link_value(link: Any, *names: str) -> str | None:
    for name in names:
        if isinstance(link, dict) and link.get(name) is not None:
            return str(link[name])
        value = getattr(link, name, None)
        if value is not None:
            return str(value)
    return None


def _reachable_artifacts(artifact_links: list[Any], changed_artifact_id: str | None) -> tuple[list[str], list[str]]:
    if not artifact_links:
        return [], []
    start = str(changed_artifact_id) if changed_artifact_id else _link_value(
        artifact_links[0], "source_id", "source_artifact_id"
    )
    if not start:
        return [], []
    visited = {start}
    stale: list[str] = []
    queue = [start]
    while queue:
        current = queue.pop(0)
        for link in artifact_links:
            source = _link_value(link, "source_id", "source_artifact_id")
            target = _link_value(link, "target_id", "target_artifact_id")
            if source != current or not target or target in visited:
                continue
            visited.add(target)
            stale.append(target)
            queue.append(target)
    return stale, list(visited)


def _default_impact_selector(change_description: str, decision_nodes: dict[str, DecisionNode]) -> list[str]:
    normalized_change = _normalize_statement(change_description)
    tokens = {token for token in normalized_change.split() if len(token) >= 4}
    affected: list[str] = []
    for node_id, node in decision_nodes.items():
        statement = _normalize_statement(node.get("statement", ""))
        if tokens and any(token in statement for token in tokens):
            affected.append(node_id)
            continue
        if any(token in normalized_change for token in ("giao", "kênh", "kenh", "delivery", "đa kênh")) and any(
            token in statement for token in ("thu ngân", "tai quay", "tại quầy", "khách/ngày", "1 ghé")
        ):
            affected.append(node_id)
    return affected


def _llm_selected_ids(
    llm: Any,
    change_description: str,
    decision_nodes: dict[str, DecisionNode],
    stale_artifact_ids: list[str],
) -> list[str] | None:
    if llm is None:
        return None
    result = llm(change_description, decision_nodes, stale_artifact_ids) if callable(llm) else None
    if isinstance(result, dict):
        result = result.get("affected_node_ids") or result.get("node_ids")
    if result is None:
        return None
    return [str(item) for item in result]


def impact(
    change_description: str,
    decision_nodes: dict[str, DecisionNode],
    artifact_links: list[Any],
    llm: Any = None,
    changed_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Mark nodes affected by a cross-artifact change as needs_confirmation.

    The LLM/callable only selects affected ids; this function enforces the invariants: no statement
    rewrite, no touching nodes outside the list, artifact-link traversal uses a visited-set guard.
    """
    stale_artifact_ids, visited_artifact_ids = _reachable_artifacts(artifact_links, changed_artifact_id)
    selected = _llm_selected_ids(llm, change_description, decision_nodes, stale_artifact_ids)
    affected_ids = selected if selected is not None else _default_impact_selector(change_description, decision_nodes)
    affected_ids = [
        node_id for node_id in dict.fromkeys(affected_ids)
        if node_id in decision_nodes and decision_nodes[node_id].get("status") != "superseded"
    ]
    updated = _clone(decision_nodes)
    for node_id in affected_ids:
        updated[node_id]["status"] = "needs_confirmation"
    return {
        "decision_nodes": updated,
        "affected_node_ids": affected_ids,
        "stale_artifact_ids": stale_artifact_ids,
        "visited_artifact_ids": visited_artifact_ids,
    }


def park_sync_debt(
    decision_nodes: dict[str, DecisionNode],
    question: str,
    affected_node_ids: list[str],
    origin: dict[str, Any],
) -> tuple[dict[str, DecisionNode], DecisionNode]:
    """Record a sync debt as a parked open_question whose blocks point to the stale nodes."""
    node = create_node(
        kind="open_question",
        statement=question,
        origin=origin,
        status="parked",
        blocks=[node_id for node_id in affected_node_ids if node_id in decision_nodes],
    )
    updated = {**_clone(decision_nodes), node["id"]: node}
    return updated, node


# Status sets driving the projection: superseded never renders; parked folds into its own section;
# everything else is "active" and renders into the body.
_ACTIVE_STATUSES = {"confirmed", "inferred", "needs_confirmation"}

# Per-artifact section layout: ordered (heading, kinds-in-it). brd is requirement-shaped; prd folds
# decision+fact into a Business Rules section. Two explicit templates so they diverge cleanly later.
_BRD_SECTIONS = [
    ("Vision & Objectives", ("objective",)),
    ("Scope", ("scope",)),
    ("Assumptions", ("assumption",)),
    ("Decisions", ("decision",)),
    ("Risks", ("risk",)),
    ("Open Questions", ("open_question",)),
    ("Facts", ("fact",)),
]
_PRD_SECTIONS = [
    ("Objectives", ("objective",)),
    ("Scope", ("scope",)),
    ("Business Rules", ("decision", "fact")),
    ("Assumptions", ("assumption",)),
    ("Risks", ("risk",)),
    ("Open Questions", ("open_question",)),
]
_SECTION_TEMPLATES = {"prd": _PRD_SECTIONS}


def _render_line(node: dict[str, Any]) -> str:
    marker = " ⟨needs_confirmation⟩" if node.get("status") == "needs_confirmation" else ""
    return f"- {node.get('statement', '')}{marker}"


def render_view(decision_nodes: dict[str, DecisionNode], artifact_type: str) -> str:
    """Project the decision graph to markdown — a derived view, never the source of truth.

    Renders active nodes (confirmed/inferred/needs_confirmation) grouped into the artifact's sections,
    hides superseded entirely, and folds parked nodes into a trailing Parked section. Empty graph →
    a valid (mostly empty) string, never a crash.
    """
    sections = _SECTION_TEMPLATES.get(artifact_type, _BRD_SECTIONS)
    active = [n for n in decision_nodes.values() if n.get("status") in _ACTIVE_STATUSES]
    parked = [n for n in decision_nodes.values() if n.get("status") == "parked"]

    blocks: list[str] = []
    for heading, kinds in sections:
        lines = [_render_line(n) for n in active if n.get("kind") in kinds]
        if lines:
            blocks.append(f"## {heading}\n" + "\n".join(lines))

    if parked:
        lines = [_render_line(n) for n in parked]
        blocks.append("## Parked (tạm gác)\n" + "\n".join(lines))

    return "\n\n".join(blocks)
