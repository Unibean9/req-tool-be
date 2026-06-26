"""Đồ thị quyết định — các hàm thuần thao tác trên dict[str, DecisionNode].

Hai bất biến cốt lõi:
- Không destructive: đổi quyết định = tạo node mới supersedes node cũ; node cũ chuyển superseded, không xóa.
- Ripple bắt buộc: supersede một node → các node phụ thuộc bị đánh dấu stale (reconfirm) hoặc treo (abandon).

Mọi hàm mutate trả về dict mới (immutable) vì LangGraph Command.update thay nguyên decision_nodes,
không merge nested. get_dependents dùng visited-set guard nên graph có cycle vẫn kết thúc hữu hạn.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.graphs.state import DecisionNode

# Số dependent tối thiểu để coi một node decision là "định hướng" và suy ra cascade abandon.
ABANDON_THRESHOLD = 2

VALID_KINDS = {"objective", "scope", "assumption", "decision", "risk", "open_question", "fact"}
VALID_STATUSES = {"proposed", "confirmed", "inferred", "needs_confirmation", "parked", "superseded"}

_VALID_CASCADE_MODES = {"reconfirm", "abandon"}


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
    """Tạo một DecisionNode hoàn chỉnh; mặc định status=proposed.

    node_id cho phép caller đặt id ổn định, ngắn gọn để tham chiếu sau này; bỏ trống → sinh uuid.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"status không hợp lệ: {status!r}")
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
    """Deep-enough copy: mỗi node được sao riêng để mutation không rò ngược lên input."""
    return {node_id: dict(node) for node_id, node in nodes.items()}


def update_node(nodes: dict[str, DecisionNode], node_id: str, **updates: Any) -> dict[str, DecisionNode]:
    """Cập nhật tại chỗ status/statement/... của một node; không tạo node mới, không supersede."""
    if node_id not in nodes:
        raise KeyError(f"node {node_id!r} không tồn tại")
    if nodes[node_id].get("status") == "superseded":
        raise ValueError(f"node {node_id!r} đã superseded; không được sửa lịch sử")
    if "status" in updates and updates["status"] not in VALID_STATUSES:
        raise ValueError(f"status không hợp lệ: {updates['status']!r}")
    result = _clone(nodes)
    result[node_id].update(updates)
    return result


def get_dependents(
    nodes: dict[str, DecisionNode], node_id: str, visited: set[str] | None = None
) -> list[str]:
    """Trả về các node phụ thuộc (transitive) vào node_id, đi ngược cạnh depends_on.

    visited-set guard chống cycle: một node chỉ được duyệt một lần nên graph có vòng vẫn kết thúc.
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
    """Suy ra cascade mode khi agent không truyền tường minh.

    abandon nếu node là quyết định định hướng (kind=decision và là root hoặc nhiều dependent);
    còn lại reconfirm (chỉnh sửa cục bộ, nhánh cũ vẫn valid nhưng cần re-confirm).
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
    """Đảo một quyết định không destructive + ripple xuống dependent.

    Tạo node mới supersedes old_id; old_id → superseded. cascade_mode khi None được suy ra từ
    infer_cascade_mode (agent override được bằng cách truyền tường minh): reconfirm → dependent thành
    needs_confirmation (stale), abandon → dependent thành parked (nhánh treo, recoverable).
    """
    if old_id not in nodes:
        raise KeyError(f"node {old_id!r} không tồn tại")
    if cascade_mode is None:
        cascade_mode = infer_cascade_mode(nodes, old_id)
    if cascade_mode not in _VALID_CASCADE_MODES:
        raise ValueError(f"cascade_mode không hợp lệ: {cascade_mode!r}")

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
