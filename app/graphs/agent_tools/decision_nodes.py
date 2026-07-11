"""Decision graph — create / update / supersede / dismiss nodes (flag-gated).

The decision graph is the source of truth for the artifact view. These tools mutate
decision_nodes in state (no DB, no interrupt) via Command.update — the whole dict is replaced
each call because LangGraph does not merge nested state. All writes are behind
DECISION_GRAPH_ENABLED so an in-progress graph model never leaks into a persisted checkpoint.

Graph mutation logic lives in app.graphs.decision_graph; this module is the tool surface over it.
"""

import logging
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.config import settings
from app.graphs.agent_tools._shared import (
    RecoverableToolError,
    _missing_required_arg_update,
    _node_origin,
    _recoverable_tool_update,
    _tool_not_available_update,
)
from app.graphs.decision_graph import (
    VALID_KINDS,
    VALID_STATUSES,
    create_node,
    dismiss_node,
    get_dependents,
    infer_cascade_mode,
    supersede_node,
    update_node,
)
from app.graphs.state import WorkflowState

logger = logging.getLogger(__name__)

_TOOL_EDITABLE_STATUSES = VALID_STATUSES - {"superseded"}


def _decision_graph_off_update(tool_name: str, tool_call_id: str) -> Command:
    logger.warning("tool=%s skipped: DECISION_GRAPH_ENABLED is off", tool_name)
    return _tool_not_available_update(
        tool_name, "decision graph is disabled (DECISION_GRAPH_ENABLED=false)", tool_call_id
    )


def _section_content(statement: str | None, fields: dict[str, str] | None) -> str:
    """Text a section write exposes to validation: the statement plus any table-cell values."""
    parts = [str(statement or "")]
    if fields:
        parts.extend(str(value) for value in fields.values())
    return "\n".join(part for part in parts if part)


def _section_findings_update(state: WorkflowState, node: dict[str, Any]) -> dict[str, Any]:
    """Validate the section a decision-node write touched and record findings.

    Never blocks the write. Keyed by section heading, replace-on-write per key: a passing section
    stores [] so a re-validation clears a prior defect through the merge reducer. A node with no
    section has nothing to validate, so the findings map is left untouched.
    """
    section = node.get("section")
    if not section:
        return {}
    from app.graphs.analysis.section_validation import validate_section

    findings = validate_section(
        state.get("artifact_type") or "brd", section, _section_content(node.get("statement"), node.get("fields"))
    )
    return {"section_findings": {**(state.get("section_findings") or {}), section: findings}}


async def _create_decision_node_impl(
    kind: str,
    statement: str,
    depends_on: list[str] | None,
    technique: str | None,
    state: WorkflowState,
    tool_call_id: str,
    node_id: str | None = None,
    status: str | None = None,
    blocks: list[str] | None = None,
    section: str | None = None,
    fields: dict[str, str] | None = None,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("create_decision_node", tool_call_id)
    if kind not in VALID_KINDS:
        return _tool_not_available_update(
            "create_decision_node", f"invalid kind '{kind}'; choose one of {sorted(VALID_KINDS)}", tool_call_id
        )
    if not str(statement or "").strip():
        return _missing_required_arg_update("create_decision_node", "statement", tool_call_id)
    if status is not None and status not in _TOOL_EDITABLE_STATUSES:
        return _tool_not_available_update(
            "create_decision_node",
            f"invalid status '{status}'; choose one of {sorted(_TOOL_EDITABLE_STATUSES)}",
            tool_call_id,
        )

    nodes_state = state.get("decision_nodes") or {}
    unknown = [dep for dep in (depends_on or []) if dep not in nodes_state]
    if unknown:
        return _tool_not_available_update(
            "create_decision_node", f"depends_on points to missing nodes: {unknown}", tool_call_id
        )
    node = create_node(
        kind=kind,
        statement=statement,
        origin=_node_origin(state, technique),
        depends_on=depends_on,
        node_id=node_id,
        status=status or "proposed",
        blocks=blocks,
        section=section,
        fields=fields,
    )
    if node["id"] in nodes_state:
        return _tool_not_available_update(
            "create_decision_node",
            f"node_id '{node['id']}' already exists; use update/supersede instead of overwriting",
            tool_call_id,
        )
    updated = {**nodes_state, node["id"]: node}
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=f"node {node['id']} ({kind})", tool_call_id=tool_call_id)],
            **_section_findings_update(state, node),
        }
    )


async def _update_decision_node_impl(
    node_id: str,
    status: str | None,
    statement: str | None,
    state: WorkflowState,
    tool_call_id: str,
    section: str | None = None,
    fields: dict[str, str] | None = None,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("update_decision_node", tool_call_id)
    nodes_state = state.get("decision_nodes") or {}
    if node_id not in nodes_state:
        return _tool_not_available_update("update_decision_node", f"node '{node_id}' does not exist", tool_call_id)
    if nodes_state[node_id].get("status") == "superseded":
        return _tool_not_available_update(
            "update_decision_node",
            f"node '{node_id}' is superseded; use the replacement node for further edits",
            tool_call_id,
        )
    if status is not None and status not in _TOOL_EDITABLE_STATUSES:
        return _tool_not_available_update(
            "update_decision_node",
            f"invalid status '{status}'; choose one of {sorted(_TOOL_EDITABLE_STATUSES)}",
            tool_call_id,
        )

    updates = {
        k: v
        for k, v in {
            "status": status,
            "statement": statement,
            "section": section,
            "fields": fields,
        }.items()
        if v is not None
    }
    if not updates:
        return _missing_required_arg_update("update_decision_node", "status or statement", tool_call_id)
    updated = update_node(nodes_state, node_id, **updates)
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=f"node {node_id} updated", tool_call_id=tool_call_id)],
            **_section_findings_update(state, updated[node_id]),
        }
    )


async def _supersede_decision_node_impl(
    node_id: str,
    new_statement: str,
    cascade_mode: str | None,
    state: WorkflowState,
    tool_call_id: str,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("supersede_decision_node", tool_call_id)
    nodes_state = state.get("decision_nodes") or {}
    if node_id not in nodes_state:
        return _tool_not_available_update("supersede_decision_node", f"node '{node_id}' does not exist", tool_call_id)
    if not str(new_statement or "").strip():
        return _missing_required_arg_update("supersede_decision_node", "new_statement", tool_call_id)
    if cascade_mode is not None and cascade_mode not in {"reconfirm", "abandon"}:
        return _tool_not_available_update(
            "supersede_decision_node", f"invalid cascade_mode '{cascade_mode}'", tool_call_id
        )

    resolved_mode = cascade_mode or infer_cascade_mode(nodes_state, node_id)
    rippled = [d for d in get_dependents(nodes_state, node_id) if nodes_state[d].get("status") != "superseded"]
    updated = supersede_node(nodes_state, node_id, new_statement, _node_origin(state, None), resolved_mode)
    new_id = updated[node_id]["superseded_by"]
    summary = f"superseded {node_id} → {new_id}; cascade={resolved_mode}; dependents={rippled or 'none'}"
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
            **_section_findings_update(state, updated[new_id]),
        }
    )


@tool
async def create_decision_node(
    kind: Annotated[str, "objective | scope | assumption | decision | risk | open_question | fact"],
    statement: Annotated[str, "The decision/objective/etc. as a single clear statement, in the user's locale."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    depends_on: Annotated[list[str] | None, "ids of nodes this one builds on; [] for a root."] = None,
    technique: Annotated[str | None, "Elicitation technique that produced this node, for provenance."] = None,
    node_id: Annotated[str | None, "Optional stable short id (e.g. N1); auto-generated if omitted."] = None,
    status: Annotated[
        str | None,
        "Optional initial status: proposed|confirmed|inferred|needs_confirmation|parked. Defaults to proposed.",
    ] = None,
    blocks: Annotated[
        list[str] | None,
        "For parked open_question nodes: ids currently blocked by this question.",
    ] = None,
    section: Annotated[
        str | None,
        "Target heading in the output contract, e.g. '## Objectives' or '## Success Metrics'.",
    ] = None,
    fields: Annotated[
        dict[str, str] | None,
        "Column values when the section renders as a table; keys must match the output contract table_columns.",
    ] = None,
) -> Command:
    """Record a new decision-graph node (objective, decision, risk, ...) with provenance.

    Use to capture a piece of analysis as durable state instead of prose in a draft. Each node keeps
    why it exists (origin) and what it builds on (depends_on) so later reversals can ripple correctly.
    """
    return await _create_decision_node_impl(
        kind, statement, depends_on, technique, state, tool_call_id, node_id, status, blocks, section, fields
    )


@tool
async def update_decision_node(
    node_id: Annotated[str, "id of the node to update."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    status: Annotated[str | None, "New status: proposed|confirmed|inferred|needs_confirmation|parked."] = None,
    statement: Annotated[str | None, "Revised statement (a local edit, NOT a reversal)."] = None,
    section: Annotated[
        str | None,
        "New target heading in the output contract, e.g. '## Objectives' or '## Success Metrics'.",
    ] = None,
    fields: Annotated[
        dict[str, str] | None,
        "New column values for the section table; keys must match the output contract table_columns.",
    ] = None,
) -> Command:
    """Update a node's status or statement in place — a local edit that does not rewrite history.

    Use to confirm a proposed node or refine its wording. To reverse a decision (keep history + ripple
    to dependents) use supersede_decision_node instead.
    """
    return await _update_decision_node_impl(node_id, status, statement, state, tool_call_id, section, fields)


@tool
async def supersede_decision_node(
    node_id: Annotated[str, "id of the node being reversed."],
    new_statement: Annotated[str, "The replacement decision/objective, in the user's locale."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    cascade_mode: Annotated[
        str | None,
        "How dependents ripple: 'reconfirm' (local edit, dependents need re-confirm) or 'abandon' "
        "(direction reversal, dependents parked). Omit to infer: a root decision node or one with "
        "many dependents → abandon; otherwise reconfirm. Pass explicitly when you know the intent.",
    ] = None,
) -> Command:
    """Reverse a decision without destroying history and ripple the change to dependents.

    Creates a new node that supersedes the old one (old → superseded, kept for provenance) and marks
    every dependent stale (reconfirm) or parked (abandon). Use for a genuine change of direction; use
    update_decision_node for a local wording/status edit.
    """
    return await _supersede_decision_node_impl(node_id, new_statement, cascade_mode, state, tool_call_id)


async def _dismiss_question_impl(
    node_id: str,
    reason: str,
    state: WorkflowState,
    tool_call_id: str,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("dismiss_question", tool_call_id)
    if not str(node_id or "").strip():
        return _missing_required_arg_update("dismiss_question", "node_id", tool_call_id)
    if not str(reason or "").strip():
        return _missing_required_arg_update("dismiss_question", "reason", tool_call_id)

    nodes_state = state.get("decision_nodes") or {}
    node = nodes_state.get(node_id)
    if node is None or node.get("kind") != "open_question":
        return _recoverable_tool_update(
            RecoverableToolError(
                code="dismiss_target_invalid",
                message=f"Cannot dismiss '{node_id}': not a known open_question node.",
                recovery="Pass the id of a parked open_question from the feedback signals.",
            ),
            tool_call_id,
        )

    origin = _node_origin(state, None)
    updated = dismiss_node(nodes_state, node_id, reason, origin)
    # SECURITY: `reason` is agent/user-supplied text persisted as data on the node's audit trail
    # (dismiss_node). The confirmation below is a fixed string — the reason is never echoed back as
    # an instruction.
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=f"Dismissed {node_id}.", tool_call_id=tool_call_id)],
        }
    )


@tool
async def dismiss_question(
    node_id: Annotated[str, "id of the parked open_question node to dismiss."],
    reason: Annotated[str, "Substantive reason for dismissing the question, for the audit trail."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Dismiss a resurfaced open_question with an auditable reason instead of answering it.

    Use when a parked question is no longer worth pursuing (superseded by other decisions, out of
    scope, etc.) — not an escape hatch for work you should still do. The dismissal (reason + turn) is
    recorded on the node and the question is removed from the resurfacing/blocker set.
    """
    return await _dismiss_question_impl(node_id, reason, state, tool_call_id)
