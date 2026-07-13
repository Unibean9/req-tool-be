"""note — scratchpad note (no interrupt, no DB, no approval).

`critique_note`/`explore_note` used to be two separate tools for two lenses (critique /
exploration), but both were thin wrappers over the same `_write_note_impl` with no distinct
branching and no classification persisted to state — so they were merged into one shared `note`
tool. The two old functions are kept as deprecated aliases so `ToolNode` can still re-execute an
old tool_call when resuming a checkpoint created before the merge; they no longer appear in the
tool menu given to the model.

`_tool_is_available` (the note step-limit guard) is registry-backed and lives in the coordinator;
reach it through the module reference at call time to avoid an import cycle.
"""

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.graphs import agent_tools
from app.graphs.agent_tools._shared import (
    _missing_required_arg_update,
    _node_origin,
    _tool_not_available_update,
)
from app.graphs.decision_graph import create_node
from app.graphs.note_parser import extract_structured_objects
from app.graphs.state import WorkflowState


async def _write_note_impl(content: str, state: WorkflowState, tool_call_id: str, tool_name: str):
    if not str(content or "").strip():
        return _missing_required_arg_update(tool_name, "content", tool_call_id, state.get("locale"))
    # Availability always looks up the canonical name "note" — "critique_note"/"explore_note" were
    # removed from get_available_tools's candidates (kept in the registry only for resume), so
    # looking up the old tool_name would always return False even when state allows writing a
    # note, wrongly rejecting a resumed old note call (no crash, but the note content is lost).
    if not agent_tools._tool_is_available(state, "note"):
        return _tool_not_available_update(
            tool_name,
            "note step limit reached; ask the user, respond, or switch tools instead of adding more notes.",
            tool_call_id,
            state.get("locale"),
        )

    # The note text lives in the message history (decision 3): no `notes` state field, no DB row.
    # Beyond that, tagged lines are parsed into structured objects. risks/key_facts carry additive
    # reducers, so emit ONLY the new entries — returning prior+new would re-append every existing
    # entry through the reducer. ASSUMPTION:/OPEN_QUESTION: lines instead create decision nodes
    # because the graph is the single source of truth for assumptions/open-questions — assumptions
    # land needs_confirmation (agent-authored, not user-confirmed), open_questions proposed.
    extracted = extract_structured_objects(content)
    update: dict[str, Any] = {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    for bucket in ("risks", "key_facts"):
        if extracted[bucket]:
            update[bucket] = extracted[bucket]

    new_nodes: dict[str, Any] = {}
    origin = _node_origin(state, None)
    for assumption in extracted["assumptions"]:
        node = create_node(
            kind="assumption", statement=assumption["statement"], origin=origin, status="needs_confirmation"
        )
        new_nodes[node["id"]] = node
    for question in extracted["open_questions"]:
        node = create_node(kind="open_question", statement=question["question"], origin=origin, status="proposed")
        new_nodes[node["id"]] = node
    if new_nodes:
        update["decision_nodes"] = {**(state.get("decision_nodes") or {}), **new_nodes}
    return Command(update=update)


@tool
async def note(
    content: Annotated[
        str,
        "The note content. Prefix tagged lines (ASSUMPTION: / RISK: / OPEN_QUESTION: / KEY_FACT:) "
        "to record structured items.",
    ],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Silent scratchpad note — no interrupt, no user approval needed.

    Use it to critique (point out weaknesses, risky assumptions, contradictions) or broaden the
    perspective (surface directions/options not yet considered) before asking or drafting. Not
    shown to the user — use respond to surface note content to the user.
    """
    return await _write_note_impl(content, state, tool_call_id, "note")


@tool
async def critique_note(
    content: Annotated[
        str, "The critique. Prefix tagged lines (ASSUMPTION: / RISK: / OPEN_QUESTION:) to record structured items."
    ],  # noqa: E501
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Deprecated alias of `note`; kept so ToolNode resuming an old checkpoint doesn't error."""
    return await _write_note_impl(content, state, tool_call_id, "critique_note")


@tool
async def explore_note(
    content: Annotated[
        str, "The exploration. Prefix tagged lines (ASSUMPTION: / RISK: / OPEN_QUESTION:) to record structured items."
    ],  # noqa: E501
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Deprecated alias of `note`; kept so ToolNode resuming an old checkpoint doesn't error."""
    return await _write_note_impl(content, state, tool_call_id, "explore_note")
