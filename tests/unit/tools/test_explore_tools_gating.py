"""Note scratchpad tool + its dynamic gating.

Covers the `note` tool and the note-specific branches of `get_available_tools(state)`:
the note tool is not capped by a step-limit gate, a `respond` turn resets any note streak, the
streak never blocks `run_critique`, and `_write_note_impl` appends a ToolMessage.

Menu composition for finalize/readiness/respond/read is covered exhaustively in
test_menu_gating_matrix.py; this file keeps only the note-tool behavior unique to it.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graphs.agent_tools import get_available_tools
from app.graphs.decision_graph import create_node


def _names(tools):
    return [t.name for t in tools]


def _draft_state(statement: str = "Mot content draft") -> dict:
    nodes = {
        "N1": create_node(
            kind="objective",
            statement=statement,
            origin={"source": "test"},
            status="confirmed",
        )
    }
    return {"artifact_type": "brd", "decision_nodes": nodes}


def _note_turn(call_id: str):
    """An AIMessage choosing a note tool — one note turn in the loop's history."""
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": "note", "args": {"content": "x"}}])


def _respond_turn(call_id: str):
    return AIMessage(
        content="", tool_calls=[{"id": call_id, "name": "respond", "args": {"message": "x", "mode": "critique"}}]
    )


def test_note_tool_remains_available_after_many_notes():
    messages = [_note_turn(f"c{i}") for i in range(5)]
    names = _names(get_available_tools({"messages": messages}))

    assert "note" in names


def test_respond_resets_note_step_limit():
    messages = [_note_turn(f"c{i}") for i in range(5)] + [_respond_turn("r1")]
    names = _names(get_available_tools({"messages": messages}))
    assert "note" in names


def test_note_step_limit_does_not_block_run_critique():
    messages = [_note_turn(f"c{i}") for i in range(5)]
    names = _names(get_available_tools({**_draft_state(), "messages": messages, "user_confirmed": True}))
    assert "run_critique" in names


@pytest.mark.asyncio
async def test_write_note_appends_to_messages():
    from app.graphs.agent_tools import _write_note_impl

    command = await _write_note_impl("Assumption X may be wrong because...", {}, "call_1", "explore_note")

    appended = command.update["messages"]
    assert len(appended) == 1
    msg = appended[0]
    assert isinstance(msg, ToolMessage)
    assert "Assumption X" in msg.content
    assert msg.tool_call_id == "call_1"
