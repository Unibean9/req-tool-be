"""Phase 4 — Explore Tools + Dynamic Gating.

Adds the note scratchpad tools (`critique_note`/`explore_note`) and `get_available_tools(state)`,
the state-driven gate that decides which tools the loop may pick each turn. Two gates:
- `finalize` appears only once `working_draft` is non-empty (the single hard-gate).
- the note tools are dropped after N consecutive note turns, forcing the loop to ask_user/write_draft
  so it cannot spam notes forever (S4 — no infinite loop).

The consecutive-note count is derived on-the-fly from message history (N2) — no new state field.

The bind_tools/system-prompt wiring into analyze_node (spec steps 4–5) and the T7 emergent-chain
eval are deferred to Phase 5: the production LLMClient has no bind_tools and analyze_node emits no
native tool_calls until the enum is removed.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graphs.agent_tools import NOTE_STEP_LIMIT, get_available_tools


def _names(tools):
    return [t.name for t in tools]


def _note_turn(call_id: str):
    """An AIMessage choosing a note tool — one note turn in the loop's history."""
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": "critique_note", "args": {"content": "x"}}])


# ---------------------------------------------------------------------------
# T1 — finalize gated when there is no draft body
# ---------------------------------------------------------------------------

def test_finalize_not_available_without_draft_body():
    assert "finalize" not in _names(get_available_tools({"messages": []}))
    # Field present but empty/None is still CLOSED.
    assert "finalize" not in _names(get_available_tools({"messages": [], "working_draft": None}))
    assert "finalize" not in _names(get_available_tools({"messages": [], "working_draft": "  "}))


# ---------------------------------------------------------------------------
# T2 — finalize available once working_draft is non-empty
# ---------------------------------------------------------------------------

def test_finalize_available_after_write_draft():
    tools = get_available_tools({"messages": [], "working_draft": "Một nội dung draft"})
    assert "finalize" in _names(tools)


# ---------------------------------------------------------------------------
# T3 — step-limit computed from message history forces ask/draft after N notes
# ---------------------------------------------------------------------------

def test_step_limit_forces_ask_or_draft_after_N_notes():
    messages = [_note_turn(f"c{i}") for i in range(NOTE_STEP_LIMIT)]
    names = _names(get_available_tools({"messages": messages}))

    # the note tools are dropped, so the loop is forced toward ask_user/write_draft.
    assert "critique_note" not in names and "explore_note" not in names
    assert "ask_user" in names or "write_draft" in names


def test_write_note_available_below_step_limit():
    messages = [_note_turn(f"c{i}") for i in range(NOTE_STEP_LIMIT - 1)]
    names = _names(get_available_tools({"messages": messages}))
    assert "critique_note" in names and "explore_note" in names


# ---------------------------------------------------------------------------
# T5 — respond is always offered and breaks the note streak (a user-facing pause)
# ---------------------------------------------------------------------------

def _respond_turn(call_id: str):
    return AIMessage(
        content="", tool_calls=[{"id": call_id, "name": "respond", "args": {"message": "x", "mode": "critique"}}]
    )


def test_respond_always_available():
    assert "respond" in _names(get_available_tools({"messages": []}))


def test_respond_resets_note_step_limit():
    # NOTE_STEP_LIMIT notes would normally drop the note tools; a respond turn after them is a
    # user-facing pause that resets the streak, so the note tools are offered again.
    messages = [_note_turn(f"c{i}") for i in range(NOTE_STEP_LIMIT)] + [_respond_turn("r1")]
    names = _names(get_available_tools({"messages": messages}))
    assert "critique_note" in names and "explore_note" in names


def test_note_step_limit_does_not_block_run_critique():
    # run_critique is not a NOTE_TOOL, so the note step-limit must not gate it — only the draft
    # presence + critique-rounds cap do. After NOTE_STEP_LIMIT notes it is still offered.
    messages = [_note_turn(f"c{i}") for i in range(NOTE_STEP_LIMIT)]
    names = _names(get_available_tools({"messages": messages, "working_draft": "draft body"}))
    assert "run_critique" in names


# ---------------------------------------------------------------------------
# T4 — write_note appends a ToolMessage to messages (decision 3: no notes field)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_note_appends_to_messages():
    from app.graphs.agent_tools import _write_note_impl

    command = await _write_note_impl("Giả định X có thể sai vì...", {}, "call_1")

    appended = command.update["messages"]
    assert len(appended) == 1
    msg = appended[0]
    assert isinstance(msg, ToolMessage)
    assert "Giả định X" in msg.content
    assert msg.tool_call_id == "call_1"
