"""`critique_note`/`explore_note` merged into a single `note` tool on the menu; the two old tools
stay in the full registry (`get_all_analyzer_tools`) and remain executable so `ToolNode` can
re-execute an old tool_call when resuming a checkpoint created before the merge."""

import pytest
from langchain_core.messages import ToolMessage

from app.graphs.agent_tools import (
    _write_note_impl,
    critique_note,
    explore_note,
    get_all_analyzer_tools,
    get_available_tools,
    note,
)


def _names(tools):
    return [t.name for t in tools]


def test_note_is_offered_on_menu():
    assert "note" in _names(get_available_tools({"messages": []}))


def test_deprecated_aliases_are_not_offered_on_menu():
    names = _names(get_available_tools({"messages": []}))
    assert "critique_note" not in names
    assert "explore_note" not in names


def test_deprecated_aliases_remain_in_full_registry_for_tool_node():
    names = _names(get_all_analyzer_tools())
    assert "note" in names
    assert "critique_note" in names
    assert "explore_note" in names


@pytest.mark.asyncio
async def test_note_tool_writes_message_like_old_aliases():
    command = await _write_note_impl("KEY_FACT: DAU target la 10k", {}, "call_1", "note")

    appended = command.update["messages"]
    assert len(appended) == 1
    msg = appended[0]
    assert isinstance(msg, ToolMessage)
    assert "DAU target" in msg.content
    assert command.update["key_facts"]


def _tool_call(name: str, content: str, call_id: str) -> dict:
    return {"name": name, "args": {"content": content, "state": {}}, "id": call_id, "type": "tool_call"}


@pytest.mark.asyncio
async def test_note_tool_direct_invocation_executes():
    command = await note.ainvoke(_tool_call("note", "Exploration angle", "call_1"))
    assert command.update["messages"][0].content == "Exploration angle"


@pytest.mark.asyncio
async def test_deprecated_alias_direct_invocation_still_executes_for_resume():
    critique_result = await critique_note.ainvoke(_tool_call("critique_note", "Old critique note", "call_1"))
    explore_result = await explore_note.ainvoke(_tool_call("explore_note", "Old exploration note", "call_2"))

    assert critique_result.update["messages"][0].content == "Old critique note"
    assert explore_result.update["messages"][0].content == "Old exploration note"
