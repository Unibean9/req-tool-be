"""Phase 2 — ScriptedLLM tool-call harness extension.

These guard R2: before this extension ScriptedLLM silently ignored the `tools=`
kwarg and any tool-call scenario fell through to the `_DONE_TURN` fallback,
producing tests that looked green but never exercised a tool call.

Lives under tests/ (not tests/scenarios/) on purpose: it needs no DB, so it must
not inherit the scenarios session-scoped sqlite fixture.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tests.scenarios.scripted_llm import (
    TOOL_CALL_SCHEMA,
    ScriptedLLM,
    tool_call,
)


@pytest.mark.asyncio
async def test_scripted_llm_returns_ai_message_with_tool_calls():
    """T4 — a tool-call scenario returns an AIMessage in the exact shape ToolNode dispatches."""
    llm = ScriptedLLM(tool_calls=[tool_call("ping", {"x": "hi"}, call_id="c1")])

    result, _usage = await llm.generate(
        messages=[],
        response_format=TOOL_CALL_SCHEMA,
        tools=[{"name": "ping"}],
    )

    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) > 0
    tc = result.tool_calls[0]
    assert all(k in tc for k in ("id", "name", "args"))

    # The shape must survive a real ToolNode dispatch. A minimal compiled graph injects the
    # Runtime that ToolNode needs, so we exercise the real dispatch path without internal hooks.
    @tool
    def ping(x: str) -> str:
        """echo back x"""
        return x

    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([ping]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    dispatch = builder.compile()

    out = dispatch.invoke({"messages": [result]})
    assert out["messages"][-1].content == "hi"
