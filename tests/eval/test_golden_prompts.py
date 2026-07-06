"""Semantic prompt regression for behavior scenarios.

The previous version compared full rendered prompt transcripts byte-for-byte.
That made routine prompt and context refactors expensive to review. This test now
guards the stable contract instead: critical system-policy fragments are present,
the offered tool menu contains every scripted tool the analyst dispatches, and
the checkpoint preserves those dispatched tool names.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage

from app.instructions import load_instructions
from tests.eval.behavior_scenarios import BEHAVIOR_SCENARIOS
from tests.integration.scenarios.driver import ScenarioDriver

pytestmark = [pytest.mark.eval, pytest.mark.golden]

_REQUIRED_SYSTEM_FRAGMENTS = (
    "The harness owns the schema and state.",
    "A human holds final authority.",
    "Record content by creating nodes instead of hand-writing the document body.",
    "Pick 1",
    "Human-approval gates are non-negotiable.",
)


async def _dispatched_tool_names(scenario_env, session_id) -> list[list[str]]:
    """Tool names per analyst AIMessage, in order, from the checkpoint messages."""
    raw = await scenario_env.get_checkpoint_field(session_id, "messages") or []
    turns: list[list[str]] = []
    for msg in raw:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            turns.append([tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None) for tc in tool_calls])
    return turns


def _tool_names_from_ai_message(message: object) -> list[str]:
    if not isinstance(message, AIMessage):
        return []
    return [tc.get("name") for tc in (message.tool_calls or []) if tc.get("name")]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_factory", BEHAVIOR_SCENARIOS, ids=lambda f: f.__name__)
async def test_golden_prompts(scenario_factory, client, scenario_env, scenario_project):
    load_instructions()
    headers, project = scenario_project
    scenario = scenario_factory()
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)
    await driver.run()

    tool_select_calls = [call for call in scenario.llm.calls if call["route"] == "tool_select"]
    assert tool_select_calls, f"{scenario.name}: no analyst tool-selection calls captured"

    expected_dispatched = [_tool_names_from_ai_message(call["result"]) for call in tool_select_calls]
    actual_dispatched = await _dispatched_tool_names(scenario_env, driver.session_id)
    assert actual_dispatched == expected_dispatched

    for index, call in enumerate(tool_select_calls):
        system = call.get("system") or ""
        offered_tools = set(call.get("tool_names") or [])
        selected_tools = set(expected_dispatched[index])

        for fragment in _REQUIRED_SYSTEM_FRAGMENTS:
            assert fragment in system, f"{scenario.name} turn {index}: missing system fragment {fragment!r}"
        assert selected_tools <= offered_tools, (
            f"{scenario.name} turn {index}: dispatched tool not present in offered menu; "
            f"selected={sorted(selected_tools)} offered={sorted(offered_tools)}"
        )
