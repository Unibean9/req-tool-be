"""Golden-transcript regression: rendered prompts + tool menus must stay byte-identical.

Captures, per analyst turn of each behavior scenario (stub mode): the rendered system prompt, the
last user-side prompt message, the tool-schema names offered, and the dispatched tool names from
the checkpoint. Volatile tokens (UUIDs, hex hashes) are normalized to stable placeholders.

Refactors that must be behavior-neutral (plan 260702 Phase 1) are gated on this comparison.
Regenerate intentionally with UPDATE_GOLDENS=1 after a reviewed behavior change.
"""

import json
import os
import re
import uuid
from pathlib import Path

import pytest

from app.instructions import load_instructions
from tests.eval.behavior_scenarios import BEHAVIOR_SCENARIOS
from tests.integration.scenarios.driver import ScenarioDriver

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_HEX_RE = re.compile(r"\b[0-9a-f]{32}\b|\b[0-9a-f]{16}\b", re.IGNORECASE)


def _normalize(text, seen: dict[str, str]) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str):
        # Message content may be a list of content blocks — serialize before normalizing.
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)

    def _sub_uuid(match: re.Match) -> str:
        key = match.group(0).lower()
        if key not in seen:
            seen[key] = f"<uuid-{len(seen) + 1}>"
        return seen[key]

    return _HEX_RE.sub("<hex>", _UUID_RE.sub(_sub_uuid, text))


async def _dispatched_tool_names(scenario_env, session_id) -> list[list[str]]:
    """Tool names per analyst AIMessage, in order, from the checkpoint messages."""
    raw = await scenario_env.get_checkpoint_field(session_id, "messages") or []
    turns: list[list[str]] = []
    for msg in raw:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            turns.append([tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None) for tc in tool_calls])
    return turns


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_factory", BEHAVIOR_SCENARIOS, ids=lambda f: f.__name__)
async def test_golden_prompts(scenario_factory, client, scenario_env, scenario_project):
    load_instructions()
    headers, project = scenario_project
    scenario = scenario_factory()
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)
    await driver.run()

    seen: dict[str, str] = {}
    turns = [
        {
            "system": _normalize(call.get("system"), seen),
            "prompt": _normalize(call.get("last_message"), seen),
            "tool_names": call.get("tool_names"),
        }
        for call in scenario.llm.calls
        if call["route"] == "tool_select"
    ]
    golden = {
        "scenario": scenario.name,
        "turns": turns,
        "dispatched": await _dispatched_tool_names(scenario_env, driver.session_id),
    }

    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = _GOLDEN_DIR / f"{scenario.name}.json"
    payload = json.dumps(golden, ensure_ascii=False, indent=2, sort_keys=True)
    if os.environ.get("UPDATE_GOLDENS") or not path.exists():
        path.write_text(payload, encoding="utf-8")
        return

    expected = path.read_text(encoding="utf-8")
    assert payload == expected, (
        f"Golden prompt transcript changed for {scenario.name}. If intentional, regenerate with "
        "UPDATE_GOLDENS=1 and review the diff."
    )
