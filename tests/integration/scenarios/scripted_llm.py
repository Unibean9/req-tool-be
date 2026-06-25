"""Deterministic scripted LLM client for behavior scenarios.

The whole graph (analyze, summarize) calls a single `llm_client.generate(...)`. This client routes
each call to a scripted/auto response by inspecting `response_format`, so a scenario only needs to
script the analyst tool-selection turns — everything else gets a sensible default.

Routing keys (matching app/graphs):
- TOOL_CALL_SCHEMA  -> has property "__tool_call__" -> native AIMessage(tool_calls) (harness self-test)
- tools param set   -> native tool-selection: next scripted tool-selection turn as an AIMessage
- ON_TOPIC_SCHEMA   -> has property "on_topic" -> judge response (M2 harness only)
- TRIAGE_SCHEMA     -> has property "turn_type" -> triage classifier
- SUMMARY_SCHEMA    -> property set == {"summary"} -> summary response

`generate` returns `(result, usage)` like the real clients; usage is None
(AgentRun.token_usage is nullable).
"""

from typing import Any

from langchain_core.messages import AIMessage

# Test-harness-only schema: a binary judgment of whether an agent question maps to a focus
# slot of the artifact_type (M2). Never sent to a production LLM.
ON_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "on_topic": {"type": "boolean"},
        "matched_slot": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["on_topic"],
}


# Test-harness-only schema marking a native tool-call turn. When generate sees this as
# response_format it returns an AIMessage(tool_calls=[...]) — the shape the LangGraph ToolNode
# dispatches. The marker property is dunder-namespaced so it can never collide with a real schema
# field; _route keys on it.
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {"__tool_call__": {"type": "boolean"}},
}


class ScriptedLLM:
    """A scripted, deterministic stand-in for an LLM client.

    Parameters
    ----------
    tool_brain:
        Ordered tool-SELECTION turns, consumed one per `analyze_node` run and converted to an
        AIMessage(tool_calls). When exhausted, a plain AIMessage with no tool_calls is returned and
        the loop ends.
    summary:
        Response for `summarize_node`.
    judge:
        Response for the M2 on-topic harness judge (default: passes).
    tool_calls:
        Ordered native tool-call turns, consumed one per TOOL_CALL_SCHEMA generate (harness self-test).
    """

    def __init__(
        self,
        *,
        summary: dict[str, Any] | None = None,
        judge: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_brain: list[dict[str, Any]] | None = None,
    ) -> None:
        # Ordered native tool-call turns, consumed one per "tool_call"-routed generate.
        self._tool_calls = list(tool_calls or [])
        self._tool_call_idx = 0
        # Ordered tool-SELECTION turns: the analyst returns a dict naming a tool, consumed one per
        # TOOL_SELECTION_SCHEMA generate. analyze_node converts it to an AIMessage.
        self._tool_brain = list(tool_brain or [])
        self._tool_brain_idx = 0
        self._summary = summary or {"summary": ""}
        # Default judge passes (on_topic=True): infra wiring, not a real M2 measurement (which needs
        # a real LLM judge outside CI).
        self._judge = judge or {"on_topic": True, "matched_slot": "", "reason": ""}
        # Audit trail of every routed call — surfaced in the transcript.
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # LLM client protocol
    # ------------------------------------------------------------------

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **_kwargs: Any,
    ) -> tuple[Any, None]:
        route = self._route(response_format, tools=tools)
        result = self._respond(route)
        self.calls.append({"route": route, "result": result})
        return result, None

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route(self, response_format: dict[str, Any] | None, *, tools: list[Any] | None = None) -> str:
        props = ((response_format or {}).get("properties")) or {}
        # __tool_call__ marker wins so the harness self-test (which passes BOTH tools and
        # TOOL_CALL_SCHEMA) still routes to the native tool_call brain.
        if "__tool_call__" in props:
            return "tool_call"
        # Native tool-selection: analyze_node passes the bound tool schemas (response_format=None).
        if tools is not None:
            return "tool_select"
        # Judge guard before the others: the on-topic harness schema.
        if "on_topic" in props:
            return "judge"
        # Triage classifier: scenarios are all requirements work, so default to "work" and skip the
        # conversational branch.
        if "turn_type" in props:
            return "triage"
        if set(props.keys()) == {"summary"}:
            return "summary"
        # Unknown call shape — return an empty dict (harmless).
        return "empty"

    def _respond(self, route: str) -> Any:
        if route == "tool_call":
            if self._tool_call_idx < len(self._tool_calls):
                tc = self._tool_calls[self._tool_call_idx]
                self._tool_call_idx += 1
                return AIMessage(content="", tool_calls=[dict(tc)])
            return AIMessage(content="", tool_calls=[])
        if route == "tool_select":
            if self._tool_brain_idx < len(self._tool_brain):
                turn = self._tool_brain[self._tool_brain_idx]
                self._tool_brain_idx += 1
                return _tool_select_to_ai_message(turn)
            # Exhausted -> terminal: a plain AIMessage with no tool_calls. analyze_node sees the empty
            # tool_calls and ends the turn (route_node -> END).
            return AIMessage(content="", tool_calls=[])
        if route == "triage":
            return {"turn_type": "work", "locale": "vi"}
        if route == "judge":
            return dict(self._judge)
        if route == "summary":
            return dict(self._summary)
        return {}


# ---------------------------------------------------------------------------
# Brain-turn builders — keep scenario definitions terse and readable.
# ---------------------------------------------------------------------------

def _tool_select_to_ai_message(turn: dict[str, Any]) -> AIMessage:
    """Convert a scripted tool-selection turn → AIMessage(tool_calls=[...]).

    Mirrors the native client path: analyze_node now receives an AIMessage directly (not a dict), so
    the scripted turn `{"tools": [{"name": "ask_user", "args": {...}}], "active_mode": ...}` is mapped
    to `AIMessage(tool_calls=[{"id": "scripted:0", "name": ..., "args": ...}])`. The analytic
    `active_mode` is dropped — analyze_node derives it from the picked tool, same as production.
    """
    tool_calls = [
        {"id": f"scripted:{i}", "name": item["name"], "args": dict(item.get("args") or {})}
        for i, item in enumerate(turn.get("tools") or [])
    ]
    return AIMessage(content="", tool_calls=tool_calls)


def tool_call(name: str, args: dict[str, Any], *, call_id: str = "call_1") -> dict[str, Any]:
    """A scripted native tool-call turn. Shape matches a LangGraph ToolNode tool_call."""
    return {"id": call_id, "name": name, "args": args}


def tool_select(tool: str, *, active_mode: str | None = None, **args: Any) -> dict[str, Any]:
    """A scripted tool-SELECTION turn: the analyst names a tool plus its args.

    analyze_node converts this dict into an AIMessage(tool_calls=[...]). `active_mode` is kept as a
    top-level analytic field (eval reads it from analysis_result); remaining kwargs become the
    per-tool args in the new D1 schema: {"tools": [{"name": tool, "args": {...}}]}.
    """
    turn: dict[str, Any] = {"tools": [{"name": tool, "args": dict(args)}]}
    if active_mode is not None:
        turn["active_mode"] = active_mode
    return turn
