"""Deterministic scripted LLM client for behavior scenarios.

The whole graph (intent_router, analyze, summarize, quality_gate critic +
regenerate) calls a single `llm_client.generate(...)`. This client routes each
call to a scripted/auto response by inspecting `response_format` and `system`,
so a scenario only needs to script the analyst "brain" turns — everything else
gets a sensible default.

Routing keys (matching app/graphs):
- INTENT_SCHEMA   -> has property "intent"            -> intent response
- SUMMARY_SCHEMA  -> property set == {"summary"}      -> summary response
- CRITIC_SCHEMA   -> has property "suggestions"       -> critic response
- ANALYSIS_SCHEMA -> has property "next_action":
    * system == regenerate system  -> regenerate response (improved proposals)
    * otherwise                    -> next scripted analyze "brain" turn

`generate` returns `(result_dict, usage)` like the real clients; usage is None
(AgentRun.token_usage is nullable).
"""

from typing import Any

from langchain_core.messages import AIMessage

from app.graphs.critic import _REGENERATE_SYSTEM


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


# Test-harness-only schema marking a native tool-call turn (Phase 2). When generate sees this as
# response_format it returns an AIMessage(tool_calls=[...]) instead of a JSON dict — the shape the
# LangGraph ToolNode dispatches. The marker property is dunder-namespaced so it can never collide
# with a real artifact-schema field; _route keys on it.
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {"__tool_call__": {"type": "boolean"}},
}


# A terminal analyze turn used when a scenario's brain script is exhausted.
_DONE_TURN: dict[str, Any] = {
    "next_action": "done",
    "confidence": 1.0,
    "gaps": [],
    "message": "",
    "proposals": [],
}


class ScriptedLLM:
    """A scripted, deterministic stand-in for an LLM client.

    Parameters
    ----------
    brain:
        Ordered analyze responses (ANALYSIS_SCHEMA shape). Consumed one per
        `analyze_node` run. When exhausted, a terminal "done" turn is returned.
    intent:
        Response for `intent_router_node` (default: task / vi).
    critic:
        Response for the quality-gate critic (default: passes the gate).
    summary:
        Response for `summarize_node`.
    followup:
        Response for missing-coverage follow-up repair.
    """

    def __init__(
        self,
        *,
        brain: list[dict[str, Any]] | None = None,
        intent: dict[str, Any] | None = None,
        critic: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        followup: dict[str, Any] | None = None,
        judge: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self._brain = list(brain or [])
        self._brain_idx = 0
        # Ordered native tool-call turns, consumed one per "tool_call"-routed generate (Phase 2).
        self._tool_calls = list(tool_calls or [])
        self._tool_call_idx = 0
        self._intent = intent or {"intent": "task", "locale": "vi"}
        self._critic = critic or {
            "scores": {
                "unambiguous": 0.9,
                "verifiable": 0.85,
                "complete": 0.9,
                "consistent": 0.9,
                "traceable": 0.8,
                "feasible": 0.85,
                "invest": None,
                "smart": None,
            },
            "overall": 0.88,
            "rationale": "Các proposal rõ ràng và khả thi.",
            "suggestions": [],
        }
        self._summary = summary or {"summary": ""}
        # Default judge passes (on_topic=True) — see test_m2 note: this is infra wiring, not a
        # real M2 measurement (which needs a real LLM judge outside CI).
        self._judge = judge or {"on_topic": True, "matched_slot": "", "reason": ""}
        self._followup = followup or {
            "message": "Mình cần làm rõ thêm phần còn thiếu trước khi viết artifact. Bạn có thể mô tả thêm điểm quan trọng nhất không?"
        }
        # Audit trail of every routed call — surfaced in the transcript.
        self.calls: list[dict[str, Any]] = []
        # Tools recorded by bind_tools so a test can assert what the loop offered (Phase 4 stub).
        self._bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "ScriptedLLM":
        """Harness stub: record the bound tools and return self (chaining). Not a real LLM bind."""
        self._bound_tools = list(tools)
        return self

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
        # `tools` is accepted but does not drive routing: the scenario's response_format decides
        # whether a turn is a tool call (TOOL_CALL_SCHEMA) or an enum turn. Accepting the kwarg
        # stops a future tool-bound call from silently falling through to the _DONE_TURN fallback (R2).
        route = self._route(response_format, system)
        result = self._respond(route)
        self.calls.append({"route": route, "result": result})
        return result, None

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route(self, response_format: dict[str, Any] | None, system: str | None) -> str:
        props = ((response_format or {}).get("properties")) or {}
        # Tool-call guard: a native tool-call turn is flagged by the dunder marker, detected before
        # the enum branches (Phase 2).
        if "__tool_call__" in props:
            return "tool_call"
        # Judge guard MUST be first: a future schema carrying both on_topic and next_action would
        # otherwise route to "analyze" and the judge branch would never run.
        if "on_topic" in props:
            return "judge"
        if "intent" in props:
            return "intent"
        if "suggestions" in props:
            return "critic"
        if set(props.keys()) == {"summary"}:
            return "summary"
        if set(props.keys()) == {"message"}:
            return "followup"
        if "next_action" in props:
            if system and system.strip() == _REGENERATE_SYSTEM.strip():
                return "regenerate"
            return "analyze"
        # Unknown call shape — fall back to a harmless analyze "done".
        return "analyze"

    def _respond(self, route: str) -> Any:
        if route == "tool_call":
            if self._tool_call_idx < len(self._tool_calls):
                tc = self._tool_calls[self._tool_call_idx]
                self._tool_call_idx += 1
                return AIMessage(content="", tool_calls=[dict(tc)])
            return AIMessage(content="", tool_calls=[])
        if route == "judge":
            return dict(self._judge)
        if route == "intent":
            return dict(self._intent)
        if route == "critic":
            return dict(self._critic)
        if route == "summary":
            return dict(self._summary)
        if route == "followup":
            return dict(self._followup)
        if route == "regenerate":
            # Echo the last brain turn's proposals (already-improved content is
            # the responsibility of the scenario's critic score, not this stub).
            last = self._brain[self._brain_idx - 1] if self._brain_idx > 0 else _DONE_TURN
            return {"next_action": "propose", "confidence": 0.9, "proposals": last.get("proposals", [])}
        # route == "analyze"
        if self._brain_idx < len(self._brain):
            turn = self._brain[self._brain_idx]
            self._brain_idx += 1
            return dict(turn)
        return dict(_DONE_TURN)


# ---------------------------------------------------------------------------
# Brain-turn builders — keep scenario definitions terse and readable.
# ---------------------------------------------------------------------------

def ask(
    message: str,
    *,
    acknowledgment: str = "",
    gaps: list[str] | None = None,
    slot_assessment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """An analyze turn that asks the user one question."""
    turn: dict[str, Any] = {
        "next_action": "ask",
        "confidence": 0.4,
        "gaps": gaps or [],
        "message": message,
    }
    if acknowledgment:
        turn["acknowledgment"] = acknowledgment
        turn["answer_assessment"] = "complete"
    if slot_assessment is not None:
        turn["slot_assessment"] = slot_assessment
    return turn


def propose(*proposals: dict[str, Any], confidence: float = 0.9) -> dict[str, Any]:
    """An analyze turn that proposes artifacts (routes through confirm -> gate)."""
    return {
        "next_action": "propose",
        "confidence": confidence,
        "gaps": [],
        "message": "",
        "proposals": list(proposals),
    }


def artifact(artifact_type: str, title: str, body: str, rationale: str = "") -> dict[str, Any]:
    """A single proposal block."""
    return {"artifact_type": artifact_type, "title": title, "body": body, "rationale": rationale}


def tool_call(name: str, args: dict[str, Any], *, call_id: str = "call_1") -> dict[str, Any]:
    """A scripted native tool-call turn (Phase 2). Shape matches a LangGraph ToolNode tool_call."""
    return {"id": call_id, "name": name, "args": args}
