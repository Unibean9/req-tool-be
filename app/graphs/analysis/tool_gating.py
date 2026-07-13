"""Tool-menu schemas, the post-LLM solo-invariant gate, and dispatch-time arg coercion.

`agent_tools` is imported as a module reference (`from app.graphs import agent_tools`,
attributes resolved lazily at call time as `agent_tools.<name>`) rather than importing
specific names out of it, for the same reason `gating/menu_rules.py` does the same for
`agent_tools`: `agent_tools.py` imports `app.graphs.gating.menu_rules`, which imports
`app.graphs.gating.dispatch_rules`, which imports this module — a name import here
(`from app.graphs.agent_tools import current_session_phase`) would fail whenever this
module's import is triggered mid-way through `agent_tools.py`'s own load.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from app.graphs import agent_tools
from app.graphs.gating import Mode, check, check_batch, menu_rules
from app.graphs.lifecycle_context import lifecycle_blocked_tool_names
from app.graphs.session_phase import phase_allows
from app.graphs.state import WorkflowState
from app.graphs.tool_metadata import interrupt_bearing_tools, side_effect_free_note_tools

# Injected tool params are runtime wiring (LangGraph fills them), never LLM-visible args — strip
# them from the schema passed to the provider so the model only sees real arguments.
_INJECTED_TOOL_PARAMS = frozenset({"state", "config", "tool_call_id"})


def required_args(tool: BaseTool) -> list[str]:
    """Required LLM-visible args derived from the tool's schema (injected params stripped)."""
    raw = tool.args_schema.model_json_schema() if tool.args_schema else {}
    return [r for r in (raw.get("required") or []) if r not in _INJECTED_TOOL_PARAMS]


_COERCED_ASK_FALLBACK_BY_LOCALE = {
    "vi": (
        "Mình cần làm rõ thêm một ý trước khi có thể viết phần này chắc hơn. "
        "Bạn có thể chia sẻ thêm thông tin quan trọng nhất còn thiếu không?"
    ),
    "en": (
        "I need to clarify one more point before I can write this section with confidence. "
        "Can you share the most important missing context?"
    ),
}

_RESPOND_FALLBACK_BY_LOCALE = {
    "vi": (
        "Dựa trên thông tin hiện có, mình cần phân tích thêm trước khi kết luận. "
        "Bạn bổ sung thêm bối cảnh hoặc xác nhận các điểm chính để mình tiếp tục nhé?"
    ),
    "en": (
        "Based on the current information, I need more analysis before concluding. "
        "Please add context or confirm the key points so I can continue."
    ),
}

# Tools that call interrupt() — they must always run solo (no composite dispatch).
# DB-writing tools (write_draft, finalize) are also in this set: they interrupt and must not
# be paired with another tool in the same turn to preserve idempotency invariants.
_INTERRUPT_BEARING_TOOLS: frozenset[str] = interrupt_bearing_tools()

# Silent scratchpad notes: no interrupt, no DB write, pure state append (assumptions/risks/
# open_questions/key_facts). They may ride along with an interrupt-bearing tool because the ToolNode
# discards their partial update when the interrupt fires and re-applies it exactly once on resume —
# so the model can record what it learned in the SAME turn it asks a question, instead of having the
# note dropped by solo enforcement (the only key_facts populator, which starved the anti-re-ask block).
_SIDE_EFFECT_FREE_NOTE_TOOLS: frozenset[str] = side_effect_free_note_tools()


def _strip_injected_params(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove injected params (state, config, tool_call_id) from a tool's JSON-Schema properties."""
    props = {k: v for k, v in (schema.get("properties") or {}).items() if k not in _INJECTED_TOOL_PARAMS}
    required = [r for r in (schema.get("required") or []) if r not in _INJECTED_TOOL_PARAMS]
    return {**schema, "properties": props, "required": required}


def gate_model_selection(
    state: WorkflowState, ai_message: AIMessage
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any], list[str]]:
    """Post-LLM gate: session-phase menu + the solo invariant; availability is otherwise enforced
    by the state-driven tool surface. Returns (model_tool_calls, gated_tools, dropped_tools,
    next_feedback, out_of_phase_tools)."""
    model_tool_calls = _model_tool_calls(ai_message)
    raw_tools = [_tool_call_for_gate(tc) for tc in model_tool_calls]
    phase = agent_tools.current_session_phase(state)
    out_of_phase_tools = [
        str(item.get("name") or "") for item in raw_tools if not phase_allows(phase, str(item.get("name") or ""))
    ]
    gated_tools = _gate_selected_tools(state, raw_tools)
    dropped_tools = _dropped_tool_names(raw_tools, gated_tools)
    # One-shot: clear the previous turn's notice (already rendered into this turn's prompt) and stage
    # this turn's drops for the next prompt — so the model sees its dropped tools exactly once.
    next_feedback = dict(state.get("feedback_summary") or {})
    next_feedback.pop("dropped_tools", None)
    next_feedback.pop("out_of_phase_tools", None)
    next_feedback.pop("lifecycle_blocked_tools", None)
    if out_of_phase_tools:
        next_feedback["out_of_phase_tools"] = {"phase": phase, "dropped": out_of_phase_tools}
    lifecycle_blocked = [
        item for item in lifecycle_blocked_tool_names(state, raw_tools) if item["name"] not in out_of_phase_tools
    ]
    lifecycle_blocked_names = {item["name"] for item in lifecycle_blocked}
    if lifecycle_blocked:
        next_feedback["lifecycle_blocked_tools"] = lifecycle_blocked
    solo_dropped = [
        name for name in dropped_tools if name not in out_of_phase_tools and name not in lifecycle_blocked_names
    ]
    if solo_dropped:
        next_feedback["dropped_tools"] = solo_dropped
    return model_tool_calls, gated_tools, dropped_tools, next_feedback, out_of_phase_tools


def _build_tool_schemas(tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Convert state-valid tools into provider-agnostic schemas for generate(tools=...)."""
    schemas: list[dict[str, Any]] = []
    for t in tools:
        raw = t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}}
        params = _strip_injected_params(raw)
        schemas.append({"name": t.name, "description": t.description or "", "parameters": params})
    return schemas


def _model_tool_calls(ai_message: AIMessage) -> list[dict[str, Any]]:
    provider_tool_calls = getattr(ai_message, "additional_kwargs", {}).get("provider_tool_calls") or {}
    result: list[dict[str, Any]] = []
    for index, tc in enumerate(getattr(ai_message, "tool_calls", None) or []):
        tool_call = {
            "id": tc.get("id"),
            "name": tc.get("name") or "",
            "args": dict(tc.get("args") or {}),
        }
        key = str(tc.get("id") or f"__index_{index}")
        provider_metadata = provider_tool_calls.get(key)
        if isinstance(provider_metadata, dict):
            tool_call["provider_metadata"] = provider_metadata
        result.append(tool_call)
    return result


def _tool_call_for_gate(tool_call: dict[str, Any]) -> dict[str, Any]:
    result = {"name": tool_call.get("name") or "", "args": dict(tool_call.get("args") or {})}
    if tool_call.get("id") is not None:
        result["id"] = tool_call.get("id")
    provider_metadata = tool_call.get("provider_metadata")
    if isinstance(provider_metadata, dict):
        result["provider_metadata"] = provider_metadata
    return result


def _ai_text_content(ai_message: AIMessage) -> str:
    content = getattr(ai_message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _looks_like_question(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    if "?" in lowered or "？" in lowered:
        return True
    starters = (
        "bạn ",
        "anh/chị ",
        "vui lòng ",
        "hãy ",
        "can you ",
        "could you ",
        "what ",
        "which ",
        "how ",
        "do you ",
    )
    return any(lowered.startswith(prefix) for prefix in starters)


def _plain_response_tool(ai_message: AIMessage, locale: str) -> dict[str, Any]:
    content = _ai_text_content(ai_message)
    if _looks_like_question(content):
        return {"name": "ask_user", "args": {"message": content}}
    message = content or _RESPOND_FALLBACK_BY_LOCALE.get(locale, _RESPOND_FALLBACK_BY_LOCALE["en"])
    return {"name": "respond", "args": {"message": message, "mode": "critique"}}


def _response_message_incomplete(message: Any) -> bool:
    text = str(message or "").strip()
    return not text or text.endswith(":")


def _log_tool_error(code: str, tool_name: str, message: str) -> None:
    """Emit a tool-control error in a grep-friendly format for eval/logs."""
    import logging

    logging.getLogger(__name__).info(
        "tool-error code=%s tool=%s message=%s",
        code,
        tool_name,
        message,
    )


def _dropped_tool_names(requested: list[dict], kept: list[dict]) -> list[str]:
    """Names the gate removed from the model's selection this turn.

    Closes the feedback loop: a silently dropped tool gives the model no ground truth to self-correct,
    so it keeps re-pairing the same tools. The diff is a name-multiset subtraction (the gate never
    substitutes a tool, only drops), surfaced next turn via feedback_summary['dropped_tools'].
    """
    from collections import Counter

    kept_counts = Counter(item.get("name") or "" for item in kept)
    dropped: list[str] = []
    for item in requested:
        name = item.get("name") or ""
        if kept_counts.get(name, 0) > 0:
            kept_counts[name] -= 1
        else:
            dropped.append(name)
    return dropped


def _gate_selected_tools(_state: WorkflowState, requested: list[dict]) -> list[dict]:
    """Enforce the ToolNode safety invariants without picking a tool on the model's behalf.

    Two rules: (1) the session-phase + lifecycle menu — a tool the current phase excludes, or that
    the focused artifact's lifecycle state blocks, is dropped (the model gets the reason via feedback
    next turn); (2) solo enforcement for interrupt-bearing tools: keep the first interrupt plus
    side-effect-free notes, drop the rest. Tools still decide unavailable/missing-arg via a
    tool_result error.

    Both rules are evaluated through the Policy layer (`app.graphs.gating`): phase+lifecycle via
    `PhaseLifecycleMenuRule(mode=Mode.DISPATCH)` (registered for `Mode.DISPATCH`), solo enforcement
    via `SoloInvariantBatchRule` (a batch rule). This function still owns the `dropped_out_of_phase_tool`
    logging (the rule deliberately stays silent on phase exclusion — see that rule's docstring) and
    stays silent when the rule denies for a lifecycle reason, since the rule itself already logged
    `log_gate_decision("lifecycle_tool_gate", ...)` in that case.
    """
    menu_rules.ensure_dispatch_rules_registered()

    # Normalize to a stable dispatch shape; the model's chosen tools are never substituted.
    validated = [_tool_call_for_gate(item) for item in requested]

    # Per-phase + per-lifecycle enforcement (defense in depth — the provider only sees in-phase
    # schemas, but replayed/edited selections can still arrive here out of phase or lifecycle-blocked).
    phase = agent_tools.current_session_phase(_state)
    survivors: list[dict] = []
    for item in validated:
        verdict = check({"name": item["name"], "args": item["args"], "phase": phase}, _state, Mode.DISPATCH)
        if verdict.is_allow:
            survivors.append(item)
        elif verdict.reason == "phase_excludes_tool":
            _log_tool_error(
                "dropped_out_of_phase_tool",
                item["name"],
                f"dropped: not available in session phase '{phase}'",
            )
        # else: a lifecycle reason string — the rule already logged
        # `log_gate_decision("lifecycle_tool_gate", ...)`; do not log again here.

    # Solo enforcement: at most one interrupt-bearing tool per turn (two interrupts in a node is
    # unsafe). When one is present, keep it plus any side-effect-free notes (so their structured facts
    # persist this turn) and drop everything else, preserving original order.
    return check_batch(survivors, _state)
