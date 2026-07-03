"""AgentRun bookkeeping, audit hashing, token estimation, and turn fingerprints."""

import hashlib
import uuid
from typing import Any

from langchain_core.messages import AIMessage

from app.graphs.analysis.tool_gating import (
    _COERCED_ASK_FALLBACK_BY_LOCALE,
    _RESPOND_FALLBACK_BY_LOCALE,
    _ai_text_content,
    _plain_response_tool,
    _response_message_incomplete,
)
from app.models.agent import AgentRun

# Number of consecutive identical (name + args) tool-call fingerprints that trigger route_node's
# early exit. 3 (not 1 or 2) is conservative enough to tolerate a legitimate one-off repeat (e.g. the
# model re-issuing the same idempotent call after a transient tool error) while still catching a model
# stuck looping well before the 30-turn max_agent_turns ceiling.
_REPEATED_TOOL_CALL_EXIT_THRESHOLD = 3
# Only the last N fingerprints are ever needed to test the threshold; bounding the list keeps the
# checkpointed WorkflowState field small regardless of session length.
_RECENT_TOOL_CALLS_MAXLEN = _REPEATED_TOOL_CALL_EXIT_THRESHOLD

_AUDIT_TEXT_ARG_KEYS = frozenset(
    {
        "body",
        "message",
        "content",
        "summary",
        "statement",
        "title",
        "question",
        "change_description",
        "seed",
    }
)

# P10: the LLM API returns only aggregate input/output/total token counts, never a per-component
# split, so per-component figures here are a proxy, not an exact count. We use a uniform
# chars-to-tokens ratio (4 chars/token — the commonly cited average for English/mixed-language text;
# no tokenizer is exposed by our LLM client interface, and adding a tiktoken dependency purely for an
# estimate would be disproportionate). Applied identically to every component, so the four figures are
# comparable to each other even though none of them is individually precise.
_CHARS_PER_TOKEN_ESTIMATE = 4


def _tool_call_fingerprint(name: str, args: dict[str, Any]) -> str:
    """Fingerprint a dispatched tool call by name + sorted-args, not the full payload (P9)."""
    return f"{name}:{sorted(args.items())!r}"


def _has_repeated_tool_calls(recent_tool_calls: list[str]) -> bool:
    """True when the last N fingerprints are all identical (P9 early-exit trigger)."""
    if len(recent_tool_calls) < _REPEATED_TOOL_CALL_EXIT_THRESHOLD:
        return False
    tail = recent_tool_calls[-_REPEATED_TOOL_CALL_EXIT_THRESHOLD:]
    return len(set(tail)) == 1


def _audit_text_value(value: str) -> dict[str, Any]:
    encoded = value.encode("utf-8")
    return {
        "omitted": True,
        "length": len(value),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _audit_arg_value(key: str, value: Any) -> Any:
    if isinstance(value, str) and key in _AUDIT_TEXT_ARG_KEYS:
        return _audit_text_value(value)
    if isinstance(value, dict):
        return {
            nested_key: _audit_arg_value(str(nested_key), nested_value)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_audit_arg_value(key, item) for item in value]
    return value


def _audit_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = dict(tool_call.get("args") or {})
    audited = {
        "name": tool_call.get("name") or "",
        "args": {key: _audit_arg_value(str(key), value) for key, value in args.items()},
    }
    if tool_call.get("id") is not None:
        audited["id"] = tool_call.get("id")
    return audited


def _estimate_tokens(text: str) -> int:
    return max(0, len(text)) // _CHARS_PER_TOKEN_ESTIMATE


def _estimate_token_breakdown(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    draft_body: str | None,
) -> dict[str, int]:
    """Additive per-component token estimate (P10) — system/history/tools/draft, char-proxy based."""
    history_text = "\n".join(str(message.get("content") or "") for message in messages)
    tools_text = "\n".join(str(schema) for schema in tool_schemas)
    return {
        "system": _estimate_tokens(system_prompt),
        "history": _estimate_tokens(history_text),
        "tools": _estimate_tokens(tools_text),
        "draft": _estimate_tokens(draft_body or ""),
    }


def build_analysis_result_base(
    *,
    gated_tools: list[dict[str, Any]],
    model_tool_calls: list[dict[str, Any]],
    dropped_tools: list[str],
    available_tools: list[Any],
    locale: str,
    coverage_complete: Any,
    session_phase: str | None = None,
) -> dict[str, Any]:
    """The pre-dispatch AgentRun analysis_result — audited copies of the model's selection."""
    return {
        "tools": [_audit_tool_call(item) for item in gated_tools],
        "model_tool_calls": [_audit_tool_call(item) for item in model_tool_calls],
        "raw_model_tool_calls": [_audit_tool_call(item) for item in model_tool_calls],
        "dropped_tool_calls": dropped_tools,
        "available_tools": [tool.name for tool in available_tools],
        "locale": locale,
        "coverage_complete": coverage_complete,
        # The phase whose profile shaped this turn's prompt, so the eval can group the
        # per-component token estimate (token_usage.by_component) by phase.
        "session_phase": session_phase,
    }


def annotate_token_usage(
    usage: Any,
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    draft_body: str | None,
) -> Any:
    """P10: additive per-component estimate, alongside (never replacing) whatever keys the client's
    usage dict already carries."""
    token_usage = dict(usage) if isinstance(usage, dict) else usage
    if isinstance(token_usage, dict):
        token_usage["by_component"] = _estimate_token_breakdown(
            system_prompt=system_prompt,
            messages=messages,
            tool_schemas=tool_schemas,
            draft_body=draft_body,
        )
    return token_usage


def append_turn_fingerprint(
    recent_tool_calls: list[str] | None, dispatched_tools: list[dict[str, Any]]
) -> list[str]:
    """P9: append one fingerprint per turn (not per dispatched call) summarizing this turn's whole
    dispatched-tool batch, so route_node's threshold means "N consecutive turns", not "N dispatched
    calls" — a turn that dispatches several identical tool calls at once must not be mistaken for a
    multi-turn stuck loop.
    """
    recent = list(recent_tool_calls or [])
    if dispatched_tools:
        turn_fingerprint = "|".join(
            sorted(_tool_call_fingerprint(item["name"], item["args"]) for item in dispatched_tools)
        )
        recent.append(turn_fingerprint)
    return recent[-_RECENT_TOOL_CALLS_MAXLEN:]


async def record_run_and_dispatch(
    *,
    session_factory,
    session_id: uuid.UUID,
    analysis_result_base: dict[str, Any],
    token_usage: Any,
    latency_ms: int,
    gated_tools: list[dict[str, Any]],
    ai_message: AIMessage,
    locale: str,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Persist the AgentRun and coerce the gated tools into dispatchable calls.

    The dispatch ids embed the run id, so coercion happens inside the same DB scope that flushes
    the run — exactly the block analyze_node used to hold inline.
    """
    async with session_factory() as db:
        run = AgentRun(
            session_id=session_id,
            analysis_result=analysis_result_base,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )
        db.add(run)
        await db.flush()
        run_id = str(run.id)
        dispatched_tool_calls: list[dict[str, Any]] = []
        dispatched_tools: list[dict[str, Any]] = []
        if gated_tools:
            for i, item in enumerate(gated_tools):
                tool = item.get("name") or ""
                args = dict(item.get("args") or {})
                # Per-tool post-processing (coercions that must happen at dispatch time).
                if tool == "ask_user" and not str(args.get("message") or "").strip():
                    # Prefer the gate-set message (names the gated tool) over the generic fallback.
                    gate_msg = str(analysis_result_base.get("message") or "")
                    args["message"] = gate_msg.strip() or _COERCED_ASK_FALLBACK_BY_LOCALE.get(
                        locale, _COERCED_ASK_FALLBACK_BY_LOCALE["en"]
                    )
                if tool == "respond":
                    if _response_message_incomplete(args.get("message")):
                        args["message"] = _RESPOND_FALLBACK_BY_LOCALE.get(locale, _RESPOND_FALLBACK_BY_LOCALE["en"])
                    args["mode"] = args.get("mode") or "critique"
                dispatched_tools.append({"name": tool, "args": args})
                dispatched_tool_calls.append({"id": f"{run_id}-{i}", "name": tool, "args": args})
        if not dispatched_tool_calls and _ai_text_content(ai_message):
            fallback_tool = _plain_response_tool(ai_message, locale)
            dispatched_tools.append(fallback_tool)
            dispatched_tool_calls.append({"id": f"{run_id}-fallback", **fallback_tool})
        analysis_result = {
            **analysis_result_base,
            "tools": [_audit_tool_call(item) for item in dispatched_tools],
            "dispatched_tool_calls": [_audit_tool_call(item) for item in dispatched_tool_calls],
        }
        run.analysis_result = analysis_result
        await db.commit()
    return run_id, analysis_result, dispatched_tools, dispatched_tool_calls
