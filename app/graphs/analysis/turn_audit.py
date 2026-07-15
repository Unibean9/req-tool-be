"""AgentRun bookkeeping, audit hashing, token estimation, and turn fingerprints."""

import hashlib
import uuid
from typing import Any

from sqlalchemy import select

from app.graphs.analysis.tool_gating import (
    _COERCED_ASK_FALLBACK_BY_LOCALE,
    _RESPOND_FALLBACK_BY_LOCALE,
    _response_message_incomplete,
)
from app.models.agent import AgentMessage, AgentMessageRole, AgentRun


async def _direct_response_already_saved(db, session_id: uuid.UUID, content: str) -> bool:
    """Prevent duplicate writes when a node replays after DB commit but before checkpointing.

    Compare only the latest non-queued message. A message queued before the graph checkpoints must
    not bypass the guard; a new user turn may legitimately save the same content again.
    """
    latest = (
        await db.execute(
            select(AgentMessage)
            .where(
                AgentMessage.session_id == session_id,
                AgentMessage.payload["queued"].as_boolean().is_not(True),
            )
            .order_by(AgentMessage.created_at.desc(), AgentMessage.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return bool(
        latest
        and latest.role == AgentMessageRole.AGENT
        and latest.content == content
        and isinstance(latest.payload, dict)
        and latest.payload.get("kind") == "response"
    )

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


def _calibrate_breakdown(raw: dict[str, int], real_total: int) -> dict[str, int]:
    """Scale the raw character-count estimate (`raw`) so its total matches `real_total` (the real
    token count from the provider).

    Keeps the relative ratio between components (system/history/tools/draft) of the character
    estimate, only rescaling the total to match the real figure. When there is no trustworthy real
    figure (`real_total <= 0`) or the raw estimate is empty (`sum(raw.values()) <= 0`), returns
    `raw` unchanged — this is the plain character-estimate fallback.
    """
    raw_total = sum(raw.values())
    if real_total <= 0 or raw_total <= 0:
        return raw
    calibrated = {key: round(value * real_total / raw_total) for key, value in raw.items()}
    # Rounding remainder (can be negative or positive) is added to the largest component so the
    # total matches real_total exactly, avoiding cumulative drift across many turns.
    remainder = real_total - sum(calibrated.values())
    if remainder:
        largest_key = max(raw, key=lambda key: raw[key])
        calibrated[largest_key] += remainder
    return calibrated


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
    """Additive per-component estimate, alongside (never replacing) whatever keys the client's
    usage dict already carries."""
    token_usage = dict(usage) if isinstance(usage, dict) else usage
    if isinstance(token_usage, dict):
        raw_breakdown = _estimate_token_breakdown(
            system_prompt=system_prompt,
            messages=messages,
            tool_schemas=tool_schemas,
            draft_body=draft_body,
        )
        real_input = token_usage.get("input")
        # system/history/tools/draft are all prompt-side content, so calibrate against the
        # provider's real input count, not "total" (which is input + output/completion tokens —
        # calibrating to it would bleed completion tokens into every prompt component). With no
        # real input figure (None/0/missing), keep the raw character estimate.
        if isinstance(real_input, int | float) and real_input > 0:
            token_usage["by_component"] = _calibrate_breakdown(raw_breakdown, int(real_input))
        else:
            token_usage["by_component"] = raw_breakdown
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
    direct_response: str,
    locale: str,
    turn_id: uuid.UUID | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Persist the AgentRun and coerce the gated tools into dispatchable calls.

    The dispatch ids embed the run id, so coercion happens inside the same DB scope that flushes
    the run — exactly the block analyze_node used to hold inline.

    `turn_id` is attribution only (joins this run back to its logical turn for on-call
    correlation), never identity — omitting it (None) is always a valid, supported call.
    """
    async with session_factory() as db:
        run = AgentRun(
            session_id=session_id,
            turn_id=turn_id,
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
                dispatched_tool_call = {"id": f"{run_id}-{i}", "name": tool, "args": args}
                provider_metadata = item.get("provider_metadata")
                if isinstance(provider_metadata, dict):
                    dispatched_tool_call["provider_metadata"] = provider_metadata
                dispatched_tool_calls.append(dispatched_tool_call)
        analysis_result = {
            **analysis_result_base,
            "tools": [_audit_tool_call(item) for item in dispatched_tools],
            "dispatched_tool_calls": [_audit_tool_call(item) for item in dispatched_tool_calls],
            "response_mode": "direct" if direct_response else ("tool" if dispatched_tools else "none"),
        }
        run.analysis_result = analysis_result
        if direct_response and not await _direct_response_already_saved(db, session_id, direct_response):
            db.add(
                AgentMessage(
                    session_id=session_id,
                    role=AgentMessageRole.AGENT,
                    content=direct_response,
                    payload={"kind": "response", "locale": locale, "run_id": run_id},
                )
            )
        await db.commit()
    return run_id, analysis_result, dispatched_tools, dispatched_tool_calls
