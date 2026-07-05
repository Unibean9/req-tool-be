"""System-prompt block stack and per-turn analyst payload assembly."""

import difflib
from typing import Any

from app.config import settings
from app.documents.registry import get_config, output_contract
from app.graphs.agent_tools import get_available_tools
from app.graphs.analysis.context_loader import (
    _build_decision_view_block,
    _decision_view_can_hide_draft,
)
from app.graphs.analysis.turn_audit import _REPEATED_TOOL_CALL_EXIT_THRESHOLD
from app.graphs.lifecycle_context import render_artifact_history, render_situation_report
from app.graphs.policy import ancestor_types
from app.graphs.session_phase import DRAFT, ELICIT, FINALIZE, INTENT, REVIEW
from app.graphs.state import WorkflowState
from app.instructions import get_instruction

# Per-phase prompt profile: the optional blocks each session phase includes.
# A block appears only where it changes THIS phase's action, so no phase renders the full
# god-assembly. Blocks NOT listed in any profile (conversation summary, tool menu, key facts,
# feedback signals, mode hint, language lock, stuck-escalation) are cross-cutting and always
# rendered. Keyed on session_phase, which orchestrator_node writes before analyze_node runs; this
# is per-turn suffix selection only and never touches get_instruction()'s (role, has_draft) cache.
_PHASE_PROFILE_BLOCKS: dict[str, frozenset[str]] = {
    INTENT: frozenset(),
    ELICIT: frozenset({"thinking_mode", "section_coverage", "batching", "section_repair", "type_profile"}),
    DRAFT: frozenset({"artifact_contract", "section_coverage", "decision_view", "section_repair"}),
    REVIEW: frozenset({"artifact_contract", "decision_view", "section_repair", "type_profile"}),
    FINALIZE: frozenset(),
}


def _phase_includes(state: WorkflowState, block: str) -> bool:
    """Whether the current session phase's profile includes this optional prompt block.

    An unset/unknown phase (legacy checkpoint, or a caller that builds a prompt before
    orchestrator_node has assigned one) falls back to including every block so no code path that
    never sets a phase regresses.
    """
    phase = state.get("session_phase")
    if phase not in _PHASE_PROFILE_BLOCKS:
        return True
    return block in _PHASE_PROFILE_BLOCKS[phase]


def _msg_role_content(m) -> tuple[str, str]:
    """Extract role and content from a message object or dict."""
    if isinstance(m, dict):
        return m.get("role", "user"), m.get("content", "")
    role = getattr(m, "type", "user")
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    return role, str(getattr(m, "content", ""))


def _build_analyzer_messages(state: WorkflowState, prompt: str) -> list[dict[str, Any]]:
    """Build the real LLM message thread and place the workspace payload by recency.

    The latest user message must be the last message the model reads; primacy/recency is weighted
    much higher than the middle region (lost-in-the-middle). Therefore the dynamic workspace block
    is inserted immediately before the final user turn, so the user message is the final anchor.
    Only while inside a tool loop (the last message is a tool_result, not a human turn) is the
    workspace appended at the end as before; then recency should belong to tool context.
    """
    messages: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}
    for raw in _analyzer_history_messages(state):
        message = _client_message_from_state(raw, tool_names_by_id)
        if message is not None:
            _append_client_message(messages, message)
    _append_analyzer_prompt(messages, prompt)
    _append_latest_user_emphasis(messages, _latest_human_text(state))
    return messages


def _analyzer_history_messages(state: WorkflowState) -> list[Any]:
    raw_messages = list(state.get("messages") or [])
    if not (state.get("conversation_summary") or "").strip():
        return raw_messages[_bounded_history_start(raw_messages) :]
    return raw_messages[_summary_compaction_start(raw_messages) :]


def _bounded_history_start(messages: list[Any]) -> int:
    """Cap the pre-summary history window to the same recent-turn count `summary_trigger_every`
    is meant to bound, so a long conversation that hasn't triggered a summary yet doesn't resend
    every turn since session start.
    """
    human_indices = [idx for idx, message in enumerate(messages) if _is_human_turn(message)]
    if len(human_indices) <= settings.summary_trigger_every:
        return 0
    return human_indices[-settings.summary_trigger_every]


def _summary_compaction_start(messages: list[Any]) -> int:
    latest_human_index = next(
        (idx for idx in range(len(messages) - 1, -1, -1) if _is_human_turn(messages[idx])),
        None,
    )
    if latest_human_index is None:
        return 0
    if latest_human_index == 0:
        return 0
    previous_tool_call_id = _message_tool_call_id(messages[latest_human_index - 1])
    if previous_tool_call_id:
        return _matching_tool_use_index(messages, previous_tool_call_id, latest_human_index - 1)
    if _is_plain_assistant_turn(messages[latest_human_index - 1]):
        return latest_human_index - 1
    return latest_human_index


def _message_tool_call_id(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("tool_call_id") or "")
    return str(getattr(message, "tool_call_id", None) or "")


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, dict):
        return [dict(call) for call in (message.get("tool_calls") or []) if isinstance(call, dict)]
    return [dict(call) for call in (getattr(message, "tool_calls", None) or []) if isinstance(call, dict)]


def _is_plain_assistant_turn(message: Any) -> bool:
    if _message_tool_calls(message):
        return False
    if isinstance(message, dict):
        return str(message.get("role") or "") in {"assistant", "ai"}
    return getattr(message, "type", "") in {"assistant", "ai"}


def _matching_tool_use_index(messages: list[Any], tool_call_id: str, before_index: int) -> int:
    for idx in range(before_index - 1, -1, -1):
        if any(str(call.get("id") or "") == tool_call_id for call in _message_tool_calls(messages[idx])):
            return idx
    return before_index


def _is_human_turn(message: Any) -> bool:
    """A genuine human turn — a plain user message, not a tool_result/tool output or assistant turn.

    On resume the harness records the human reply as a plain ``{"role": "user", ...}`` dict (or a
    HumanMessage); tool outputs arrive as ToolMessages (role/type ``tool`` or carrying a
    tool_call_id). This distinction is what lets us re-surface the human's words without mistaking a
    mid-loop tool result for user input.
    """
    if isinstance(message, dict):
        if message.get("tool_call_id") or message.get("tool_calls"):
            return False
        return str(message.get("role") or "") in {"user", "human"}
    if getattr(message, "tool_call_id", None) or getattr(message, "tool_calls", None):
        return False
    return getattr(message, "type", "") in {"user", "human"}


def _latest_human_text(state: WorkflowState) -> str:
    """Text of the most recent genuine human turn, for recency re-surfacing (empty if none)."""
    for raw in reversed(state.get("messages") or []):
        if _is_human_turn(raw):
            _role, content = _msg_role_content(raw)
            text = str(content or "").strip()
            if text:
                return text
    return ""


def _append_latest_user_emphasis(messages: list[dict[str, Any]], human_text: str) -> None:
    """Make the human's latest message the FINAL text block the model reads.

    The conversation, not the rules, must own the recency slot: a long static/workspace payload in
    the middle is undervalued (lost-in-the-middle), so the user's actual ask is restated last. Works
    for every case — a tool_result-bearing resume turn buries the reply inside a tool_result block,
    so re-stating it as a trailing text block is the only way to keep it last.
    """
    if not human_text or not messages:
        return
    block = {"type": "text", "text": f"— Latest user turn (prioritize responding to this intent): {human_text}"}
    last = messages[-1]
    if last.get("role") == "user":
        last["content"] = [*_content_blocks(last.get("content")), block]
    else:
        messages.append({"role": "user", "content": [block]})


def _client_message_from_state(message: Any, tool_names_by_id: dict[str, str]) -> dict[str, Any] | None:
    if isinstance(message, dict):
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        tool_call_id = message.get("tool_call_id")
        name = message.get("name")
        tool_calls = message.get("tool_calls") or []
    else:
        raw_role = getattr(message, "type", "user")
        role = {"human": "user", "ai": "assistant", "tool": "tool"}.get(raw_role, str(raw_role))
        content = getattr(message, "content", "")
        tool_call_id = getattr(message, "tool_call_id", None)
        name = getattr(message, "name", None)
        tool_calls = getattr(message, "tool_calls", None) or []

    if role == "tool" or tool_call_id:
        call_id = str(tool_call_id or "")
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "name": str(name or tool_names_by_id.get(call_id) or "tool"),
                    "content": str(content or ""),
                }
            ],
        }

    if role == "assistant" and tool_calls:
        blocks = _text_blocks(content)
        for call in tool_calls:
            call_id = str(call.get("id") or "")
            tool_name = str(call.get("name") or "")
            if call_id and tool_name:
                tool_names_by_id[call_id] = tool_name
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": tool_name,
                    "input": dict(call.get("args") or {}),
                }
            )
        return {"role": "assistant", "content": blocks}

    if role not in {"user", "assistant"}:
        role = "user"
    return {"role": role, "content": str(content or "")}


def _text_blocks(content: Any) -> list[dict[str, Any]]:
    text = str(content or "")
    return [{"type": "text", "text": text}] if text else []


def _append_client_message(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    if messages and messages[-1]["role"] == message["role"] == "user":
        if _has_tool_result(messages[-1].get("content")) or _has_tool_result(message.get("content")):
            if _duplicates_last_tool_result(messages[-1].get("content"), message.get("content")):
                return
            messages[-1]["content"] = [
                *_content_blocks(messages[-1].get("content")),
                *_content_blocks(message.get("content")),
            ]
            return
    messages.append(message)


def _append_analyzer_prompt(messages: list[dict[str, Any]], prompt: str) -> None:
    prompt_block = {"type": "text", "text": prompt}
    if messages and messages[-1]["role"] == "user" and _has_tool_result(messages[-1].get("content")):
        messages[-1]["content"] = [*_content_blocks(messages[-1].get("content")), prompt_block]
        return
    messages.append({"role": "user", "content": prompt})


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return _text_blocks(content)


def _has_tool_result(content: Any) -> bool:
    return any(block.get("type") == "tool_result" for block in _content_blocks(content))


def _duplicates_last_tool_result(existing_content: Any, new_content: Any) -> bool:
    if isinstance(new_content, list):
        return False
    text = str(new_content or "").strip()
    if not text:
        return False
    return any(
        block.get("type") == "tool_result" and str(block.get("content") or "").strip() == text
        for block in _content_blocks(existing_content)
    )


def _build_draft_block(state: WorkflowState, draft_body: str | None) -> str:
    """Persisted-draft block: tell the analyst the body already on record, so it mines the delta."""
    if not draft_body:
        return ""
    return f"\n\nCURRENT DRAFT for type '{state['artifact_type']}':\n{draft_body}"


def _build_draft_delta_block(
    state: WorkflowState, draft_body: str | None, previous_draft_body: str | None
) -> str:
    """Draft block for turns after the first: send only what changed since the last turn.

    Full-body resend every turn was the token-heavy default; the previous turn's body is already
    available for free via state["draft_body"] (see analyze_node), so this diffs against it instead.
    Falls back to the full body when there is no previous body to diff against (first turn / draft
    just created), which is the safety net the diffing-correctness risk mitigation calls for.
    """
    if not previous_draft_body:
        return _build_draft_block(state, draft_body)
    if draft_body == previous_draft_body:
        return ""
    diff_lines = list(
        difflib.unified_diff(
            previous_draft_body.splitlines(),
            (draft_body or "").splitlines(),
            lineterm="",
        )
    )
    diff_text = "\n".join(diff_lines)
    return (
        f"\n\nCURRENT DRAFT for type '{state['artifact_type']}' has changed since last turn "
        f"(unified diff, not the full body):\n{diff_text}"
    )


def _build_key_facts_block(state: WorkflowState) -> str:
    """Accumulated key facts: confirmed data points the analyst must not re-ask or contradict."""
    facts = state.get("key_facts") or []
    if not facts:
        return ""
    lines = "\n".join(f"- {f['statement']}" + (f" (source: {f['source']})" if f.get("source") else "") for f in facts)
    return f"\n\nConfirmed key facts (do not ask again):\n{lines}"


def _build_situation_report_block(state: WorkflowState) -> str:
    return render_situation_report(state.get("lifecycle_reports") or [])


def _build_artifact_history_block(state: WorkflowState) -> str:
    return render_artifact_history(state.get("artifact_history") or [])


def _build_artifact_reference_policy_block(artifacts: list[dict], current_artifact_type: str) -> str:
    lines = [
        "- If the latest user turn references uploaded/pasted source documents, call "
        "`read_source_documents` before drafting; omit ids when you need bounded excerpts from the "
        "latest project source documents.",
    ]
    if any(str(item.get("type") or "") != current_artifact_type for item in artifacts):
        lines.extend(
            [
                "- If the latest user turn asks to base this work on a named artifact in Current context, "
                "call `read_artifact` for that artifact id before asking the user for content or an id.",
                "- Ask the user to paste content or provide an artifact id only after no matching Current "
                "context artifact exists, or `read_artifact` reports that it was not found.",
            ]
        )
    lines.append(
        "- Do not bundle `read_artifact` or `read_source_documents` with `ask_user`, `respond`, "
        "`write_draft`, or `finalize`; read first, then synthesize, draft, or ask on the next turn."
    )
    return "\n\nARTIFACT REFERENCE POLICY:\n" + "\n".join(lines) + "\n\n"


def _build_tool_selection_prompt(
    state: WorkflowState,
    artifacts: list[dict],
    draft_body: str | None = None,
    previous_draft_body: str | None = None,
    session_id: str | None = None,
) -> str:
    """Build the per-turn analyst payload: context the model needs to pick the next tool.

    This is dynamic payload only — artifact context, the tools available this turn, and
    state-dependent hints (coverage gaps, the running/persisted draft, a one-shot mode_hint, the
    locale lock). The conversation itself is NOT restated here: the analyst receives it as a real
    message thread (_build_analyzer_messages), so only a running summary of older turns is carried.
    All static policy — tool semantics, the section grading rubric, the proactive-mode and
    content-depth rules — lives in the instruction layers (the system prompt), so it is never
    restated here. analyze_node converts the returned dict into an AIMessage(tool_calls).
    """
    artifact_context = (
        "\n".join(f"- [{a['type']}] {a['title']} (id={a['id']})" for a in artifacts) or "(no artifacts yet)"
    )
    artifact_reference_policy = _build_artifact_reference_policy_block(artifacts, state["artifact_type"])

    # The analyst already receives the full conversation as a real message thread
    # (_build_analyzer_messages), so restating it here would double every recent turn. The payload
    # carries only the running summary — a deliberate compaction of older turns — when one exists.
    conversation_summary = (state.get("conversation_summary") or "").strip()
    summary_block = f"Accumulated conversation summary:\n{conversation_summary}\n\n" if conversation_summary else ""

    locale = (state.get("locale") or "").strip()
    language_lock = (
        f"\n\nIMPORTANT: Respond entirely in language '{locale}'. Do not mix in another language." if locale else ""
    )

    tool_menu = ", ".join(t.name for t in get_available_tools(state))
    if _phase_includes(state, "decision_view"):
        decision_view_block = _build_decision_view_block(state, session_id)
        draft_block = (
            ""
            if _decision_view_can_hide_draft(state, decision_view_block, draft_body)
            else _build_draft_delta_block(state, draft_body, previous_draft_body)
        )
    else:
        # INTENT/ELICIT have no draft to reason over; FINALIZE reasons over the readiness summary,
        # not the full body — so the draft and its rendered view stay out of those phases' payload.
        decision_view_block = ""
        draft_block = ""
    section_coverage_hint = _build_section_coverage_hint(state) if _phase_includes(state, "section_coverage") else ""
    feedback_block = _build_feedback_control_block(state)
    key_facts_block = _build_key_facts_block(state)
    situation_report_block = _build_situation_report_block(state)
    artifact_history_block = _build_artifact_history_block(state)
    # Taxonomy chain + section-coverage contract are no longer here — they moved to the system prompt
    # (see _build_artifact_contract_block) so the per-turn payload stays small next to the conversation.
    return (
        f"You are the analyst for artifact type: {state['artifact_type']}.\n\n"
        f"Current context:\n{artifact_context}\n\n"
        f"{situation_report_block}"
        f"{artifact_history_block}"
        f"{artifact_reference_policy}"
        f"{summary_block}"
        f"Tools available this turn: {tool_menu}.\n"
        "Choose 1-3 suitable tools and fill each tool's fields according to the system prompt policy."
        f"{section_coverage_hint}"
        f"{key_facts_block}"
        f"{feedback_block}"
        f"{draft_block}"
        f"{decision_view_block}"
        f"{_build_mode_hint_directive(state)}"
        f"{language_lock}"
    )


# A signal that persists unaddressed for at least this many consecutive turns (feedback_summary's
# ignored_counts) escalates its wording in the feedback block; below threshold wording is unchanged.
_IGNORED_SIGNAL_THRESHOLD = 2


def _build_feedback_control_block(state: WorkflowState) -> str:
    parts: list[str] = []
    report = state.get("quality_report") or {}
    if report:
        parts.append(f"- quality_gate: {report.get('quality_gate_result') or 'unknown'}")
        blockers = report.get("blocking_issues") or []
        if blockers:
            parts.append(f"- blockers: {_compact_list(blockers)}")
        revision_plan = report.get("revision_plan") or []
        if revision_plan:
            parts.append(f"- revision_plan: {_compact_list(revision_plan)}")
        if report.get("recommended_next_action"):
            parts.append(f"- recommended_next_action: {report['recommended_next_action']}")

    readiness = state.get("candidate_readiness") or {}
    if readiness:
        parts.append(f"- candidate_readiness: {readiness.get('state') or 'unknown'}")
        for key in ("missing", "needs_confirmation", "blocking_reasons"):
            values = readiness.get(key) or []
            if values:
                parts.append(f"- {key}: {_compact_list(values)}")

    diagnosis = state.get("diagnosis_signal") or {}
    if diagnosis and (diagnosis.get("risk_level") == "high" or diagnosis.get("judge_result")):
        parts.append(f"- diagnosis_risk: {diagnosis.get('risk_level') or 'unknown'}")
        signals = diagnosis.get("signals") or []
        if signals:
            parts.append(f"- diagnosis_signals: {_compact_list(signals)}")
        judge = diagnosis.get("judge_result") or {}
        if judge:
            parts.append(f"- diagnosis_judge_score: {judge.get('score', 'unknown')}")
            findings = judge.get("findings") or []
            if findings:
                parts.append(f"- diagnosis_judge_findings: {_compact_list(findings)}")

    feedback_summary = state.get("feedback_summary") or {}
    resurfaced = feedback_summary.get("resurfaced_questions") or []
    if resurfaced:
        rendered = "; ".join(f"{item.get('id')}: {item.get('statement')}" for item in resurfaced[:3])
        ignored_count = (feedback_summary.get("ignored_counts") or {}).get("resurfaced_questions") or 0
        if ignored_count >= _IGNORED_SIGNAL_THRESHOLD:
            parts.append(f"- resurfaced_questions: URGENT (ignored {ignored_count} turns): {rendered}")
        else:
            parts.append(f"- resurfaced_questions: {rendered}")
    if feedback_summary.get("depth_signal"):
        parts.append(f"- depth_signal: {feedback_summary['depth_signal']}")
    sweep_gaps = feedback_summary.get("sweep_gaps") or []
    if sweep_gaps:
        parts.append(f"- sweep_gaps: {_compact_list(sweep_gaps)}")
    created_parked = feedback_summary.get("created_parked_questions") or []
    if created_parked:
        rendered = "; ".join(f"{item.get('id')}: {item.get('statement')}" for item in created_parked[:3])
        parts.append(f"- created_parked_questions: {rendered}")
    if feedback_summary.get("stale_warning"):
        parts.append(f"- stale_warning: {feedback_summary['stale_warning']}")
    stale_base = feedback_summary.get("stale_base_version") or {}
    if stale_base:
        parts.append(
            "- stale_base_version: the base artifact changed under you "
            f"(base {stale_base.get('base_version_id')} -> current {stale_base.get('current_version_id')}); "
            "re-read the artifact and rebase before drafting or finalizing."
        )
    lifecycle_rejection = feedback_summary.get("lifecycle_persist_rejection") or {}
    if lifecycle_rejection:
        stale_predecessors = lifecycle_rejection.get("stale_predecessors") or []
        rendered = _compact_list(
            [
                f"{item.get('artifact_id')}: {item.get('reason')}"
                if isinstance(item, dict)
                else str(item)
                for item in stale_predecessors
            ]
        )
        suffix = f" Changed predecessors: {rendered}." if rendered else ""
        parts.append(
            "- lifecycle_persist_rejection: predecessor artifacts changed after this draft was prepared; "
            f"re-read upstream artifacts and rebase before proposing again.{suffix}"
        )
    readiness_rejection = feedback_summary.get("candidate_readiness_rejection") or {}
    if readiness_rejection:
        blockers = readiness_rejection.get("blocking_reasons") or readiness_rejection.get("missing") or []
        rendered = _compact_list(blockers)
        suffix = f" Blockers: {rendered}." if rendered else ""
        parts.append(
            "- candidate_readiness_rejection: the draft was not ready to persist "
            f"(state={readiness_rejection.get('state', 'unknown')}); revise before proposing again.{suffix}"
        )
    dropped = feedback_summary.get("dropped_tools") or []
    if dropped:
        parts.append(
            "- skipped last turn (not run because it was bundled with an interrupting tool and must run separately); "
            f"call it again in a separate turn if still needed: {_compact_list(dropped)}"
        )
    out_of_phase = feedback_summary.get("out_of_phase_tools") or {}
    if out_of_phase.get("dropped"):
        parts.append(
            f"- rejected last turn (outside the current session phase '{out_of_phase.get('phase')}'); "
            f"do not call again in this phase — pick from the tools offered this turn: "
            f"{_compact_list(out_of_phase['dropped'])}"
        )
    lifecycle_blocked = feedback_summary.get("lifecycle_blocked_tools") or []
    if lifecycle_blocked:
        rendered = "; ".join(
            f"{item.get('name')}: {item.get('reason')} ({item.get('state')})" for item in lifecycle_blocked[:3]
        )
        parts.append(f"- rejected last turn by lifecycle state: {rendered}")

    parts.extend(_repeated_tool_error_lines(state))

    if not parts:
        return ""
    return (
        "\n\nFEEDBACK CONTROL:\n"
        "- the signals below are orchestration priorities; choose suitable tools and order without ignoring them.\n"
        + "\n".join(parts)
    )


# Same-code failures at or above this count escalate: a single failure is served by its ToolMessage,
# a repeat means the model is looping on the same error and must change approach.
_REPEATED_TOOL_ERROR_THRESHOLD = 2


def _repeated_tool_error_lines(state: WorkflowState) -> list[str]:
    """Escalation lines for tool errors whose `code` recurs (>= threshold) in the accumulated
    `tool_errors` channel. Mirrors `_build_stuck_escalation_block`'s "change approach" steer, but
    keyed on the error code rather than repeated identical tool calls. Empty when nothing recurs."""
    errors = state.get("tool_errors") or []
    counts: dict[str, int] = {}
    recovery_by_code: dict[str, str] = {}
    for entry in errors:
        code = entry.get("code")
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
        if entry.get("recovery"):
            recovery_by_code[code] = entry["recovery"]
    lines: list[str] = []
    for code, count in counts.items():
        if count < _REPEATED_TOOL_ERROR_THRESHOLD:
            continue
        recovery = recovery_by_code.get(code)
        steer = f" {recovery}" if recovery else " Change approach — do not repeat the same call."
        lines.append(f"- repeated tool_errors: '{code}' has failed {count} times.{steer}")
    return lines


def _compact_list(values: list[Any], limit: int = 3) -> str:
    rendered = [str(value) for value in values[:limit] if str(value).strip()]
    if len(values) > limit:
        rendered.append(f"... (+{len(values) - limit})")
    return "; ".join(rendered)


# Technique hints per thinking mode. Every name here must exist in ELICIT_TECHNIQUES.
_THINKING_MODE_TECHNIQUE_HINTS: dict[str, tuple[str, ...]] = {
    "challenging": ("reverse", "first_principles", "challenge_assumptions"),
    "risk_probing": ("5_whys", "reverse", "pre_mortem"),
}
_THINKING_MODE_RATIONALE: dict[str, str] = {
    "challenging": (
        "Diagnosis flagged this section as low-coverage on a non-empty draft -- challenge "
        "existing assumptions before accepting the current shape."
    ),
    "risk_probing": (
        "Diagnosis flagged this section as low-coverage after a failed quality gate -- probe "
        "root causes and risks before proposing content."
    ),
}


def _build_thinking_mode_block(state: WorkflowState) -> str:
    """Thinking-mode guidance appended to the system prompt after the artifact contract block.

    Returns "" for an unset or low-risk thinking mode so the fast path's prompt stays
    byte-identical when omitted. get_instruction()'s cached, role-keyed assembly is never touched
    -- this is a per-turn suffix, same mechanism as _build_artifact_contract_block.
    """
    thinking_mode = state.get("thinking_mode")
    techniques = _THINKING_MODE_TECHNIQUE_HINTS.get(thinking_mode or "")
    if not techniques:
        return ""
    rationale = _THINKING_MODE_RATIONALE.get(thinking_mode, "")
    return (
        f"\n\nTHINKING MODE: {thinking_mode}\n{rationale}\n"
        f"Favor these elicit() techniques this turn: {', '.join(techniques)}."
    )


def _is_near_stuck(recent_tool_calls: list[str]) -> bool:
    """True one repeat before route_node's hard-stop threshold fires.

    Mirrors _has_repeated_tool_calls's tail-identity check but at _REPEATED_TOOL_CALL_EXIT_THRESHOLD
    - 1 fingerprints, so the model can be warned to change course before route_node exits the loop.
    Purely advisory -- never itself ends the turn; route_node's threshold and logic are unchanged.
    """
    threshold = _REPEATED_TOOL_CALL_EXIT_THRESHOLD - 1
    if len(recent_tool_calls) < threshold:
        return False
    tail = recent_tool_calls[-threshold:]
    return len(set(tail)) == 1


def _build_stuck_escalation_block(state: WorkflowState) -> str:
    """Prompt suffix warning the model it is one repeat away from route_node's early exit.

    Returns "" when adaptive diagnosis is disabled or the model is not near-stuck, keeping the fast
    path byte-identical when omitted.
    """
    if not settings.enable_adaptive_diagnosis:
        return ""
    if not _is_near_stuck(state.get("recent_tool_calls") or []):
        return ""
    return (
        "\n\nLOOP WARNING: You have called the same tool with the same arguments twice in a row. "
        "One more identical call will end this turn's analysis early. Change your approach -- use "
        "a different tool, different arguments, or write_draft/ask_user -- to make progress."
    )


def _build_batching_instruction_block(_state: WorkflowState) -> str:
    """ELICIT-only steer: batch related clarifications into one ask_user call.

    A per-turn suffix, not part of get_instruction()'s cached string. Attacks the serial
    one-question-at-a-time chains that made elicitation feel like an interrogation.
    """
    return (
        "\n\nBATCHING: when you need clarification, group related facets of ONE topic into a single "
        "ask_user call via its `questions` list (up to 3, each typed choice/text/confirm). Ask "
        "serially — one question, no batch — only when an answer determines the next question. Never "
        "fan out unrelated questions in one call."
    )


def _build_artifact_contract_block(state: WorkflowState) -> str:
    """Artifact-type shape appended to the SYSTEM prompt (L1), not the per-turn payload.

    The taxonomy chain and the section-coverage contract depend only on artifact_type (stable per
    session), so they belong with the static policy — kept out of the per-turn user payload so they
    do not compete with the live conversation for the recency slot.
    """
    return _build_taxonomy_chain_block(state) + _build_output_contract_block(state)


def _build_taxonomy_chain_block(state: WorkflowState) -> str:
    """Per-turn provenance: the focused artifact type plus its ancestry, each with the registry
    description. Replaces the full static taxonomy catalog — the model needs only the chain it
    derives from this turn, not every type in the engine (memory/context holds the evidence)."""
    artifact_type = state["artifact_type"]
    chain = [*reversed(ancestor_types(artifact_type)), artifact_type]
    lines = []
    for item_type in chain:
        try:
            desc = get_config(item_type).description
        except (KeyError, ValueError):
            continue
        marker = " (current)" if item_type == artifact_type else ""
        lines.append(f"- {item_type}{marker}: {desc}")
    if not lines:
        return ""
    return "\n\nARTIFACT TYPE & provenance chain:\n" + "\n".join(lines)


def _build_output_contract_block(state: WorkflowState) -> str:
    artifact_type = state["artifact_type"]
    try:
        contract = output_contract(artifact_type)
    except ValueError:
        return ""
    headings = "\n".join(f"- {heading}" for heading in contract.required_headings)
    columns = ", ".join(contract.table_columns) if contract.table_columns else "(table not required)"
    # When the contract carries an id_prefix the first column is an auto-assigned trace tag the agent
    # must not fill; other artifacts reference an entry by that tag instead of restating it.
    id_rule = (
        "\nEvery node must fill all of these fields; if a value is genuinely unknown, set it to "
        "'(needs confirmation)' rather than leaving it empty.\n"
        f"The 'id' column is assigned automatically as {contract.id_prefix}-NN — do not set it. Reference "
        f"another requirement by its id (e.g. {contract.id_prefix}-01) instead of restating its text.\n"
        if contract.id_prefix
        else ""
    )
    # Graph-first: the artifact view renders from decision nodes, so the contract is a coverage target
    # for the nodes to fill — not a Markdown body to hand-write. Only the flag-off rollback path still
    # authors a body directly, so keep the body-shape contract for that case.
    if settings.decision_graph_enabled:
        # Keep only artifact-specific content in the per-turn payload; node/status/no-fabrication
        # policy already lives in the system prompt, so do not repeat it here.
        return (
            "\n\nSECTION COVERAGE REQUIRED (view rendered from the decision graph - "
            "create nodes to fill it, do not hand-write the Markdown body):\n"
            f"{headings}\n"
            f"Table columns when using a table: {columns}\n"
            f"{id_rule}"
            "Prioritize current/accepted artifact versions and accepted predecessors over chat history."
        )
    return (
        "\n\nREQUIRED OUTPUT CONTRACT:\n"
        f"- Artifact type: {artifact_type}\n"
        "- Body must be Markdown following this artifact standard, not a JSON/form dump.\n"
        "- Conversation/user input is only evidence/context; do not copy the transcript into the body.\n"
        "- Agent-inferred content or content needing user confirmation must be noted inline in parentheses, "
        f"for example {contract.confirmation_note}.\n"
        "- When input is thin, the candidate must still keep the full structure and mark clearly: "
        "`inferred` for agent-inferred content, `missing` for missing evidence, "
        "`needs_confirmation` for assumptions needing user confirmation.\n"
        "- Do not weaken the body by dropping headings; if data is insufficient, keep headings "
        "and mark missing content clearly.\n"
        f"- Guidance: {contract.guidance}\n"
        "Required headings:\n"
        f"{headings}\n"
        f"Table columns when using a table: {columns}\n"
        "Prioritize current/accepted artifact versions and accepted predecessors over chat history."
    )


def _build_mode_hint_directive(state: WorkflowState) -> str:
    """Inject a user-supplied `mode_hint` — an explicit override to switch operating angle this turn.

    Dynamic per-turn payload only. The proactive-mode policy (when to leave plain Q&A, prefer
    respond over burying an assessment in a question) is static and lives in the decision-policy
    instruction layer, not here.
    """
    mode_hint = (state.get("mode_hint") or "").strip()
    if not mode_hint:
        return ""
    return (
        f"\n\nMODE REQUEST: the user wants to switch to mode '{mode_hint}'. Switch immediately "
        f"this turn and respond according to that mode."
    )


def _build_section_coverage_hint(state: WorkflowState) -> str:
    if state.get("coverage_complete") is not False:
        return ""
    section_coverage = state.get("section_coverage") or {}
    # Stall: coverage stopped advancing — re-pinning the same gaps would reproduce the previous
    # question verbatim, so steer the model to synthesize what it has and move on or propose.
    if (state.get("section_coverage_stall_count") or 0) >= 2:
        return (
            "\n\nSection coverage has not improved across multiple turns. Do not repeat the same exploration path - "
            "synthesize what exists and consider proposing, or switch to a completely different angle."
        )
    # Gap-inventory: list every weak section (missing first, then partial/needs_review) so the LLM
    # picks the angle that fits the conversation instead of being pinned to one scripted question.
    gap_lines = [
        f"- {get_config(section).description} ({section_coverage.get(section)})"
        for status in ("missing", "partial", "needs_review")
        for section in section_coverage
        if section_coverage.get(section) == status
    ]
    inventory = "\n".join(gap_lines)
    return (
        "\n\nSection coverage - aspects still missing or unclear (reference only, not required order):\n"
        f"{inventory}\n"
        "Choose the best angle for the conversation flow to advance - explore more, make reasonable "
        "inferences, or draft when enough is known."
    )


def build_system_prompt(state: WorkflowState, agent_role: str | None, *, has_draft: bool) -> str:
    """The full analyst system prompt: instruction layers + per-turn suffix blocks, in the exact
    pre-decomposition order (contract, thinking mode, stuck escalation)."""
    system_prompt = get_instruction(
        artifact_type=state["artifact_type"],
        workflow_area=state["workflow_area"],
        agent_role=agent_role,
        context={"has_draft": has_draft},
    )
    # Artifact-type shape (taxonomy chain + section-coverage contract) belongs with the static policy
    # in L1, not the per-turn payload — appended last so the static prefix stays cache-friendly.
    # The two artifact/technique suffixes are gated by session phase: the full contract is a
    # drafting concern; the technique shortlist is an elicitation concern. Stuck-escalation is
    # cross-cutting and always appended.
    system_prompt = system_prompt or ""
    if _phase_includes(state, "artifact_contract"):
        system_prompt = system_prompt + _build_artifact_contract_block(state)
    if _phase_includes(state, "thinking_mode"):
        system_prompt = system_prompt + _build_thinking_mode_block(state)
    if _phase_includes(state, "batching"):
        system_prompt = system_prompt + _build_batching_instruction_block(state)
    if _phase_includes(state, "section_repair"):
        system_prompt = system_prompt + _build_section_repair_block(state)
    if _phase_includes(state, "type_profile"):
        system_prompt = system_prompt + _build_type_profile_block(state)
    return system_prompt + _build_stuck_escalation_block(state)


def _build_type_profile_block(state: WorkflowState) -> str:
    """Per-artifact-type behavior within the current phase (data-driven, from the output contract).

    ELICIT renders the type's elicitation checklist + suggested technique; REVIEW renders the type's
    critique criteria. The DRAFT section scaffold is already carried by the artifact-contract block
    (required_headings), so it is not repeated here. A type with no profile fields, or an unknown
    type, yields "" — the phase behaves generically (no-op), preserving today's behavior.
    """
    try:
        contract = output_contract(state["artifact_type"])
    except ValueError:
        return ""
    phase = state.get("session_phase")
    if phase == ELICIT:
        lines = [f"- {item}" for item in contract.elicit_checklist]
        if contract.elicit_technique:
            lines.append(f"Suggested technique: elicit(technique='{contract.elicit_technique}').")
        if not lines:
            return ""
        return "\n\nELICITATION FOCUS for this artifact type:\n" + "\n".join(lines)
    if phase == REVIEW:
        if not contract.review_criteria:
            return ""
        criteria = "\n".join(f"- {item}" for item in contract.review_criteria)
        return "\n\nREVIEW CRITERIA for this artifact type:\n" + criteria
    return ""


def _build_section_repair_block(state: WorkflowState) -> str:
    """Targeted repair line for sections whose last write recorded a structural finding.

    Severity is phase-scoped to avoid training the model to ignore the line: only `violation`
    findings surface in ELICIT (early, terse sections should not be nagged for style); warnings join
    from DRAFT onward. Findings clear when the section re-validates clean, so a repaired section
    drops off the list next turn.
    """
    findings_map = state.get("section_findings") or {}
    if not findings_map:
        return ""
    warnings_ok = state.get("session_phase") != ELICIT
    lines: list[str] = []
    for section, findings in findings_map.items():
        for finding in findings or []:
            severity = finding.get("severity")
            if severity == "warning" and not warnings_ok:
                continue
            lines.append(f"- {section}: {finding.get('message')} ({severity})")
    if not lines:
        return ""
    return (
        "\n\nSECTION REPAIR — fix these written sections before advancing:\n" + "\n".join(lines)
    )
