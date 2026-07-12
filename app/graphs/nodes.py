import logging
import re
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from sqlalchemy import exists, select

from app.config import settings
from app.documents.registry import children_of, status_score
from app.graphs.agent_tools import DIAGNOSIS_JUDGE_CALLS_MAX, _phase_signals, current_session_phase, get_available_tools

# analyze_node's concerns live in app.graphs.analysis.* now. The private
# names are re-exported here because existing tests/evals import them from nodes.
from app.graphs.analysis.context_loader import (  # noqa: F401
    TurnContext,
    _document_coverage,
    _missing_required_headings,
    load_turn_context,
)
from app.graphs.analysis.prompt_assembly import (  # noqa: F401
    _THINKING_MODE_RATIONALE,
    _THINKING_MODE_TECHNIQUE_HINTS,
    _analyzer_history_messages,
    _build_analyzer_messages,
    _build_artifact_contract_block,
    _build_artifact_history_block,
    _build_draft_block,
    _build_draft_delta_block,
    _build_feedback_control_block,
    _build_key_facts_block,
    _build_mode_hint_directive,
    _build_output_contract_block,
    _build_section_coverage_hint,
    _build_situation_report_block,
    _build_stuck_escalation_block,
    _build_thinking_mode_block,
    _build_tool_selection_prompt,
    _compact_list,
    _is_human_turn,
    _is_near_stuck,
    _latest_human_text,
    _msg_role_content,
    build_system_prompt,
)
from app.graphs.analysis.section_validation import validated_coverage
from app.graphs.analysis.tool_gating import (  # noqa: F401
    _COERCED_ASK_FALLBACK_BY_LOCALE,
    _INTERRUPT_BEARING_TOOLS,
    _RESPOND_FALLBACK_BY_LOCALE,
    _SIDE_EFFECT_FREE_NOTE_TOOLS,
    _ai_text_content,
    _build_tool_schemas,
    _dropped_tool_names,
    _gate_selected_tools,
    _log_tool_error,
    _looks_like_question,
    _model_tool_calls,
    _plain_response_tool,
    _response_message_incomplete,
    gate_model_selection,
    required_args,
)
from app.graphs.analysis.turn_audit import (  # noqa: F401
    _RECENT_TOOL_CALLS_MAXLEN,
    _REPEATED_TOOL_CALL_EXIT_THRESHOLD,
    _audit_tool_call,
    _estimate_token_breakdown,
    _has_repeated_tool_calls,
    _tool_call_fingerprint,
    annotate_token_usage,
    append_turn_fingerprint,
    build_analysis_result_base,
    record_run_and_dispatch,
)
from app.graphs.decision_graph import (
    add_parked_questions_for_gaps,
    completeness_sweep,
    is_brd_stable,
    migrate_legacy_notes,
    scan_parked_questions,
)

# Moved to app.graphs.interrupts (neutral leaf) to break the nodes ↔ agent_tools import cycle;
# re-exported here because existing tests and callers import them from nodes.
from app.graphs.interrupts import (  # noqa: F401
    _agent_message_already_saved,
    _save_and_interrupt_ask,
)
from app.graphs.session_phase import DRAFT, FINALIZE, REVIEW, IllegalPhaseTransition
from app.graphs.session_phase import derive_phase as _derive_phase
from app.graphs.session_phase import transition as phase_transition
from app.graphs.state import DEFAULT_METHOD_PROFILE, WorkflowState
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
)

logger = logging.getLogger(__name__)

# Native tool calling replaces the old JSON tool-selection schema: analyze_node binds the available
# tool schemas to the provider API (see _build_tool_schemas) and the model returns native tool_calls.
# Analytic fields (locale, workflow_mode) are derived from the picked tool + state.

# Valid planning tracks; _normalize_planning_track falls back to quick on miss.
_PLANNING_TRACKS = {"quick", "standard", "enterprise"}

# Tool schemas, gating, audit hashing, and token estimation moved to app.graphs.analysis.*;
# re-exported below for existing import paths.


def _normalize_planning_track(track: Any) -> str:
    raw = str(track or "").strip().lower()
    return raw if raw in _PLANNING_TRACKS else "quick"


def _derive_artifact_chain(section_coverage: dict[str, str] | None) -> dict[str, str]:
    """BMAD artifact-chain status (missing/partial/complete) derived from 7-section coverage.

    Sole source is section_coverage mapped to 0.0–1.0 scores — no 9-slot data.
    brief tracks the first four sections; prd tracks all seven.
    """
    cov = section_coverage or {}
    brd_items = children_of("brd")
    scores = {section: status_score(cov.get(section)) for section in brd_items}
    nonzero = [v for v in scores.values() if v > 0]
    strong = [v for v in scores.values() if v >= 0.5]

    if not nonzero:
        brainstorming = "missing"
    elif len(strong) >= 5:
        brainstorming = "complete"
    else:
        brainstorming = "partial"

    brief_sections = ["problem_statement", "vision_objectives", "stakeholder_register", "scope_capabilities"]
    brief_scores = [scores[s] for s in brief_sections]
    if all(v >= 0.6 for v in brief_scores):
        product_brief = "complete"
    elif any(v > 0 for v in brief_scores):
        product_brief = "partial"
    else:
        product_brief = "missing"

    all_scores = list(scores.values())
    if all(v >= 0.7 for v in all_scores):
        prd = "complete"
    elif any(v > 0 for v in all_scores):
        prd = "partial"
    else:
        prd = "missing"

    return {"brainstorming": brainstorming, "product_brief": product_brief, "prd": prd}


def _infer_workflow_mode(state: WorkflowState) -> str:
    """Fallback workflow_mode from section coverage when the LLM does not report one.

    Low coverage everywhere -> still brainstorming. Once some signal exists, suggest the next
    planning artifact by what the session targets. Never an override — only a default.
    """
    coverage = state.get("section_coverage") or {}
    scored = {"filled": 1.0, "needs_review": 0.5, "partial": 0.5, "missing": 0.0}
    ratios = [scored.get(v, 0.0) for v in coverage.values()]
    if not ratios or max(ratios) < 0.3:
        return "brainstorm"
    return "prd" if state.get("artifact_type") == "product_brief" else "brief"


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}

SUMMARY_SYSTEM = (
    "You summarize product requirements conversations. "
    "Preserve important constraints, especially numbers, names, deadlines, and scope."
)

# Triage schema: classify a fresh turn and, for a conversational one, draft the reply in one cheap
# call. `reply` is only meaningful when turn_type == "converse".
TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "turn_type": {"type": "string", "enum": ["converse", "work"]},
        "locale": {"type": "string", "enum": ["vi", "en"]},
        "reply": {"type": "string"},
    },
    "required": ["turn_type"],
}

TRIAGE_SYSTEM = (
    "You classify the opening turn for a product requirements analyst assistant. "
    "Decide whether this turn is small talk or requirements analysis work."
)

# Locale-templated fallback when the classifier returns no reply text for a converse turn.
_FALLBACK_GREETING = {
    "vi": "Xin chào! Tôi là trợ lý phân tích yêu cầu. Bạn muốn bắt đầu từ đâu?",
    "en": "Hello! I'm your requirements analysis assistant. Where would you like to start?",
}

# Mid-session triage-skip heuristic: only fires in draft/review/finalize (unreachable on turn one),
# and only for a message long enough and not matching a bare greeting/smalltalk/ack pattern —
# anything shorter or ambiguous still falls through to the real LLM triage call below.
_TRIAGE_SKIP_MIN_LENGTH = 12
_GREETING_ONLY_PATTERN = re.compile(
    r"^\s*(hi+|hello+|hey+|xin\s+ch[aà]o|ch[aà]o(\s+b[aạ]n)?|c[aả]m\s*[ơo]n|thank\s*you|thanks?"
    r"|ok(ay)?|oke|đ[ưu][ợo]c|v[aâ]ng|d[aạ]|[ừu]m?)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _triage_heuristic_certain_work(phase: str | None, message: str) -> bool:
    """True when phase + message content make the LLM triage call for this turn unnecessary."""
    if phase not in (DRAFT, REVIEW, FINALIZE):
        return False
    stripped = message.strip()
    if len(stripped) < _TRIAGE_SKIP_MIN_LENGTH:
        return False
    return not _GREETING_ONLY_PATTERN.match(stripped)


async def triage_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """Entry node: classify a fresh turn so a greeting/smalltalk skips the full analyst pass.

    One cheap LLM call (the standard client, not the strong one) decides ``converse`` vs ``work``,
    detects the locale, and — for a conversational turn — drafts the reply in the same call.
    ``work`` falls straight through to analyze_node. The classifier runs once per fresh invocation;
    on resume LangGraph re-enters the interrupted node (converse/tools), so it never re-runs.

    Mid-session (draft/review/finalize) with an unambiguous work message skips this call entirely
    (`_triage_heuristic_certain_work`) — the phase and locale are already established from earlier
    turns, so there is nothing left for the classifier to resolve.
    """
    last_user = ""
    for m in reversed(state.get("messages") or []):
        role, content = _msg_role_content(m)
        if role == "user":
            last_user = content
            break

    phase = current_session_phase(state)
    if _triage_heuristic_certain_work(phase, last_user):
        return {"turn_type": "work", "locale": state.get("locale"), "triage_reply": None}

    cfg = config["configurable"]
    llm_client = cfg["llm_client"]
    if llm_client is None:
        raise ValueError("LLM provider is not configured. Add an API key in settings.")

    prompt = (
        "Classify the user's message.\n\n"
        f"Message: {last_user!r}\n\n"
        "turn_type: 'converse' if only greeting, thanks, small talk, or off-topic; "
        "'work' if is a request to analyze, clarify, or create an artifact.\n"
        "locale: 'vi' if Vietnamese, 'en' if English.\n"
        "If turn_type='converse', set 'reply' to a short, friendly sentence in the user's exact language "
        "- greet back, briefly say what you can help with, and invite them to share what they want to build."
    )
    started_at = time.monotonic()
    try:
        result, _usage = await llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            system=TRIAGE_SYSTEM,
            max_tokens=300,
            response_format=TRIAGE_SCHEMA,
        )
    except ValueError:
        # A non-conforming classifier response must not crash the turn — default to a work turn so
        # it falls through to the full analyst pass (the safe, non-conversational branch).
        result = {}
    logger.debug("node=triage latency_ms=%d", int((time.monotonic() - started_at) * 1000))
    reported = result if isinstance(result, dict) else {}
    turn_type = reported.get("turn_type") or "work"
    locale = state.get("locale") or reported.get("locale")
    reply = reported.get("reply") if turn_type == "converse" else None
    return {"turn_type": turn_type, "locale": locale, "triage_reply": reply}


def route_after_triage(state: WorkflowState) -> str:
    """Conversational turns peel off to converse_node; everything else goes to the analyst.

    Non-converse turns enter via `orchestrator` (which flows into analyze); the label matches its
    target node name.
    """
    return "converse" if state.get("turn_type") == "converse" else "orchestrator"


async def converse_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """Reply to a conversational turn and pause for the human — no analyst pass, no LLM call here.

    Uses the reply triage already drafted (``triage_reply``), so on resume this node re-runs without
    any new model call: the idempotent save (keyed on content) skips, and ``interrupt`` returns the
    human reply, which flows on to analyze_node for the real work.
    """
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    locale = state.get("locale") or "vi"
    message = (state.get("triage_reply") or "").strip() or _FALLBACK_GREETING.get(locale, _FALLBACK_GREETING["vi"])

    async with session_factory() as db:
        already_saved = (
            await db.execute(
                select(
                    exists().where(
                        AgentMessage.session_id == session_id,
                        AgentMessage.role == AgentMessageRole.AGENT,
                        AgentMessage.content == message,
                    )
                )
            )
        ).scalar()
        if not already_saved:
            db.add(
                AgentMessage(
                    session_id=session_id,
                    role=AgentMessageRole.AGENT,
                    content=message,
                    payload={"kind": "greeting", "locale": locale},
                )
            )
        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.ASK_HUMAN
        await db.commit()

    user_response = interrupt({"type": "ask_human", "message": message})
    user_content = user_response.get("content", "") if isinstance(user_response, dict) else str(user_response or "")
    return {
        "messages": [
            {"role": "assistant", "content": message},
            {"role": "user", "content": user_content},
        ]
    }


# Thinking modes the diagnosis heuristic can select. Mapping: low risk -> structuring (cold
# start/early section) or synthesizing (section mostly filled); high risk -> challenging (weak
# coverage on a non-empty draft) or risk_probing (a prior quality-gate failure on top of weak
# coverage).
_THINKING_MODES = ("structuring", "challenging", "synthesizing", "risk_probing")

# Coverage below settings.low_coverage_ratio counts as "low coverage" for the diagnosis
# conjunction. Read from settings (not a module constant) so a single weak signal can't silently
# escalate every section, and so eval sweeps can vary it via env.
_COVERAGE_SCORES = {"filled": 1.0, "needs_review": 0.5, "partial": 0.5, "missing": 0.0}


def _diagnose_section(state: WorkflowState) -> dict[str, Any]:
    """Cheap, LLM-free risk/ambiguity diagnosis for the current section.

    Never calls an LLM — pure function of state already available in-turn.

    Cold start (no coverage data yet, e.g. turn 1): defaults to low risk / "structuring" rather
    than erroring or guessing high risk. This is intentional, not an overlooked edge case —
    adaptivity kicks in from turn 2 once coverage/quality data exists.
    """
    coverage = state.get("section_coverage")
    if not coverage:
        return {"risk_level": "low", "signals": [], "thinking_mode": "structuring"}

    # Validated coverage: a section carrying a structural `violation` finding does not
    # count as covered, so the ratio reflects "covered with acceptable content", not just presence.
    coverage = validated_coverage(coverage, state.get("section_findings"))
    ratios = [_COVERAGE_SCORES.get(value, 0.0) for value in coverage.values()]
    coverage_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    low_coverage = coverage_ratio < settings.low_coverage_ratio

    quality_report = state.get("quality_report") or {}
    quality_gate_failed = quality_report.get("quality_gate_result") == "fail"

    draft_body = state.get("draft_body")
    sparse_draft = not str(draft_body or "").strip()

    signals: list[str] = []
    if low_coverage:
        signals.append("low_coverage")
    if quality_gate_failed:
        signals.append("quality_gate_failed")
    if sparse_draft:
        signals.append("sparse_draft")

    # Named conjunction of concrete signals, not a single unbounded soft score: a lone weak signal
    # (e.g. low coverage alone on a fresh section) never escalates risk on its own.
    high_risk = (low_coverage and quality_gate_failed) or (low_coverage and sparse_draft)

    if high_risk:
        thinking_mode = "risk_probing" if quality_gate_failed else "challenging"
        return {"risk_level": "high", "signals": signals, "thinking_mode": thinking_mode}

    thinking_mode = "synthesizing" if coverage_ratio >= 0.6 else "structuring"
    return {"risk_level": "low", "signals": signals, "thinking_mode": thinking_mode}


def _diagnosis_llm_client(config: RunnableConfig | None) -> Any:
    """Tolerant LLM-client extraction for orchestrator_node's diagnosis step.

    Unlike analyze_node's strict config["configurable"]["llm_client"] access, orchestrator_node
    may run with config=None or config={} (existing test suite already exercises this), so a
    missing client must degrade to None rather than raise -- _invoke_judge already handles a
    None client gracefully.
    """
    if not config:
        return None
    cfg = config.get("configurable") or {}
    return cfg.get("strong_llm_client") or cfg.get("llm_client")


async def _apply_judge_escalation(
    diagnosis: dict[str, Any], state: WorkflowState, config: RunnableConfig | None
) -> dict[str, Any]:
    """Escalate a heuristic high-risk diagnosis to an LLM judge call, budget-gated.

    Low-risk sections never reach the judge. High-risk sections spend one judge call per turn up
    to DIAGNOSIS_JUDGE_CALLS_MAX; once the budget is exhausted the call is skipped with a distinct
    "escalation_skipped_budget" signal rather than silently degrading like "not_needed".
    """
    calls_used = state.get("diagnosis_judge_calls_used") or 0
    if diagnosis["risk_level"] != "high":
        return {"escalation": "not_needed", "judge_calls_used": calls_used}

    if calls_used >= DIAGNOSIS_JUDGE_CALLS_MAX:
        return {"escalation": "escalation_skipped_budget", "judge_calls_used": calls_used}

    from app.graphs.critique import _invoke_judge

    llm_client = _diagnosis_llm_client(config)
    started_at = time.monotonic()
    judge_result = await _invoke_judge(state.get("draft_body") or "", "risk_review", llm_client)
    logger.debug("node=judge latency_ms=%d", int((time.monotonic() - started_at) * 1000))
    return {"escalation": "escalated", "judge_calls_used": calls_used + 1, "judge_result": judge_result}


def _should_run_completeness_sweep(state: WorkflowState) -> bool:
    feedback = state.get("feedback_summary") or {}
    if state.get("completeness_sweep_requested") or state.get("user_requested_prd_descent"):
        return True
    if state.get("artifact_type") != "prd":
        return False
    if feedback.get("brd_stable_sweep_done"):
        return False
    return is_brd_stable(state.get("decision_nodes") or {})


async def orchestrator_node(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Pre-analyst orchestration: session-phase transition (single writer), parked-blocker
    resurfacing, and the completeness sweep when triggered."""
    # Session phase — the ONLY assignment site (session_phase.transition validates edge legality;
    # current=None adopts the derived phase, which is also the legacy-checkpoint migration).
    previous_phase = state.get("session_phase")
    signals = _phase_signals(state)
    from app.graphs.gate_logging import log_gate_decision as _log_phase

    try:
        session_phase = phase_transition(previous_phase, signals)
        reason = f"{previous_phase or 'unset'}->{session_phase}"
    except IllegalPhaseTransition:
        # An unmodeled composite must never silently strand a live session: adopt the derived phase
        # (always a safe target) and record the illegal edge for calibration rather than crashing.
        session_phase = _derive_phase(signals)
        reason = f"illegal:{previous_phase}->{session_phase}(adopted)"
        _log_phase(
            "session_phase_illegal_transition",
            "adopted",
            reason=reason,
            extra={"previous_phase": previous_phase, "adopted_phase": session_phase},
        )
    if session_phase != previous_phase:
        _log_phase("session_phase", session_phase, reason=reason)

    if settings.enable_adaptive_diagnosis:
        diagnosis = _diagnose_section(state)
        escalation = await _apply_judge_escalation(diagnosis, state, config)
        from app.graphs.gate_logging import log_gate_decision

        log_gate_decision(
            "diagnosis",
            diagnosis["risk_level"],
            reason=escalation["escalation"],
            extra={"signals": diagnosis["signals"]},
        )
        diagnosis_signal: dict[str, Any] = {
            "risk_level": diagnosis["risk_level"],
            "signals": diagnosis["signals"],
            "escalation": escalation["escalation"],
        }
        if "judge_result" in escalation:
            diagnosis_signal["judge_result"] = escalation["judge_result"]
        diagnosis_update: dict[str, Any] = {
            "thinking_mode": diagnosis["thinking_mode"],
            "diagnosis_signal": diagnosis_signal,
            "diagnosis_judge_calls_used": escalation["judge_calls_used"],
        }
    else:
        # Kill switch: no-op defaults, identical to a never-diagnosed session.
        diagnosis_update = {"thinking_mode": None, "diagnosis_signal": None}

    decision_nodes = state.get("decision_nodes") or {}
    # Legacy migration: an older checkpoint may carry note-parsed assumptions/open_questions in the
    # dropped state fields. Fold any that a node does not already cover into
    # decision nodes so a resumed session loses nothing. No-op once the fields are absent (the common
    # case — LangGraph does not surface a channel removed from the schema, so this only fires when the
    # keys are still present in the state dict).
    legacy_assumptions = state.get("assumptions")
    legacy_open_questions = state.get("open_questions")
    migration_update: dict[str, Any] = {}
    if legacy_assumptions or legacy_open_questions:
        decision_nodes, migrated = migrate_legacy_notes(
            decision_nodes,
            legacy_assumptions,
            legacy_open_questions,
            {"turn": state.get("turn_count") or 0, "by": "migration", "technique": "legacy_notes", "source": None},
        )
        if migrated:
            migration_update["decision_nodes"] = decision_nodes
    feedback = dict(state.get("feedback_summary") or {})
    resurfaced = scan_parked_questions(decision_nodes)
    if resurfaced:
        feedback["resurfaced_questions"] = [
            {"id": node["id"], "statement": node["statement"], "blocks": list(node.get("blocks") or [])}
            for node in resurfaced
        ]
    else:
        feedback.pop("resurfaced_questions", None)

    # Ignored-signal escalation: a per-signal counter of consecutive turns the signal has surfaced
    # unaddressed. Incremented while present, dropped (not zeroed) once resolved/dismissed so the key
    # is absent for legacy checkpoints and for sessions that never trip the signal.
    ignored_counts = dict(feedback.get("ignored_counts") or {})
    if resurfaced:
        ignored_counts["resurfaced_questions"] = ignored_counts.get("resurfaced_questions", 0) + 1
    else:
        ignored_counts.pop("resurfaced_questions", None)
    if ignored_counts:
        feedback["ignored_counts"] = ignored_counts
    else:
        feedback.pop("ignored_counts", None)

    update: dict[str, Any] = {
        "feedback_summary": feedback,
        "session_phase": session_phase,
        **diagnosis_update,
        **migration_update,
    }
    if _should_run_completeness_sweep(state):
        gaps = completeness_sweep(decision_nodes, state.get("artifact_type") or "brd")
        updated_nodes, created = add_parked_questions_for_gaps(
            decision_nodes,
            gaps,
            {"turn": state.get("turn_count") or 0, "by": "agent", "technique": "completeness_sweep", "source": None},
        )
        feedback["brd_stable_sweep_done"] = True
        feedback["depth_signal"] = "BRD stable -> can descend to PRD"
        feedback["sweep_gaps"] = list(gaps)
        feedback["created_parked_questions"] = [{"id": node["id"], "statement": node["statement"]} for node in created]
        update["feedback_summary"] = feedback
        if created:
            update["decision_nodes"] = updated_nodes
    return update


async def analyze_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """Analyst turn — orchestration over app.graphs.analysis.* (logic moved there verbatim)."""
    cfg = config["configurable"]
    session_id = uuid.UUID(cfg["thread_id"])
    llm_client = cfg.get("strong_llm_client") or cfg["llm_client"]
    if llm_client is None:
        raise ValueError("LLM provider is not configured. Add an API key in settings.")

    ctx = await load_turn_context(state, config)
    effective_state, coverage = ctx.effective_state, ctx.coverage

    prompt = _build_tool_selection_prompt(effective_state, ctx.artifacts, ctx.draft_body, ctx.previous_draft_body)
    system_prompt = build_system_prompt(effective_state, cfg.get("agent_role"), has_draft=ctx.draft_body is not None)
    available_tools = get_available_tools(effective_state)
    tool_schemas = _build_tool_schemas(available_tools)
    analyzer_messages = _build_analyzer_messages(effective_state, prompt)
    started_at = time.monotonic()
    ai_message, usage = await llm_client.generate(
        messages=analyzer_messages,
        system=system_prompt,
        max_tokens=settings.analyze_max_tokens,
        tools=tool_schemas,
        tool_choice=settings.tool_choice_mode,
    )
    latency_ms = int((time.monotonic() - started_at) * 1000)
    token_usage = annotate_token_usage(
        usage,
        system_prompt=system_prompt or "",
        messages=analyzer_messages,
        tool_schemas=tool_schemas,
        draft_body=ctx.draft_body,
    )

    model_tool_calls, gated_tools, dropped_tools, next_feedback, out_of_phase_tools = gate_model_selection(
        effective_state, ai_message
    )
    # Analytic fields are derived from state, not self-reported by the LLM: locale sticky-from-state
    # (default vi). Drafts of record flow through decision_nodes and write_draft.
    locale = effective_state.get("locale") or "vi"
    analysis_result_base = build_analysis_result_base(
        gated_tools=gated_tools,
        model_tool_calls=model_tool_calls,
        dropped_tools=dropped_tools,
        available_tools=available_tools,
        locale=locale,
        coverage_complete=coverage["coverage_complete"],
        session_phase=effective_state.get("session_phase"),
    )
    run_id, analysis_result, dispatched_tools, dispatched_tool_calls = await record_run_and_dispatch(
        session_factory=cfg["session_factory"],
        session_id=session_id,
        analysis_result_base=analysis_result_base,
        token_usage=token_usage,
        latency_ms=latency_ms,
        gated_tools=gated_tools,
        ai_message=ai_message,
        locale=locale,
    )

    return _analysis_turn_result(
        ctx=ctx,
        analysis_result=analysis_result,
        run_id=run_id,
        locale=locale,
        next_feedback=next_feedback,
        dispatched_tools=dispatched_tools,
        dispatched_tool_calls=dispatched_tool_calls,
        out_of_phase_count=len(out_of_phase_tools),
    )


def _analysis_turn_result(
    *,
    ctx: TurnContext,
    analysis_result: dict[str, Any],
    run_id: str,
    locale: str,
    next_feedback: dict[str, Any],
    dispatched_tools: list[dict[str, Any]],
    dispatched_tool_calls: list[dict[str, Any]],
    out_of_phase_count: int = 0,
) -> dict[str, Any]:
    """Assemble analyze_node's WorkflowState update — moved verbatim from the inline block."""
    effective_state, coverage = ctx.effective_state, ctx.coverage
    # BMAD method profile: workflow_mode is inferred from coverage (no longer LLM-reported);
    # planning_track normalized to quick on miss. Merge so other profile fields persist.
    method_profile = dict(effective_state.get("method_profile") or DEFAULT_METHOD_PROFILE)
    method_profile["current_workflow"] = _infer_workflow_mode(effective_state)
    method_profile["planning_track"] = _normalize_planning_track(method_profile.get("planning_track"))

    next_turn_count = effective_state["turn_count"] + 1
    recent_tool_calls = append_turn_fingerprint(effective_state.get("recent_tool_calls"), dispatched_tools)
    result = {
        "analysis_result": analysis_result,
        "turn_count": next_turn_count,
        "last_agent_run_id": run_id,
        # Locale stays sticky once set so the output language lock holds across turns.
        "locale": locale,
        # Persist the DB-loaded draft body so run_critique can target it next turn.
        "draft_body": ctx.draft_body,
        "turn_context_artifacts": [dict(item) for item in ctx.artifacts],
        "lifecycle_reports": [dict(item) for item in ctx.lifecycle_reports],
        "artifact_history": [dict(item) for item in ctx.artifact_history],
        "method_profile": method_profile,
        # Display/persistence snapshot; recommend_next_workflow re-derives inline to avoid staleness.
        "artifact_chain": _derive_artifact_chain(coverage.get("section_coverage")),
        # Multi-angle (S2): the mode_hint is a one-shot steer. It has already been folded into
        # this turn's prompt, so clear it now — the next turn returns to proactive default.
        "mode_hint": None,
        "feedback_summary": next_feedback,
        "recent_tool_calls": recent_tool_calls,
        "out_of_phase_tool_calls": (effective_state.get("out_of_phase_tool_calls") or 0) + out_of_phase_count,
        **ctx.focus_reset_update,
        **coverage,
    }
    # User-facing text must pass through tools so service persistence/interrupt handling owns delivery.
    # Bedrock Anthropic rejects ":" in replayed tool_use ids; keep ids within ^[a-zA-Z0-9_-]+$.
    if dispatched_tool_calls:
        dispatch_stop_reason = _pending_dispatch_stop_reason(next_turn_count, recent_tool_calls)
        result["messages"] = [_dispatch_ai_message(dispatched_tool_calls)]
        if dispatch_stop_reason:
            result["messages"].extend(_synthetic_tool_results(dispatched_tool_calls, dispatch_stop_reason))
    return result


def _dispatch_ai_message(dispatched_tool_calls: list[dict[str, Any]]) -> AIMessage:
    tool_calls: list[dict[str, Any]] = []
    provider_tool_calls: dict[str, dict[str, Any]] = {}
    for tool_call in dispatched_tool_calls:
        public_tool_call = {
            "id": str(tool_call.get("id") or ""),
            "name": tool_call.get("name") or "",
            "args": dict(tool_call.get("args") or {}),
        }
        tool_calls.append(public_tool_call)
        provider_metadata = tool_call.get("provider_metadata")
        if isinstance(provider_metadata, dict):
            provider_tool_calls[public_tool_call["id"]] = provider_metadata
    additional_kwargs = {"provider_tool_calls": provider_tool_calls} if provider_tool_calls else {}
    return AIMessage(content="", tool_calls=tool_calls, additional_kwargs=additional_kwargs)


def _pending_dispatch_stop_reason(next_turn_count: int, recent_tool_calls: list[str]) -> str | None:
    """Return why route_node will END instead of dispatching the just-emitted tool calls."""
    if next_turn_count >= settings.max_agent_turns:
        return "max_agent_turns"
    if _has_repeated_tool_calls(recent_tool_calls):
        return "repeated_tool_calls"
    return None


def _synthetic_tool_results(tool_calls: list[dict[str, Any]], reason: str) -> list[ToolMessage]:
    """Close suppressed tool calls so provider replay never sees dangling tool_use blocks."""
    return [
        ToolMessage(
            content=f"Tool call was not executed because the analysis loop stopped: {reason}.",
            tool_call_id=str(tool_call.get("id") or ""),
            status="error",
        )
        for tool_call in tool_calls
    ]


async def summarize_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    if route_before_analyze(state) != "summarize":
        return {"conversation_summary": state.get("conversation_summary", "")}

    cfg = config["configurable"]
    llm_client = cfg["llm_client"]
    if llm_client is None:
        raise ValueError("LLM provider is not configured. Add an API key in settings.")

    prompt = _build_summary_prompt(state)
    started_at = time.monotonic()
    try:
        result, _usage = await llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            system=SUMMARY_SYSTEM,
            max_tokens=1000,
            response_format=SUMMARY_SCHEMA,
        )
    except ValueError:
        # The summary is an optional running aid (prompt context only), so a non-conforming LLM
        # response must not crash the turn — keep the prior summary and continue to analyze.
        return {"conversation_summary": state.get("conversation_summary", "")}
    logger.debug("node=summarize latency_ms=%d", int((time.monotonic() - started_at) * 1000))
    if isinstance(result, dict):
        summary = str(result.get("summary", "")).strip()
    else:
        summary = str(result or "").strip()
    return {"conversation_summary": summary or state.get("conversation_summary", "")}


def route_before_analyze(state: WorkflowState) -> str:
    # "orchestrator" is the analyst-loop entry (orchestrator -> analyze); the label matches its target.
    messages = state.get("messages") or []
    if not messages or not _is_human_turn(messages[-1]):
        return "orchestrator"
    trigger = settings.summary_trigger_every
    human_turns_after_initial = max(0, sum(1 for message in messages if _is_human_turn(message)) - 1)
    if trigger > 0 and human_turns_after_initial > 0 and human_turns_after_initial % trigger == 0:
        return "summarize"
    return "orchestrator"


def route_node(state: WorkflowState) -> str:
    if state["turn_count"] >= settings.max_agent_turns:
        return END
    # P9: the model repeating the same (name + args) tool call N times in a row is stuck, not making
    # progress — exit the analyze/tools cycle via the same path the turn-count ceiling uses rather than
    # waiting for max_agent_turns (which stays 30; this is an earlier, narrower exit condition).
    if _has_repeated_tool_calls(state.get("recent_tool_calls") or []):
        return END
    # analyze_node emitted an AIMessage; dispatch on its tool_calls. No tool_calls means the loop
    # has nothing to run this turn -> finish. The finalize hard-gate and the coverage signal live in
    # get_available_tools / _build_section_coverage_hint, not here.
    return "tools" if _last_message_has_tool_calls(state) else END




def _last_message_has_tool_calls(state: WorkflowState) -> bool:
    """Whether the most recent message carries tool_calls (a tool-loop dispatch signal)."""
    messages = state.get("messages") or []
    if not messages:
        return False
    return bool(getattr(messages[-1], "tool_calls", None))


def _build_summary_prompt(state: WorkflowState) -> str:
    current_summary = (state.get("conversation_summary") or "").strip() or "(none yet)"
    recent_messages = (
        "\n".join(
            f"{role}: {content}"
            for role, content in (
                _msg_role_content(m) for m in (state.get("messages") or [])[-settings.summary_trigger_every :]
            )
        )
        or "(no new conversation)"
    )

    return (
        "Update the running summary for the product requirements conversation.\n\n"
        f"CURRENT SUMMARY:\n{current_summary}\n\n"
        f"NEW CONVERSATION:\n{recent_messages}\n\n"
        "Return exactly the following four sections:\n"
        "Confirmed requirements\n"
        "Constraints - DO NOT paraphrase\n"
        "Unclear gaps\n"
        "Agreed decisions\n\n"
        "In the constraints section, keep all numbers, deadlines, proper names, and scope limits verbatim."
    )
