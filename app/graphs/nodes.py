import difflib
import hashlib
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.types import interrupt
from sqlalchemy import exists, select

from app.config import settings
from app.documents.registry import children_of, get_config, output_contract, status_score
from app.graphs.decision_graph import (
    add_parked_questions_for_gaps,
    completeness_sweep,
    is_brd_stable,
    scan_parked_questions,
)
from app.graphs.policy import ancestor_types
from app.graphs.state import DEFAULT_METHOD_PROFILE, WorkflowState
from app.graphs.tools import read_artifacts, read_current_body
from app.instructions import get_instruction
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
)
from app.services.document_service import DocumentService

# Native tool calling replaces the old JSON tool-selection schema: analyze_node binds the available
# tool schemas to the provider API (see _build_tool_schemas) and the model returns native tool_calls.
# Analytic fields (locale, workflow_mode) are derived from the picked tool + state.

# Valid planning tracks; _normalize_planning_track falls back to quick on miss.
_PLANNING_TRACKS = {"quick", "standard", "enterprise"}

# Tool impls now reject empty required args via a ToolMessage error, so this table no longer drives
# dispatch; it survives only as the required-arg contract that the intent_gate eval and unit tests assert.
_TOOL_REQUIRED_ARGS = {
    "write_draft": ["body"],
    "finalize": ["summary"],
    "run_critique": ["mode"],
    "confirm_intent": ["summary"],
}
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

# Injected tool params are runtime wiring (LangGraph fills them), never LLM-visible args — strip
# them from the schema passed to the provider so the model only sees real arguments.
_INJECTED_TOOL_PARAMS = frozenset({"state", "config", "tool_call_id"})

# Number of consecutive identical (name + args) tool-call fingerprints that trigger route_node's
# early exit. 3 (not 1 or 2) is conservative enough to tolerate a legitimate one-off repeat (e.g. the
# model re-issuing the same idempotent call after a transient tool error) while still catching a model
# stuck looping well before the 30-turn max_agent_turns ceiling.
_REPEATED_TOOL_CALL_EXIT_THRESHOLD = 3
# Only the last N fingerprints are ever needed to test the threshold; bounding the list keeps the
# checkpointed WorkflowState field small regardless of session length.
_RECENT_TOOL_CALLS_MAXLEN = _REPEATED_TOOL_CALL_EXIT_THRESHOLD


def _strip_injected_params(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove injected params (state, config, tool_call_id) from a tool's JSON-Schema properties."""
    props = {k: v for k, v in (schema.get("properties") or {}).items() if k not in _INJECTED_TOOL_PARAMS}
    required = [r for r in (schema.get("required") or []) if r not in _INJECTED_TOOL_PARAMS]
    return {**schema, "properties": props, "required": required}


def _build_tool_schemas(tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Convert state-valid tools into provider-agnostic schemas for generate(tools=...)."""
    schemas: list[dict[str, Any]] = []
    for t in tools:
        raw = t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}}
        params = _strip_injected_params(raw)
        schemas.append({"name": t.name, "description": t.description or "", "parameters": params})
    return schemas


def _model_tool_calls(ai_message: AIMessage) -> list[dict[str, Any]]:
    return [
        {
            "id": tc.get("id"),
            "name": tc.get("name") or "",
            "args": dict(tc.get("args") or {}),
        }
        for tc in (getattr(ai_message, "tool_calls", None) or [])
    ]


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


# P10: the LLM API returns only aggregate input/output/total token counts, never a per-component
# split, so per-component figures here are a proxy, not an exact count. We use a uniform
# chars-to-tokens ratio (4 chars/token — the commonly cited average for English/mixed-language text;
# no tokenizer is exposed by our LLM client interface, and adding a tiktoken dependency purely for an
# estimate would be disproportionate). Applied identically to every component, so the four figures are
# comparable to each other even though none of them is individually precise.
_CHARS_PER_TOKEN_ESTIMATE = 4


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


def _audit_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = dict(tool_call.get("args") or {})
    audited = {
        "name": tool_call.get("name") or "",
        "args": {key: _audit_arg_value(str(key), value) for key, value in args.items()},
    }
    if tool_call.get("id") is not None:
        audited["id"] = tool_call.get("id")
    return audited


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

    brief_sections = ["vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities"]
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


async def _document_coverage(
    *,
    db,
    project_id: uuid.UUID,
    artifact_type: str,
    focused_artifact_id: uuid.UUID | None,
) -> dict[str, Any]:
    return await DocumentService(db).document_coverage(
        project_id=project_id,
        artifact_type=artifact_type,
        focused_artifact_id=focused_artifact_id,
    )


async def triage_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """Entry node: classify a fresh turn so a greeting/smalltalk skips the full analyst pass.

    One cheap LLM call (the standard client, not the strong one) decides ``converse`` vs ``work``,
    detects the locale, and — for a conversational turn — drafts the reply in the same call.
    ``work`` falls straight through to analyze_node. The classifier runs once per fresh invocation;
    on resume LangGraph re-enters the interrupted node (converse/tools), so it never re-runs.
    """
    cfg = config["configurable"]
    llm_client = cfg["llm_client"]
    if llm_client is None:
        raise ValueError("LLM provider is not configured. Add an API key in settings.")

    last_user = ""
    for m in reversed(state.get("messages") or []):
        role, content = _msg_role_content(m)
        if role == "user":
            last_user = content
            break

    prompt = (
        "Classify the user's message.\n\n"
        f"Message: {last_user!r}\n\n"
        "turn_type: 'converse' if only greeting, thanks, small talk, or off-topic; "
        "'work' if is a request to analyze, clarify, or create an artifact.\n"
        "locale: 'vi' if Vietnamese, 'en' if English.\n"
        "If turn_type='converse', set 'reply' to a short, friendly sentence in the user's exact language "
        "- greet back, briefly say what you can help with, and invite them to share what they want to build."
    )
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
    reported = result if isinstance(result, dict) else {}
    turn_type = reported.get("turn_type") or "work"
    locale = state.get("locale") or reported.get("locale")
    reply = reported.get("reply") if turn_type == "converse" else None
    return {"turn_type": turn_type, "locale": locale, "triage_reply": reply}


def route_after_triage(state: WorkflowState) -> str:
    """Conversational turns peel off to converse_node; everything else goes to the analyst."""
    return "converse" if state.get("turn_type") == "converse" else "analyze"


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

# Below this coverage ratio a section counts as "low coverage" for the diagnosis conjunction.
# Named constant (not a soft score) so a single weak signal can't silently escalate every section.
_LOW_COVERAGE_RATIO = 0.34
_COVERAGE_SCORES = {"filled": 1.0, "needs_review": 0.5, "partial": 0.5, "missing": 0.0}


def _diagnose_section(state: WorkflowState) -> dict[str, Any]:
    """Cheap, LLM-free risk/ambiguity diagnosis for the current section (Phase 1 of the adaptive
    analysis loop). Never calls an LLM — pure function of state already available in-turn.

    Cold start (no coverage data yet, e.g. turn 1): defaults to low risk / "structuring" rather
    than erroring or guessing high risk. This is intentional, not an overlooked edge case —
    adaptivity kicks in from turn 2 once coverage/quality data exists.
    """
    coverage = state.get("section_coverage")
    if not coverage:
        return {"risk_level": "low", "signals": [], "thinking_mode": "structuring"}

    ratios = [_COVERAGE_SCORES.get(value, 0.0) for value in coverage.values()]
    coverage_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    low_coverage = coverage_ratio < _LOW_COVERAGE_RATIO

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
    """Escalate a heuristic high-risk diagnosis to an LLM judge call, budget-gated (Phase 4).

    Low-risk sections never reach the judge. High-risk sections spend one judge call per turn up
    to DIAGNOSIS_JUDGE_CALLS_MAX; once the budget is exhausted the call is skipped with a distinct
    "escalation_skipped_budget" signal rather than silently degrading like "not_needed".
    """
    calls_used = state.get("diagnosis_judge_calls_used") or 0
    if diagnosis["risk_level"] != "high":
        return {"escalation": "not_needed", "judge_calls_used": calls_used}

    from app.graphs.agent_tools import DIAGNOSIS_JUDGE_CALLS_MAX

    if calls_used >= DIAGNOSIS_JUDGE_CALLS_MAX:
        return {"escalation": "escalation_skipped_budget", "judge_calls_used": calls_used}

    from app.graphs.critique import _invoke_judge

    llm_client = _diagnosis_llm_client(config)
    judge_result = await _invoke_judge(state.get("draft_body") or "", "risk_review", llm_client)
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
    """Pre-analyst orchestration: resurface parked blockers and run completeness sweep when triggered."""
    if settings.enable_adaptive_diagnosis:
        diagnosis = _diagnose_section(state)
        escalation = await _apply_judge_escalation(diagnosis, state, config)
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
    feedback = dict(state.get("feedback_summary") or {})
    resurfaced = scan_parked_questions(decision_nodes)
    if resurfaced:
        feedback["resurfaced_questions"] = [
            {"id": node["id"], "statement": node["statement"], "blocks": list(node.get("blocks") or [])}
            for node in resurfaced
        ]
    else:
        feedback.pop("resurfaced_questions", None)

    update: dict[str, Any] = {"feedback_summary": feedback, **diagnosis_update}
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
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    project_id = uuid.UUID(cfg["project_id"])
    llm_client = cfg.get("strong_llm_client") or cfg["llm_client"]
    if llm_client is None:
        raise ValueError("LLM provider is not configured. Add an API key in settings.")

    effective_state: WorkflowState = state
    focus_reset_update: dict[str, Any] = {}

    # Context for the analyst = artifacts of the current type (avoid duplicates)
    # plus its full transitive ancestry — the upstream sources it must derive
    # from (e.g. a `story` traces back through `epic` ... up to `intent`). Using
    # the closure (not just direct parents) makes provenance complete for every
    # type regardless of how ARTIFACT_PREDECESSORS declares it; dedup keeps it
    # token-light since read_artifacts returns title-only rows (no body).
    artifact_type = state["artifact_type"]
    context_types = [artifact_type, *ancestor_types(artifact_type)]
    async with session_factory() as db:
        db_focused_artifact_id = (
            await db.execute(select(AgentSession.focused_artifact_id).where(AgentSession.id == session_id))
        ).scalar_one_or_none()
        state_focused_artifact_id = (
            uuid.UUID(str(state["focused_artifact_id"])) if state.get("focused_artifact_id") else None
        )
        if db_focused_artifact_id != state_focused_artifact_id:
            focus_reset_update = {
                "focused_artifact_id": (str(db_focused_artifact_id) if db_focused_artifact_id is not None else None),
                "critique_rounds": 0,
                "quality_report": None,
                "last_critiqued_draft_hash": None,
                "candidate_readiness": None,
                "feedback_summary": None,
                "verification_status": None,
                "latest_checked_revision": None,
            }
            effective_state = {**state, **focus_reset_update}

        # Batched into one query for the whole ancestor-type chain instead of one round trip per type.
        artifacts = await read_artifacts(
            db=db,
            project_id=project_id,
            artifact_type=context_types,
            context={"workflow_area": effective_state["workflow_area"]},
        )
        # Load the current draft body for this artifact_type so the analyst can mine
        # the delta instead of re-asking what the draft already records (M7/M8).
        draft = await read_current_body(
            db=db,
            project_id=project_id,
            artifact_type=artifact_type,
            artifact_id=db_focused_artifact_id,
        )
        coverage = await _document_coverage(
            db=db,
            project_id=project_id,
            artifact_type=artifact_type,
            focused_artifact_id=db_focused_artifact_id,
        )
        previous_accepted = sum(
            1 for value in (effective_state.get("section_coverage") or {}).values() if value == "filled"
        )
        current_accepted = sum(1 for value in (coverage.get("section_coverage") or {}).values() if value == "filled")
        if (
            coverage["coverage_complete"]
            or current_accepted > previous_accepted
            or effective_state.get("section_coverage") is None
        ):
            coverage["section_coverage_stall_count"] = 0
        else:
            coverage["section_coverage_stall_count"] = (effective_state.get("section_coverage_stall_count") or 0) + 1
        effective_state = {**effective_state, **coverage}
    # Captured before the freshly-read draft below shadows it: state["draft_body"] on entry is
    # exactly what this node persisted as "draft_body" last turn (see the result dict further
    # down), so it doubles as the "last sent to the model" snapshot the diff needs at zero cost.
    previous_draft_body = state.get("draft_body")
    draft_body = draft["body"] if draft else None

    prompt = _build_tool_selection_prompt(
        effective_state, artifacts, draft_body, previous_draft_body, str(session_id)
    )
    system_prompt = get_instruction(
        artifact_type=effective_state["artifact_type"],
        workflow_area=effective_state["workflow_area"],
        agent_role=cfg.get("agent_role"),
        context={"has_draft": draft_body is not None},
    )
    # Artifact-type shape (taxonomy chain + section-coverage contract) belongs with the static policy
    # in L1, not the per-turn payload — appended last so the static prefix stays cache-friendly.
    system_prompt = (system_prompt or "") + _build_artifact_contract_block(effective_state)
    system_prompt = system_prompt + _build_thinking_mode_block(effective_state)
    system_prompt = system_prompt + _build_stuck_escalation_block(effective_state)
    from app.graphs.agent_tools import get_available_tools

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
    # P10: additive per-component estimate, alongside (never replacing) whatever keys the client's
    # usage dict already carries.
    token_usage = dict(usage) if isinstance(usage, dict) else usage
    if isinstance(token_usage, dict):
        token_usage["by_component"] = _estimate_token_breakdown(
            system_prompt=system_prompt or "",
            messages=analyzer_messages,
            tool_schemas=tool_schemas,
            draft_body=draft_body,
        )

    # Post-LLM gate keeps only the solo invariant; availability is enforced by the state-driven tool surface.
    model_tool_calls = _model_tool_calls(ai_message)
    raw_tools = [{"name": tc["name"], "args": dict(tc["args"])} for tc in model_tool_calls]
    gated_tools = _gate_selected_tools(effective_state, raw_tools)
    dropped_tools = _dropped_tool_names(raw_tools, gated_tools)
    # One-shot: clear the previous turn's notice (already rendered into this turn's prompt) and stage
    # this turn's drops for the next prompt — so the model sees its dropped tools exactly once.
    next_feedback = dict(effective_state.get("feedback_summary") or {})
    next_feedback.pop("dropped_tools", None)
    if dropped_tools:
        next_feedback["dropped_tools"] = dropped_tools

    # Analytic fields are derived from state, not self-reported by the LLM: locale sticky-from-state
    # (default vi). Drafts of record flow through decision_nodes and write_draft.
    locale = effective_state.get("locale") or "vi"

    analysis_result_base: dict[str, Any] = {
        "tools": [_audit_tool_call(item) for item in gated_tools],
        "model_tool_calls": [_audit_tool_call(item) for item in model_tool_calls],
        "raw_model_tool_calls": [_audit_tool_call(item) for item in model_tool_calls],
        "dropped_tool_calls": dropped_tools,
        "available_tools": [tool.name for tool in available_tools],
        "locale": locale,
        "coverage_complete": coverage["coverage_complete"],
    }

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

    # BMAD method profile: workflow_mode is inferred from coverage (no longer LLM-reported);
    # planning_track normalized to quick on miss. Merge so other profile fields persist.
    method_profile = dict(effective_state.get("method_profile") or DEFAULT_METHOD_PROFILE)
    method_profile["current_workflow"] = _infer_workflow_mode(effective_state)
    method_profile["planning_track"] = _normalize_planning_track(method_profile.get("planning_track"))

    # P9: append one fingerprint per turn (not per dispatched call) summarizing this turn's whole
    # dispatched-tool batch, so route_node's threshold means "N consecutive turns", not "N dispatched
    # calls" — a turn that dispatches several identical tool calls at once must not be mistaken for a
    # multi-turn stuck loop.
    recent_tool_calls = list(effective_state.get("recent_tool_calls") or [])
    if dispatched_tools:
        turn_fingerprint = "|".join(
            sorted(_tool_call_fingerprint(item["name"], item["args"]) for item in dispatched_tools)
        )
        recent_tool_calls.append(turn_fingerprint)
    recent_tool_calls = recent_tool_calls[-_RECENT_TOOL_CALLS_MAXLEN:]

    result = {
        "analysis_result": analysis_result,
        "turn_count": effective_state["turn_count"] + 1,
        "last_agent_run_id": run_id,
        # Locale stays sticky once set so the output language lock holds across turns.
        "locale": locale,
        # Persist the DB-loaded draft body so run_critique can target it next turn.
        "draft_body": draft_body,
        "method_profile": method_profile,
        # Display/persistence snapshot; recommend_next_workflow re-derives inline to avoid staleness.
        "artifact_chain": _derive_artifact_chain(coverage.get("section_coverage")),
        # Multi-angle (S2): the mode_hint is a one-shot steer. It has already been folded into
        # this turn's prompt, so clear it now — the next turn returns to proactive default.
        "mode_hint": None,
        "feedback_summary": next_feedback,
        "recent_tool_calls": recent_tool_calls,
        **focus_reset_update,
        **coverage,
    }
    # User-facing text must pass through tools so service persistence/interrupt handling owns delivery.
    # Bedrock Anthropic rejects ":" in replayed tool_use ids; keep ids within ^[a-zA-Z0-9_-]+$.
    if dispatched_tool_calls:
        result["messages"] = [AIMessage(content="", tool_calls=dispatched_tool_calls)]
    else:
        result["messages"] = [AIMessage(content="")]
    return result


def _response_message_incomplete(message: Any) -> bool:
    text = str(message or "").strip()
    return not text or text.endswith(":")


async def summarize_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    if route_before_analyze(state) != "summarize":
        return {"conversation_summary": state.get("conversation_summary", "")}

    cfg = config["configurable"]
    llm_client = cfg["llm_client"]
    if llm_client is None:
        raise ValueError("LLM provider is not configured. Add an API key in settings.")

    prompt = _build_summary_prompt(state)
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
    if isinstance(result, dict):
        summary = str(result.get("summary", "")).strip()
    else:
        summary = str(result or "").strip()
    return {"conversation_summary": summary or state.get("conversation_summary", "")}


def route_before_analyze(state: WorkflowState) -> str:
    messages = state.get("messages") or []
    if not messages or not _is_human_turn(messages[-1]):
        return "analyze"
    trigger = settings.summary_trigger_every
    human_turns_after_initial = max(0, sum(1 for message in messages if _is_human_turn(message)) - 1)
    if trigger > 0 and human_turns_after_initial > 0 and human_turns_after_initial % trigger == 0:
        return "summarize"
    return "analyze"


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


async def _save_and_interrupt_ask(
    state: WorkflowState,
    config: RunnableConfig,
    content: str,
    *,
    run_id,
    kind: str = "question",
    mode: str | None = None,
    interrupt_kind: str = "ask_human",
) -> str:
    """Persist one agent turn (idempotently), mark the session, then interrupt.

    Shared by ask_user and respond. interrupt_kind controls both DB fields:
    - "ask_human"      → status=WAITING_FOR_HUMAN, interrupt_type=ASK_HUMAN (default, approval flow)
    - "stream_response" → status=ACTIVE, interrupt_type=STREAM_RESPONSE (conversational Q&A)

    Keying the idempotency guard on run_id makes an HTTP-resume (which re-executes the tool body
    from the top) skip the duplicate insert (R1).
    """
    _INTERRUPT_KIND_MAP = {
        "ask_human": (AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionInterruptType.ASK_HUMAN),
        "stream_response": (AgentSessionStatus.ACTIVE, AgentSessionInterruptType.STREAM_RESPONSE),
    }
    session_status, session_interrupt_type = _INTERRUPT_KIND_MAP.get(
        interrupt_kind,
        _INTERRUPT_KIND_MAP["ask_human"],
    )

    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    locale = state.get("locale") or "vi"

    payload: dict[str, Any] = {"kind": kind, "locale": locale, "options": [], "blocks": [], "run_id": run_id}
    if mode:
        payload["mode"] = mode

    async with session_factory() as db:
        already_saved = await _agent_message_already_saved(db, session_id, run_id, content)
        if not already_saved:
            db.add(
                AgentMessage(
                    session_id=session_id,
                    role=AgentMessageRole.AGENT,
                    content=content,
                    payload=payload,
                )
            )
        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        session_row.status = session_status
        session_row.interrupt_type = session_interrupt_type
        await db.commit()

    interrupt_payload = {"type": "respond" if kind == "assessment" else "ask_human", "message": content}
    if mode:
        interrupt_payload["mode"] = mode
    user_response = interrupt(interrupt_payload)
    return user_response.get("content", "") if isinstance(user_response, dict) else str(user_response or "")


async def _agent_message_already_saved(db, session_id, run_id, content) -> bool:
    """Idempotency guard for the ask_user tool: save one agent message then interrupt.

    Keyed on the ToolCall.id (stored in payload.run_id) so it stays correct when the content varies
    across resumes — e.g. a different acknowledgment. Falls back to content match when no run_id is
    available, in which case the caller must pass a non-empty content for the guard to be meaningful.
    """
    if run_id:
        condition = AgentMessage.payload["run_id"].as_string() == str(run_id)
    else:
        condition = AgentMessage.content == content
    return bool(
        (
            await db.execute(
                select(
                    exists().where(
                        AgentMessage.session_id == session_id,
                        AgentMessage.role == AgentMessageRole.AGENT,
                        condition,
                    )
                )
            )
        ).scalar()
    )


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


# P7: cross-turn cache for the rendered decision-view block, keyed by (session_id, content
# fingerprint). The fingerprint is an md5 of the serialized decision_nodes, so ANY change to the
# graph produces a different key — invalidation is automatic and foolproof, no separate "invalidate
# on mutation" bookkeeping is needed. In-process, module-level dict only: it does NOT survive process
# restarts and does NOT help across multiple worker processes; it only skips redundant re-renders for
# a session that stays warm within one process's lifetime. Bounded via simple FIFO eviction so a
# long-running process can't grow this unboundedly across many sessions.
_DECISION_VIEW_CACHE: dict[tuple[str, str], str] = {}
_DECISION_VIEW_CACHE_MAX_ENTRIES = 512


def _decision_nodes_fingerprint(decision_nodes: dict[str, Any]) -> str:
    import json

    return hashlib.md5(json.dumps(decision_nodes, sort_keys=True).encode("utf-8")).hexdigest()


def _build_decision_view_block(state: WorkflowState, session_id: str | None = None) -> str:
    """Rendered decision-graph view shown as the live draft target.

    Cross-turn cached per session_id when provided (P7) — render_view is skipped when this session's
    decision_nodes are byte-identical to the last time this block was rendered for it. session_id is
    optional so callers without a session context (e.g. unit tests) still get correct, uncached output.
    """
    decision_nodes = state.get("decision_nodes") or {}
    if not decision_nodes:
        return ""

    cache_key = (session_id, _decision_nodes_fingerprint(decision_nodes)) if session_id else None
    if cache_key is not None and cache_key in _DECISION_VIEW_CACHE:
        return _DECISION_VIEW_CACHE[cache_key]

    from app.graphs.decision_graph import render_view

    view = render_view(decision_nodes, state.get("artifact_type") or "brd").strip()
    block = f"\n\nDRAFT IN PROGRESS (incrementally updated - reflects clarified points):\n{view}" if view else ""

    if cache_key is not None:
        if len(_DECISION_VIEW_CACHE) >= _DECISION_VIEW_CACHE_MAX_ENTRIES:
            _DECISION_VIEW_CACHE.pop(next(iter(_DECISION_VIEW_CACHE)))
        _DECISION_VIEW_CACHE[cache_key] = block
    return block


def _missing_required_headings(artifact_type: str, body: str) -> list[str]:
    try:
        contract = output_contract(artifact_type)
    except ValueError:
        return []
    return [heading for heading in contract.required_headings if heading not in body]


def _decision_view_can_hide_draft(state: WorkflowState, decision_view_block: str, draft_body: str | None) -> bool:
    if not decision_view_block or not settings.decision_graph_enabled:
        return False
    if not draft_body:
        return True
    return not _missing_required_headings(state.get("artifact_type") or "brd", decision_view_block)


def _last_message_has_tool_calls(state: WorkflowState) -> bool:
    """Whether the most recent message carries tool_calls (a tool-loop dispatch signal)."""
    messages = state.get("messages") or []
    if not messages:
        return False
    return bool(getattr(messages[-1], "tool_calls", None))


def _log_tool_error(code: str, tool_name: str, message: str) -> None:
    """Emit a tool-control error in a grep-friendly format for eval/logs."""
    import logging

    logging.getLogger(__name__).info(
        "tool-error code=%s tool=%s message=%s",
        code,
        tool_name,
        message,
    )


# Tools that call interrupt() — they must always run solo (no composite dispatch).
# DB-writing tools (write_draft, finalize) are also in this set: they interrupt and must not
# be paired with another tool in the same turn to preserve idempotency invariants.
_INTERRUPT_BEARING_TOOLS: frozenset[str] = frozenset(
    {
        "ask_user",
        "respond",
        "write_draft",
        "finalize",
        "confirm_intent",
    }
)

# Silent scratchpad notes: no interrupt, no DB write, pure state append (assumptions/risks/
# open_questions/key_facts). They may ride along with an interrupt-bearing tool because the ToolNode
# discards their partial update when the interrupt fires and re-applies it exactly once on resume —
# so the model can record what it learned in the SAME turn it asks a question, instead of having the
# note dropped by solo enforcement (the only key_facts populator, which starved the anti-re-ask block).
_SIDE_EFFECT_FREE_NOTE_TOOLS: frozenset[str] = frozenset({"critique_note", "explore_note"})


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
    """Enforce the ToolNode safety invariant without picking a tool on the model's behalf.

    The remaining gate is solo enforcement for interrupt-bearing tools: keep the first interrupt plus
    side-effect-free notes, drop the rest. Tools decide unavailable/missing-arg via a tool_result error.

    ``_state`` is part of the gate contract (callers pass it) but the current solo-enforcement rule
    needs only the requested set; it is intentionally unused.
    """
    # Normalize to a stable {name, args} shape; the model's chosen tools are never substituted.
    validated = [{"name": item.get("name") or "", "args": dict(item.get("args") or {})} for item in requested]

    # Solo enforcement: at most one interrupt-bearing tool per turn (two interrupts in a node is
    # unsafe). When one is present, keep it plus any side-effect-free notes (so their structured facts
    # persist this turn) and drop everything else, preserving original order.
    if any(item["name"] in _INTERRUPT_BEARING_TOOLS for item in validated):
        kept: list[dict] = []
        seen_interrupt = False
        for item in validated:
            name = item["name"]
            if name in _INTERRUPT_BEARING_TOOLS:
                if not seen_interrupt:
                    kept.append(item)
                    seen_interrupt = True
                else:
                    _log_tool_error(
                        "dropped_interrupt_tool",
                        name,
                        "dropped: an interrupt-bearing tool was already selected this turn",
                    )
            elif name in _SIDE_EFFECT_FREE_NOTE_TOOLS:
                kept.append(item)
            else:
                _log_tool_error(
                    "dropped_with_interrupt_tool",
                    name,
                    "dropped: paired with an interrupt-bearing tool",
                )
        return kept

    return validated


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
    from app.graphs.agent_tools import get_available_tools

    artifact_context = (
        "\n".join(f"- [{a['type']}] {a['title']} (id={a['id']})" for a in artifacts) or "(no artifacts yet)"
    )

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
    decision_view_block = _build_decision_view_block(state, session_id)
    draft_block = (
        ""
        if _decision_view_can_hide_draft(state, decision_view_block, draft_body)
        else _build_draft_delta_block(state, draft_body, previous_draft_body)
    )
    feedback_block = _build_feedback_control_block(state)
    key_facts_block = _build_key_facts_block(state)
    # Taxonomy chain + section-coverage contract are no longer here — they moved to the system prompt
    # (see _build_artifact_contract_block) so the per-turn payload stays small next to the conversation.
    return (
        f"You are the analyst for artifact type: {state['artifact_type']}.\n\n"
        f"Current context:\n{artifact_context}\n\n"
        f"{summary_block}"
        f"Tools available this turn: {tool_menu}.\n"
        "Choose 1-3 suitable tools and fill each tool's fields according to the system prompt policy."
        f"{_build_section_coverage_hint(state)}"
        f"{key_facts_block}"
        f"{feedback_block}"
        f"{draft_block}"
        f"{decision_view_block}"
        f"{_build_mode_hint_directive(state)}"
        f"{language_lock}"
    )


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

    feedback_summary = state.get("feedback_summary") or {}
    resurfaced = feedback_summary.get("resurfaced_questions") or []
    if resurfaced:
        rendered = "; ".join(f"{item.get('id')}: {item.get('statement')}" for item in resurfaced[:3])
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
    dropped = feedback_summary.get("dropped_tools") or []
    if dropped:
        parts.append(
            "- skipped last turn (not run because it was bundled with an interrupting tool and must run separately); "
            f"call it again in a separate turn if still needed: {_compact_list(dropped)}"
        )

    if not parts:
        return ""
    return (
        "\n\nFEEDBACK CONTROL:\n"
        "- the signals below are orchestration priorities; choose suitable tools and order without ignoring them.\n"
        + "\n".join(parts)
    )


def _compact_list(values: list[Any], limit: int = 3) -> str:
    rendered = [str(value) for value in values[:limit] if str(value).strip()]
    if len(values) > limit:
        rendered.append(f"... (+{len(values) - limit})")
    return "; ".join(rendered)


# Technique hints per thinking mode. Widened after Phase 3 landed (elicit_tool's registry now
# includes pre_mortem/challenge_assumptions) -- every name here must exist in ELICIT_TECHNIQUES.
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
    byte-identical to pre-plan behavior. get_instruction()'s cached, role-keyed assembly is never
    touched -- this is a per-turn suffix, same mechanism as _build_artifact_contract_block.
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
    """True one repeat before route_node's hard-stop threshold fires (Phase 4 safety valve).

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

    Returns "" when adaptive diagnosis is disabled or the model is not near-stuck, keeping the
    fast path byte-identical to pre-plan behavior.
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
