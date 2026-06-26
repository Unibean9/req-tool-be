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
# Analytic fields (active_mode, locale, workflow_mode) are derived from the picked tool + state.

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

# analyze_node derives active_mode from the picked (primary) tool, so the operating angle no longer
# depends on the model self-reporting it (the JSON-shim era field). Values are in the spec §7.1
# vocabulary (explore_note -> structuring, not discovery). respond falls back to critique; the
# discovery baseline covers any tool not listed.
_TOOL_ACTIVE_MODE: dict[str, str] = {
    "critique_note": "critique",
    "explore_note": "structuring",
    "ask_user": "discovery",
    "confirm_intent": "discovery",
    "write_draft": "structuring",
    "finalize": "finalization",
    "run_critique": "critique",
    "respond": "critique",
    "run_readiness_check": "finalization",
    "recommend_next_workflow": "finalization",
}

# Injected tool params are runtime wiring (LangGraph fills them), never LLM-visible args — strip
# them from the schema passed to the provider so the model only sees real arguments.
_INJECTED_TOOL_PARAMS = frozenset({"state", "config", "tool_call_id"})


def _strip_injected_params(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove injected params (state, config, tool_call_id) from a tool's JSON-Schema properties."""
    props = {k: v for k, v in (schema.get("properties") or {}).items() if k not in _INJECTED_TOOL_PARAMS}
    required = [r for r in (schema.get("required") or []) if r not in _INJECTED_TOOL_PARAMS]
    return {**schema, "properties": props, "required": required}


def _build_tool_schemas(tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Convert LangGraph tool objects to the provider-agnostic schema list for generate(tools=...).

    analyze_node binds the full registry so an out-of-turn tool self-rejects via a tool_result error
    rather than being swapped out by the graph; the per-state menu is surfaced in the prompt instead.
    """
    schemas: list[dict[str, Any]] = []
    for t in tools:
        raw = t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}}
        params = _strip_injected_params(raw)
        schemas.append({"name": t.name, "description": t.description or "", "parameters": params})
    return schemas


def _normalize_planning_track(track: Any) -> str:
    raw = str(track or "").strip().lower()
    return raw if raw in _PLANNING_TRACKS else "quick"


def _derive_artifact_chain(section_coverage: dict[str, str] | None) -> dict[str, str]:
    """BMAD artifact-chain status (missing/partial/complete) derived from 7-section coverage.

    Sole source is section_coverage (Phase 1–2 engine) mapped to 0.0–1.0 scores — no 9-slot data.
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
    "Bạn là trợ lý tóm tắt hội thoại yêu cầu sản phẩm. "
    "Giữ nguyên các ràng buộc quan trọng, đặc biệt số liệu, tên riêng, deadline và phạm vi."
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
    "Bạn là bộ phân loại lượt mở đầu cho một trợ lý phân tích yêu cầu sản phẩm. "
    "Quyết định lượt này là trò chuyện xã giao hay là công việc phân tích yêu cầu."
)

# Locale-templated fallback when the classifier returns no reply text for a converse turn.
_FALLBACK_GREETING = {
    "vi": "Xin chào! Tôi là trợ lý phân tích yêu cầu. Bạn muốn bắt đầu từ đâu?",
    "en": "Hello! I'm your requirements analysis assistant. Where would you like to start?",
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
        raise ValueError("Chưa cấu hình LLM provider. Vui lòng thêm API key trong phần cài đặt.")

    last_user = ""
    for m in reversed(state.get("messages") or []):
        role, content = _msg_role_content(m)
        if role == "user":
            last_user = content
            break

    prompt = (
        "Phân loại tin nhắn của người dùng.\n\n"
        f"Tin nhắn: {last_user!r}\n\n"
        "turn_type: 'converse' nếu chỉ là chào hỏi, cảm ơn, tán gẫu hoặc lạc đề; "
        "'work' nếu là yêu cầu phân tích/làm rõ/tạo artifact.\n"
        "locale: 'vi' nếu tiếng Việt, 'en' nếu tiếng Anh.\n"
        "Nếu turn_type='converse', đặt 'reply' là một câu đáp ngắn, thân thiện ĐÚNG ngôn ngữ "
        "người dùng — chào lại, nói ngắn gọn bạn giúp được gì, và mời họ chia sẻ điều muốn xây."
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
                select(exists().where(
                    AgentMessage.session_id == session_id,
                    AgentMessage.role == AgentMessageRole.AGENT,
                    AgentMessage.content == message,
                ))
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
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == session_id))
        ).scalar_one()
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


async def analyze_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    project_id = uuid.UUID(cfg["project_id"])
    llm_client = cfg.get("strong_llm_client") or cfg["llm_client"]
    if llm_client is None:
        raise ValueError("Chưa cấu hình LLM provider. Vui lòng thêm API key trong phần cài đặt.")

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
            await db.execute(
                select(AgentSession.focused_artifact_id).where(AgentSession.id == session_id)
            )
        ).scalar_one_or_none()
        state_focused_artifact_id = (
            uuid.UUID(str(state["focused_artifact_id"]))
            if state.get("focused_artifact_id")
            else None
        )
        if db_focused_artifact_id != state_focused_artifact_id:
            focus_reset_update = {
                "focused_artifact_id": (
                    str(db_focused_artifact_id)
                    if db_focused_artifact_id is not None
                    else None
                ),
                "critique_rounds": 0,
                "quality_report": None,
                "last_critiqued_draft_hash": None,
                "candidate_readiness": None,
                "feedback_summary": None,
                "verification_status": None,
                "latest_checked_revision": None,
            }
            effective_state = {**state, **focus_reset_update}

        artifacts: list[dict[str, Any]] = []
        for context_type in context_types:
            artifacts.extend(
                await read_artifacts(
                    db=db,
                    project_id=project_id,
                    artifact_type=context_type,
                    context={"workflow_area": effective_state["workflow_area"]},
                )
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
            1 for value in (effective_state.get("section_coverage") or {}).values()
            if value == "filled"
        )
        current_accepted = sum(
            1 for value in (coverage.get("section_coverage") or {}).values()
            if value == "filled"
        )
        if (
            coverage["coverage_complete"]
            or current_accepted > previous_accepted
            or effective_state.get("section_coverage") is None
        ):
            coverage["section_coverage_stall_count"] = 0
        else:
            coverage["section_coverage_stall_count"] = (
                effective_state.get("section_coverage_stall_count") or 0
            ) + 1
        effective_state = {**effective_state, **coverage}
    draft_body = draft["body"] if draft else None

    prompt = _build_tool_selection_prompt(effective_state, artifacts, draft_body)
    system_prompt = get_instruction(
        artifact_type=effective_state["artifact_type"],
        workflow_area=effective_state["workflow_area"],
        agent_role=cfg.get("agent_role"),
        context={"has_draft": draft_body is not None},
    )
    from app.graphs.agent_tools import get_all_analyzer_tools

    tool_schemas = _build_tool_schemas(get_all_analyzer_tools())
    started_at = time.monotonic()
    ai_message, usage = await llm_client.generate(
        messages=_build_analyzer_messages(effective_state, prompt),
        system=system_prompt,
        max_tokens=settings.analyze_max_tokens,
        tools=tool_schemas,
        tool_choice=settings.tool_choice_mode,
    )
    latency_ms = int((time.monotonic() - started_at) * 1000)

    # Post-call gate now keeps only the solo-enforcement invariant; availability and field errors
    # round-trip through tool_result so the model self-corrects next turn.
    raw_tools = [
        {"name": tc.get("name") or "", "args": dict(tc.get("args") or {})}
        for tc in (getattr(ai_message, "tool_calls", None) or [])
    ]
    gated_tools = _gate_selected_tools(effective_state, raw_tools)

    # Analytic fields are derived from the gated primary tool + state, not self-reported by the LLM:
    # active_mode from the tool's operating angle, locale sticky-from-state (default vi), and the
    # draft_update captured from any text the model emitted alongside its tool calls.
    primary_tool = gated_tools[0]["name"] if gated_tools else None
    active_mode = _TOOL_ACTIVE_MODE.get(primary_tool or "", "discovery")
    locale = effective_state.get("locale") or "vi"
    # When the model picks tools it may emit chain-of-thought as content alongside tool_use blocks;
    # that reasoning text is NOT a draft (OQ2: capturing it poisoned working_draft).
    # Only treat content as a draft_update on a terminal turn (no tool_calls), where it is the
    # model's deliberate final message — drafts of record flow through write_draft (→ draft_body).
    has_tool_calls = bool(getattr(ai_message, "tool_calls", None))
    draft_update = None if has_tool_calls else ((getattr(ai_message, "content", None) or "").strip() or None)

    analysis_result: dict[str, Any] = {
        "tools": gated_tools,
        "active_mode": active_mode,
        "locale": locale,
        "draft_update": draft_update,
        "coverage_complete": coverage["coverage_complete"],
    }

    async with session_factory() as db:
        run = AgentRun(
            session_id=session_id,
            analysis_result=analysis_result,
            token_usage=usage,
            latency_ms=latency_ms,
        )
        db.add(run)
        await db.commit()
        run_id = str(run.id)

    # BMAD method profile: workflow_mode is inferred from coverage (no longer LLM-reported);
    # planning_track normalized to quick on miss. Merge so other profile fields persist.
    method_profile = dict(effective_state.get("method_profile") or DEFAULT_METHOD_PROFILE)
    method_profile["current_workflow"] = _infer_workflow_mode(effective_state)
    method_profile["planning_track"] = _normalize_planning_track(method_profile.get("planning_track"))

    result = {
        "analysis_result": analysis_result,
        "turn_count": effective_state["turn_count"] + 1,
        "last_agent_run_id": run_id,
        # Locale stays sticky once set so the output language lock holds across turns.
        "locale": locale,
        # Persist the DB-loaded draft body so run_critique can target it next turn.
        "draft_body": draft_body,
        # Incremental write (C1): carry the running draft forward. A turn with no draft_update keeps
        # the prior draft; the draft only grows (the model is never allowed to reset it mid-session).
        "working_draft": draft_update or effective_state.get("working_draft"),
        "method_profile": method_profile,
        # Display/persistence snapshot; recommend_next_workflow re-derives inline to avoid staleness.
        "artifact_chain": _derive_artifact_chain(coverage.get("section_coverage")),
        # Multi-angle (S2): the mode_hint is a one-shot steer. It has already been folded into
        # this turn's prompt, so clear it now — the next turn returns to proactive default.
        "mode_hint": None,
        **focus_reset_update,
        **coverage,
    }
    # Emit the gated tools as AIMessage(tool_calls=[...]) so route_node dispatches to the ToolNode.
    # tool_call.id = "{run_id}:{i}" — unique per call in the same turn for LangGraph ToolNode
    # uniqueness. DB idempotency keys are handled per-tool by write_draft/(run_id, tool_name).
    # Empty tool_calls means the analyst is done: emit a plain AIMessage carrying its final text.
    if gated_tools:
        tool_calls = []
        for i, item in enumerate(gated_tools):
            tool = item.get("name") or ""
            args = dict(item.get("args") or {})
            # Per-tool post-processing (coercions that must happen at dispatch time).
            if tool == "ask_user" and not str(args.get("message") or "").strip():
                # Prefer the gate-set message (names the gated tool) over the generic fallback.
                gate_msg = str(analysis_result.get("message") or "")
                args["message"] = gate_msg.strip() or _COERCED_ASK_FALLBACK
            if tool == "respond":
                if _response_message_incomplete(args.get("message")):
                    args["message"] = _RESPOND_FALLBACK
                args["mode"] = args.get("mode") or active_mode or "critique"
            tool_calls.append({"id": f"{run_id}:{i}", "name": tool, "args": args})
        result["messages"] = [AIMessage(content="", tool_calls=tool_calls)]
    else:
        result["messages"] = [AIMessage(content=(getattr(ai_message, "content", None) or "").strip())]
    return result


_COERCED_ASK_FALLBACK = (
    "Mình cần làm rõ thêm một ý trước khi có thể viết phần này chắc hơn. "
    "Bạn có thể chia sẻ thêm thông tin quan trọng nhất còn thiếu không?"
)

_RESPOND_FALLBACK = (
    "Dựa trên thông tin hiện có, mình cần phân tích thêm trước khi kết luận. "
    "Bạn bổ sung thêm bối cảnh hoặc xác nhận các điểm chính để mình tiếp tục nhé?"
)


def _response_message_incomplete(message: Any) -> bool:
    text = str(message or "").strip()
    return not text or text.endswith(":")


async def summarize_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    if route_before_analyze(state) != "summarize":
        return {"conversation_summary": state.get("conversation_summary", "")}

    cfg = config["configurable"]
    llm_client = cfg["llm_client"]
    if llm_client is None:
        raise ValueError("Chưa cấu hình LLM provider. Vui lòng thêm API key trong phần cài đặt.")

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
    trigger = settings.summary_trigger_every
    messages_after_initial_user = max(0, len(messages) - 1)
    if trigger > 0 and messages_after_initial_user > 0 and messages_after_initial_user % trigger == 0:
        return "summarize"
    return "analyze"


def route_node(state: WorkflowState) -> str:
    if state["turn_count"] >= settings.max_agent_turns:
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
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == session_id))
        ).scalar_one()
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
                select(exists().where(
                    AgentMessage.session_id == session_id,
                    AgentMessage.role == AgentMessageRole.AGENT,
                    condition,
                ))
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
    """Dựng thread thật cho LLM: hội thoại + tool_use/tool_result + payload phân tích lượt này."""
    messages: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}
    for raw in state.get("messages") or []:
        message = _client_message_from_state(raw, tool_names_by_id)
        if message is not None:
            _append_client_message(messages, message)
    _append_analyzer_prompt(messages, prompt)
    return messages


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
            "content": [{
                "type": "tool_result",
                "tool_use_id": call_id,
                "name": str(name or tool_names_by_id.get(call_id) or "tool"),
                "content": str(content or ""),
            }],
        }

    if role == "assistant" and tool_calls:
        blocks = _text_blocks(content)
        for call in tool_calls:
            call_id = str(call.get("id") or "")
            tool_name = str(call.get("name") or "")
            if call_id and tool_name:
                tool_names_by_id[call_id] = tool_name
            blocks.append({
                "type": "tool_use",
                "id": call_id,
                "name": tool_name,
                "input": dict(call.get("args") or {}),
            })
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
    return f"\n\nDRAFT ĐANG CÓ cho loại '{state['artifact_type']}':\n{draft_body}"


def _build_key_facts_block(state: WorkflowState) -> str:
    """Accumulated key facts: confirmed data points the analyst must not re-ask or contradict."""
    facts = state.get("key_facts") or []
    if not facts:
        return ""
    lines = "\n".join(
        f"- {f['statement']}"
        + (f" (nguồn: {f['source']})" if f.get("source") else "")
        for f in facts
    )
    return f"\n\nKEY FACTS đã xác nhận (không hỏi lại):\n{lines}"


def _build_working_draft_block(state: WorkflowState) -> str:
    """Running-draft block (C1): the in-session draft accumulated across turns, newer than the
    persisted body, so the model treats it as the live target."""
    working_draft = (state.get("working_draft") or "").strip()
    if not working_draft:
        return ""
    return (
        "\n\nDRAFT ĐANG XÂY DỰNG (cập nhật tăng dần — phản ánh các ý đã rõ):\n"
        f"{working_draft}"
    )


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
_INTERRUPT_BEARING_TOOLS: frozenset[str] = frozenset({
    "ask_user", "respond", "write_draft", "finalize", "confirm_intent",
})

# Silent scratchpad notes: no interrupt, no DB write, pure state append (assumptions/risks/
# open_questions/key_facts). They may ride along with an interrupt-bearing tool because the ToolNode
# discards their partial update when the interrupt fires and re-applies it exactly once on resume —
# so the model can record what it learned in the SAME turn it asks a question, instead of having the
# note dropped by solo enforcement (the only key_facts populator, which starved the anti-re-ask block).
_SIDE_EFFECT_FREE_NOTE_TOOLS: frozenset[str] = frozenset({"critique_note", "explore_note"})


def _gate_selected_tools(_state: WorkflowState, requested: list[dict]) -> list[dict]:
    """Enforce the ToolNode safety invariant without picking a tool on the model's behalf.

    The remaining gate is solo enforcement for interrupt-bearing tools: keep the first interrupt plus
    side-effect-free notes, drop the rest. Tools decide unavailable/missing-arg via a tool_result error.

    ``_state`` is part of the gate contract (callers pass it) but the current solo-enforcement rule
    needs only the requested set; it is intentionally unused.
    """
    # Normalize to a stable {name, args} shape; the model's chosen tools are never substituted.
    validated = [
        {"name": item.get("name") or "", "args": dict(item.get("args") or {})}
        for item in requested
    ]

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

    artifact_context = "\n".join(
        f"- [{a['type']}] {a['title']} (id={a['id']})" for a in artifacts
    ) or "(chưa có artifact nào)"

    # The analyst already receives the full conversation as a real message thread
    # (_build_analyzer_messages), so restating it here would double every recent turn. The payload
    # carries only the running summary — a deliberate compaction of older turns — when one exists.
    conversation_summary = (state.get("conversation_summary") or "").strip()
    summary_block = (
        f"Tóm tắt hội thoại đã tích lũy:\n{conversation_summary}\n\n"
        if conversation_summary
        else ""
    )

    locale = (state.get("locale") or "").strip()
    language_lock = (
        f"\n\nQUAN TRỌNG: Trả lời TOÀN BỘ bằng ngôn ngữ '{locale}'. Tuyệt đối không trộn lẫn ngôn ngữ khác."
        if locale
        else ""
    )

    tool_menu = ", ".join(t.name for t in get_available_tools(state))
    draft_block = _build_draft_block(state, draft_body)
    working_draft_block = _build_working_draft_block(state)
    contract_block = _build_output_contract_block(state)
    feedback_block = _build_feedback_control_block(state)
    key_facts_block = _build_key_facts_block(state)

    return (
        f"Bạn là analyst cho loại artifact: {state['artifact_type']}.\n\n"
        f"Context hiện tại:\n{artifact_context}"
        f"{_build_taxonomy_chain_block(state)}\n\n"
        f"{summary_block}"
        f"Công cụ khả dụng lượt này: {tool_menu}.\n"
        "Chọn 1–3 công cụ phù hợp và điền các field của từng công cụ theo policy trong system prompt."
        f"{_build_section_coverage_hint(state)}"
        f"{contract_block}"
        f"{key_facts_block}"
        f"{feedback_block}"
        f"{draft_block}"
        f"{working_draft_block}"
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
    if feedback_summary.get("stale_warning"):
        parts.append(f"- stale_warning: {feedback_summary['stale_warning']}")

    if not parts:
        return ""
    return "\n\nFEEDBACK CONTROL:\n" + "\n".join(parts)


def _compact_list(values: list[Any], limit: int = 3) -> str:
    rendered = [str(value) for value in values[:limit] if str(value).strip()]
    if len(values) > limit:
        rendered.append(f"... (+{len(values) - limit})")
    return "; ".join(rendered)


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
        marker = " (đang làm)" if item_type == artifact_type else ""
        lines.append(f"- {item_type}{marker}: {desc}")
    if not lines:
        return ""
    return "\n\nLOẠI ARTIFACT & nguồn gốc (chain):\n" + "\n".join(lines)


def _build_output_contract_block(state: WorkflowState) -> str:
    artifact_type = state["artifact_type"]
    try:
        contract = output_contract(artifact_type)
    except ValueError:
        return ""
    headings = "\n".join(f"- {heading}" for heading in contract.required_headings)
    columns = ", ".join(contract.table_columns) if contract.table_columns else "(không bắt buộc table)"
    return (
        "\n\nOUTPUT CONTRACT BẮT BUỘC:\n"
        f"- Artifact type: {artifact_type}\n"
        "- Body phải là Markdown theo chuẩn artifact này, không phải JSON/form dump.\n"
        "- Hội thoại/user input chỉ là evidence/context; không copy nguyên transcript vào body.\n"
        "- Nội dung agent suy diễn hoặc cần user xác nhận phải ghi note ngay tại chỗ trong ngoặc, "
        f"ví dụ {contract.confirmation_note}.\n"
        "- Khi input còn ít, candidate vẫn phải giữ cấu trúc đầy đủ và đánh dấu rõ: "
        "`inferred` cho phần agent suy luận, `missing` cho phần thiếu evidence, "
        "`needs_confirmation` cho assumption cần user xác nhận.\n"
        "- Không làm nghèo body bằng cách bỏ heading; nếu chưa đủ dữ liệu, giữ heading và ghi phần thiếu rõ ràng.\n"
        f"- Guidance: {contract.guidance}\n"
        "Required headings:\n"
        f"{headings}\n"
        f"Table columns khi dùng table: {columns}\n"
        "Ưu tiên current/accepted artifact version và predecessor đã accepted hơn chat history."
    )


def _build_mode_hint_directive(state: WorkflowState) -> str:
    """Inject a user-supplied `mode_hint` — an explicit "cướp lái" to switch operating angle now.

    Dynamic per-turn payload only. The proactive-mode policy (when to leave plain Q&A, prefer
    respond over burying an assessment in a question) is static and lives in the decision-policy
    instruction layer, not here.
    """
    mode_hint = (state.get("mode_hint") or "").strip()
    if not mode_hint:
        return ""
    return (
        f"\n\nYÊU CẦU MODE: người dùng muốn chuyển sang chế độ '{mode_hint}'. Hãy chuyển ngay "
        f"trong lượt này, đặt active_mode='{mode_hint}' và phản hồi đúng theo chế độ đó."
    )


def _build_section_coverage_hint(state: WorkflowState) -> str:
    if state.get("coverage_complete") is not False:
        return ""
    section_coverage = state.get("section_coverage") or {}
    # Stall: coverage stopped advancing — re-pinning the same gaps would reproduce the previous
    # question verbatim, so steer the model to synthesize what it has and move on or propose.
    if (state.get("section_coverage_stall_count") or 0) >= 2:
        return (
            "\n\nĐộ phủ section không tăng qua nhiều lượt. Đừng lặp lại cùng hướng khai thác — hãy tổng "
            "hợp những gì đã có và cân nhắc propose, hoặc chuyển sang một angle hoàn toàn khác."
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
        "\n\nĐộ phủ section — các khía cạnh còn thiếu hoặc chưa rõ (tham chiếu, không phải thứ tự "
        "bắt buộc):\n"
        f"{inventory}\n"
        "Tự chọn angle phù hợp nhất với mạch hội thoại để advance — khai thác thêm, suy luận điều hợp "
        "lý, hoặc draft khi đã đủ."
    )


def _build_summary_prompt(state: WorkflowState) -> str:
    current_summary = (state.get("conversation_summary") or "").strip() or "(chưa có)"
    recent_messages = "\n".join(
        f"{role}: {content}"
        for role, content in (
            _msg_role_content(m)
            for m in (state.get("messages") or [])[-settings.summary_trigger_every:]
        )
    ) or "(chưa có hội thoại mới)"

    return (
        "Cập nhật tóm tắt chạy cho hội thoại yêu cầu sản phẩm.\n\n"
        f"TÓM TẮT HIỆN TẠI:\n{current_summary}\n\n"
        f"HỘI THOẠI MỚI:\n{recent_messages}\n\n"
        "Trả về đúng bốn section sau:\n"
        "Yêu cầu đã xác nhận\n"
        "Ràng buộc — KHÔNG paraphrase\n"
        "Khoảng trống chưa rõ\n"
        "Quyết định đã thống nhất\n\n"
        "Trong section ràng buộc, giữ nguyên verbatim mọi số liệu, deadline, tên riêng và giới hạn phạm vi."
    )
