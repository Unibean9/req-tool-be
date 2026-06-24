import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
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

# Tool-loop selection schema. The analyst names the tools to run this turn plus per-tool args.
# tools: list of 1–3 {name, args} objects. Analytic fields (active_mode, locale, etc.) are
# top-level — they inform eval and state updates, not dispatched as tool args.
TOOL_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "ask_user", "respond", "write_draft", "finalize",
                            "critique_note", "explore_note", "run_critique",
                            "recommend_next_workflow", "run_readiness_check",
                        ],
                    },
                    "args": {"type": "object"},
                },
                "required": ["name"],
            },
        },
        "confidence": {"type": "number"},
        "answer_assessment": {"type": "string", "enum": ["complete", "partial", "none"]},
        "acknowledgment": {"type": "string"},
        "active_mode": {
            "type": "string",
            "enum": ["discovery", "structuring", "critique", "revision", "finalization"],
        },
        # Detected on first contact and kept sticky by analyze_node; drives the output language lock.
        "locale": {"type": "string", "enum": ["vi", "en"]},
        "draft_update": {"type": "string"},
        # BMAD method layer (addendum §11) — separate from active_mode. workflow_mode = planning
        # stage of the project; planning_track = artifact-chain depth.
        "workflow_mode": {
            "type": "string",
            "enum": ["brainstorm", "brief", "prd", "readiness_check",
                     "architecture_readiness"],
        },
        "planning_track": {"type": "string", "enum": ["quick", "standard", "enterprise"]},
    },
    "required": ["tools"],
}

# Valid BMAD workflow modes and planning tracks, plus aliases the LLM may report. analyze_node
# normalizes (alias -> lowercase/strip -> enum) and falls back to brainstorm / quick on miss.
_WORKFLOW_MODES = {"brainstorm", "brief", "prd", "readiness_check", "architecture_readiness"}
_WORKFLOW_MODE_ALIASES = {"product_brief": "brief"}
_PLANNING_TRACKS = {"quick", "standard", "enterprise"}

# Per-tool arg names the shim copies from the selection dict into the tool_call args.
_TOOL_ARG_KEYS = {
    "ask_user": ["message"],
    "respond": ["message"],
    "write_draft": ["title", "body"],
    "finalize": ["summary"],
    "critique_note": ["content"],
    "explore_note": ["content"],
    "run_critique": ["target", "mode"],
    "recommend_next_workflow": ["current_artifact_type", "planning_track"],
    "run_readiness_check": ["target"],
}

# Args that MUST be non-empty for a pick to dispatch — a subset of _TOOL_ARG_KEYS. Emitting a
# tool_call with an empty required arg is a silent failure (write_draft body="", finalize summary=""),
# so analyze_node degrades to a re-ask naming the field instead. Tools with their own coerced
# fallback (ask_user/respond/notes) are deliberately absent. `run_critique.target` is NOT listed:
# it is cosmetic (ARG001 — the judge scores the loaded draft, not target), so only `mode` is required.
_TOOL_REQUIRED_ARGS = {
    "write_draft": ["body"],
    "finalize": ["summary"],
    "run_critique": ["mode"],
}

# Note tools commit the analyst to an operating angle; analyze_node derives active_mode from the
# picked tool so proactive S1 coverage no longer depends on the model self-reporting it. Values are
# already in the spec §7.1 vocabulary (explore_note -> structuring, not discovery).
_NOTE_TOOL_MODE = {"critique_note": "critique", "explore_note": "structuring"}


def _normalize_workflow_mode(mode: Any) -> str:
    """Alias -> lowercase/strip -> enum; fall back to 'brainstorm' on anything unrecognized."""
    raw = str(mode or "").strip().lower()
    raw = _WORKFLOW_MODE_ALIASES.get(raw, raw)
    return raw if raw in _WORKFLOW_MODES else "brainstorm"


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

# respond is a user-facing critique/exploration; its angle is the mode the model picked (defaulting
# to critique). active_mode is derived from it and also passed as the tool's `mode` arg.
_RESPOND_MODES = ("critique", "structuring")

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
    result, _usage = await llm_client.generate(
        messages=[{"role": "user", "content": prompt}],
        system=TRIAGE_SYSTEM,
        max_tokens=300,
        response_format=TRIAGE_SCHEMA,
    )
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
    response_format = TOOL_SELECTION_SCHEMA
    system_prompt = get_instruction(
        artifact_type=effective_state["artifact_type"],
        workflow_area=effective_state["workflow_area"],
        agent_role=cfg.get("agent_role"),
        context={"has_draft": draft_body is not None},
    )
    started_at = time.monotonic()
    analysis_result, usage = await llm_client.generate(
        messages=[{"role": "user", "content": prompt}],
        system=system_prompt,
        max_tokens=2000,
        response_format=response_format,
    )
    latency_ms = int((time.monotonic() - started_at) * 1000)
    if isinstance(analysis_result, dict):
        analysis_result = {
            **analysis_result,
            "coverage_complete": coverage["coverage_complete"],
        }

    # Gate and normalize the tools list. Done before persist so analysis_result records the tools
    # actually dispatched. The gate coerces unavailable tools to ask_user, then enforces that
    # interrupt-bearing tools run solo (first interrupt-bearing tool wins, others dropped).
    if isinstance(analysis_result, dict):
        raw_tools = analysis_result.get("tools") or []
        if not isinstance(raw_tools, list) or not raw_tools:
            # Backward-compat: model returned old format {"tool": "name", ...}.
            # Reconstruct per-tool args so the gate can coerce it normally.
            # NOTE: empty dict {} is a terminal signal (analyst is done) — do NOT coerce it.
            old_tool = analysis_result.get("tool") or ""
            if old_tool:
                old_args = {k: analysis_result.get(k, "") for k in _TOOL_ARG_KEYS.get(old_tool, [])}
                raw_tools = [{"name": old_tool, "args": old_args}]
        gated_tools = _gate_selected_tools(effective_state, raw_tools)
        analysis_result = {**analysis_result, "tools": gated_tools}

        # Observability: record first coerced/dropped tool as gated_tool/gated_reason so eval and
        # tests can see what was requested vs dispatched (mirrors old _degrade_reason behavior).
        _record_gate_observability(analysis_result, raw_tools, gated_tools, effective_state)

        # Derive active_mode from the primary (first) gated tool.
        primary_tool = gated_tools[0]["name"] if gated_tools else None
        if primary_tool in _NOTE_TOOL_MODE:
            analysis_result["active_mode"] = _NOTE_TOOL_MODE[primary_tool]
        elif primary_tool == "respond":
            mode = analysis_result.get("active_mode")
            analysis_result["active_mode"] = mode if mode in _RESPOND_MODES else "critique"

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

    # Incremental write (C1): carry the running draft forward. A turn that emits no
    # draft_update keeps the prior draft instead of None-ing it out. An empty-string
    # draft_update is treated as absent — intentional: the draft only grows, the model is
    # never allowed to reset it mid-session (the prompt forbids rewriting from scratch).
    draft_update = analysis_result.get("draft_update") if isinstance(analysis_result, dict) else None

    # BMAD method profile: take the LLM's workflow_mode (alias->enum, fallback brainstorm) or infer
    # from coverage when omitted; planning_track normalized to quick on miss. Merge so other profile
    # fields persist. Independent of active_mode.
    reported = analysis_result if isinstance(analysis_result, dict) else {}
    method_profile = dict(effective_state.get("method_profile") or DEFAULT_METHOD_PROFILE)
    if reported.get("workflow_mode"):
        method_profile["current_workflow"] = _normalize_workflow_mode(reported.get("workflow_mode"))
    else:
        method_profile["current_workflow"] = _infer_workflow_mode(effective_state)
    method_profile["planning_track"] = _normalize_planning_track(
        reported.get("planning_track") or method_profile.get("planning_track")
    )

    result = {
        "analysis_result": analysis_result,
        "turn_count": effective_state["turn_count"] + 1,
        "last_agent_run_id": run_id,
        # Locale is detected on first contact and then sticky: once set it overrides later turns so the
        # output language lock stays stable even if the model omits the field mid-conversation.
        "locale": effective_state.get("locale") or reported.get("locale"),
        # Persist the DB-loaded draft body so run_critique can target it next turn.
        "draft_body": draft_body,
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
    # Empty tools list means the analyst is done: emit a plain AIMessage (no tool_calls).
    if isinstance(analysis_result, dict):
        gated_tools_list = analysis_result.get("tools") or []
        if gated_tools_list:
            tool_calls = []
            for i, item in enumerate(gated_tools_list):
                tool = item.get("name") or ""
                args = dict(item.get("args") or {})
                # Per-tool post-processing (coercions that must happen at dispatch time).
                if tool == "ask_user" and not str(args.get("message") or "").strip():
                    args["message"] = _COERCED_ASK_FALLBACK
                if tool == "respond":
                    if _response_message_incomplete(args.get("message")):
                        args["message"] = _RESPOND_FALLBACK
                    args["mode"] = args.get("mode") or analysis_result.get("active_mode") or "critique"
                tool_calls.append({"id": f"{run_id}:{i}", "name": tool, "args": args})
            result["messages"] = [AIMessage(content="", tool_calls=tool_calls)]
        else:
            done_message = str(analysis_result.get("summary") or analysis_result.get("message") or "")
            result["messages"] = [AIMessage(content=done_message)]
    return result


_COERCED_ASK_FALLBACK = (
    "Mình cần làm rõ thêm một ý trước khi có thể viết phần này chắc hơn. "
    "Bạn có thể chia sẻ thêm thông tin quan trọng nhất còn thiếu không?"
)

# Fail-loud re-ask messages (Phase 2/3). User-facing, so Vietnamese; the machine-readable reason
# lives in analysis_result["gated_reason"], not here.
_MISSING_ARG_PROMPT = (
    "Mình chưa đủ thông tin để hoàn tất bước này (thiếu '{field}'). "
    "Bạn bổ sung giúp mình phần đó để mình tiếp tục nhé?"
)

_GATED_TOOL_PROMPT = (
    "Bước '{tool}' chưa khả dụng ở thời điểm này — mình cần làm rõ thêm trước đã. "
    "Bạn chia sẻ thêm thông tin quan trọng còn thiếu giúp mình nhé?"
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
    result, _usage = await llm_client.generate(
        messages=[{"role": "user", "content": prompt}],
        system=SUMMARY_SYSTEM,
        max_tokens=1000,
        response_format=SUMMARY_SCHEMA,
    )
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


def _build_draft_block(state: WorkflowState, draft_body: str | None) -> str:
    """Persisted-draft block: tell the analyst the body already on record, so it mines the delta."""
    if not draft_body:
        return ""
    return (
        f"\n\nDRAFT ĐANG CÓ cho loại '{state['artifact_type']}':\n{draft_body}\n\n"
        "QUAN TRỌNG: nội dung trên ĐÃ được ghi nhận. TUYỆT ĐỐI không hỏi lại thông tin đã "
        "có trong draft. Chỉ hỏi/khai thác phần user muốn bổ sung hoặc thay đổi (delta). "
        "Nếu user chỉ muốn cập nhật, tập trung vào điểm cần sửa, không khởi tạo lại từ đầu."
    )


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
        f"{working_draft}\n\n"
        "Với mỗi ý mới user vừa nêu, cập nhật draft trên qua field draft_update (bồi đắp, "
        "không viết lại từ đầu, không bịa nội dung chưa có). KHÔNG hỏi lại nội dung đã có "
        "trong draft."
    )


def _last_message_has_tool_calls(state: WorkflowState) -> bool:
    """Whether the most recent message carries tool_calls (a tool-loop dispatch signal)."""
    messages = state.get("messages") or []
    if not messages:
        return False
    return bool(getattr(messages[-1], "tool_calls", None))


def _record_gate_observability(
    analysis_result: dict, raw_tools: list[dict], gated_tools: list[dict], state: WorkflowState
) -> None:
    """Mutate analysis_result in-place with gated_tool/gated_reason when coercion occurred.

    Preserves the fail-loud contract: eval and tests can always observe what was requested vs
    dispatched. The first coerced or dropped tool drives the markers.
    """
    raw_names = [item.get("name") for item in raw_tools]
    gated_names = [item.get("name") for item in gated_tools]
    # Coercion: a tool was replaced by ask_user
    for orig, gated in zip(raw_names, gated_names):
        if orig != gated:
            feedback_detail = _feedback_degrade_detail(state) if orig == "finalize" else ""
            reason = f"gated: {orig} not available this turn"
            if feedback_detail:
                reason = f"{reason}; {feedback_detail}"
            analysis_result["gated_tool"] = orig
            analysis_result["gated_reason"] = reason
            analysis_result.setdefault(
                "message",
                _feedback_degrade_message(state) or _GATED_TOOL_PROMPT.format(tool=orig),
            )
            return
    # Solo enforcement: tools were dropped (interrupt-bearing kept, others dropped)
    if len(gated_tools) < len(raw_tools):
        gated_set = set(gated_names)
        dropped = [n for n in raw_names if n not in gated_set]
        if dropped:
            analysis_result["gated_tool"] = dropped[0]
            analysis_result["gated_reason"] = f"dropped: {dropped[0]} paired with interrupt-bearing tool"


# Tools that call interrupt() — they must always run solo (no composite dispatch).
# DB-writing tools (write_draft, finalize) are also in this set: they interrupt and must not
# be paired with another tool in the same turn to preserve idempotency invariants.
_INTERRUPT_BEARING_TOOLS: frozenset[str] = frozenset({
    "ask_user", "respond", "write_draft", "finalize",
})


def _gate_selected_tools(state: WorkflowState, requested: list[dict]) -> list[dict]:
    """Coerce and gate a requested tools list to what is safe to dispatch this turn.

    Three-step gate (order matters):
    1. Per-tool availability coercion: any tool not in get_available_tools() is replaced by ask_user.
    2. Required-arg validation: any tool missing a required arg (see _TOOL_REQUIRED_ARGS) is replaced
       by ask_user so the model must re-ask rather than dispatch an incomplete call.
    3. Interrupt-bearing solo enforcement: if any tool in the coerced list is interrupt-bearing,
       keep only the first such tool and drop everything else.

    Returns the gated list (1–N elements, all available, interrupt-bearing at most one).
    """
    from app.graphs.agent_tools import get_available_tools

    available = {t.name for t in get_available_tools(state)}

    coerced: list[dict] = []
    for item in requested:
        name = item.get("name") or ""
        gated_name = name if name in available else "ask_user"
        coerced.append({**item, "name": gated_name})

    validated: list[dict] = []
    for item in coerced:
        name = item.get("name") or ""
        args = item.get("args") or {}
        missing = next(
            (a for a in _TOOL_REQUIRED_ARGS.get(name, ()) if not str(args.get(a) or "").strip()),
            None,
        )
        if missing:
            # Re-ask rather than dispatch an incomplete call — the model owes a required arg it left blank.
            validated.append({
                "name": "ask_user",
                "args": {"message": _MISSING_ARG_PROMPT.format(field=missing)},
            })
        else:
            validated.append(item)

    for item in validated:
        if item["name"] in _INTERRUPT_BEARING_TOOLS:
            return [item]

    return validated


def _gate_selected_tool(state: WorkflowState, selected: str | None) -> str | None:
    """Legacy single-tool gate — kept for backward compatibility with existing call sites.

    Delegates to _gate_selected_tools; always returns a single tool name or None.
    """
    if not selected:
        return None
    result = _gate_selected_tools(state, [{"name": selected, "args": {}}])
    return result[0]["name"] if result else None


def _missing_required_arg(tool: str | None, analysis_result: dict) -> str | None:
    """First required arg of `tool` that is empty/blank in the selection, else None.

    Mirrors _TOOL_REQUIRED_ARGS — tools with their own coerced fallback have no required args here.
    """
    for arg in _TOOL_REQUIRED_ARGS.get(tool or "", ()):
        if not str(analysis_result.get(arg) or "").strip():
            return arg
    return None


def _degrade_reason(
    state: WorkflowState, requested: str | None, gated_tool: str | None, analysis_result: dict
) -> dict | None:
    """Why this pick can't dispatch as-is (fail-loud), or None when it's fine to emit.

    Returns the analysis_result overlay for the degrade: a `gated_tool`/`gated_reason` pair (observable
    for eval/tests) plus the user-facing re-ask `message`. Two cases:
    - out-of-menu (Phase 3): the model named a tool `_gate_selected_tool` clamped away.
    - missing required arg (Phase 2): the picked tool's required arg is empty.
    """
    if requested and requested != "ask_user" and gated_tool == "ask_user":
        feedback_detail = _feedback_degrade_detail(state) if requested == "finalize" else ""
        gated_reason = f"gated: {requested} not available this turn"
        if feedback_detail:
            gated_reason = f"{gated_reason}; {feedback_detail}"
        return {
            "gated_tool": requested,
            "gated_reason": gated_reason,
            "message": _feedback_degrade_message(state) or _GATED_TOOL_PROMPT.format(tool=requested),
        }
    missing = _missing_required_arg(gated_tool, analysis_result)
    if missing:
        return {
            "gated_tool": gated_tool,
            "gated_reason": f"gated: {gated_tool} missing required arg '{missing}'",
            "message": _MISSING_ARG_PROMPT.format(field=missing),
        }
    return None


def _build_tool_selection_prompt(
    state: WorkflowState,
    artifacts: list[dict],
    draft_body: str | None = None,
) -> str:
    """Build the per-turn analyst payload: context the model needs to pick the next tool.

    This is dynamic payload only — artifact context, the conversation window, the tools available
    this turn, and state-dependent hints (coverage gaps, the running/persisted draft, a one-shot
    mode_hint, the locale lock). All static policy — tool semantics, the section grading rubric, the
    proactive-mode and content-depth rules — lives in the instruction layers (the system prompt), so
    it is never restated here. analyze_node converts the returned dict into an AIMessage(tool_calls).
    """
    from app.graphs.agent_tools import get_available_tools

    artifact_context = "\n".join(
        f"- [{a['type']}] {a['title']} (id={a['id']})" for a in artifacts
    ) or "(chưa có artifact nào)"

    conversation_summary = (state.get("conversation_summary") or "").strip()
    message_window = (state.get("messages") or [])[-3:] if conversation_summary else (state.get("messages") or [])[-5:]
    messages_summary = "\n".join(
        f"{role}: {content}"
        for role, content in (_msg_role_content(m) for m in message_window)
    ) or "(chưa có hội thoại)"
    if conversation_summary:
        messages_summary = (
            "Tóm tắt hội thoại đã tích lũy:\n"
            f"{conversation_summary}\n\n"
            "Ba tin nhắn gần nhất:\n"
            f"{messages_summary}"
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
        f"Context hiện tại:\n{artifact_context}\n\n"
        f"Hội thoại gần đây:\n{messages_summary}\n\n"
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


def _feedback_degrade_detail(state: WorkflowState) -> str:
    details: list[str] = []
    report = state.get("quality_report") or {}
    if report.get("quality_gate_result"):
        details.append(f"quality_gate={report['quality_gate_result']}")
    readiness = state.get("candidate_readiness") or {}
    if readiness.get("state"):
        details.append(f"candidate_readiness={readiness['state']}")
    return "; ".join(details)


def _feedback_degrade_message(state: WorkflowState) -> str:
    report = state.get("quality_report") or {}
    readiness = state.get("candidate_readiness") or {}
    blockers = list(report.get("blocking_issues") or [])
    blockers.extend(readiness.get("blocking_reasons") or [])
    if not blockers:
        return ""
    return (
        "Chưa thể finalize vì feedback hiện tại còn blocker: "
        f"{_compact_list(blockers)}. Hãy revise candidate trước."
    )


def _compact_list(values: list[Any], limit: int = 3) -> str:
    rendered = [str(value) for value in values[:limit] if str(value).strip()]
    if len(values) > limit:
        rendered.append(f"... (+{len(values) - limit})")
    return "; ".join(rendered)


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
