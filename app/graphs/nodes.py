import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from sqlalchemy import exists, select

from app.config import settings
from app.graphs.policy import ancestor_types
from app.graphs.section_schema import (
    COVERAGE_STALL_LIMIT,
    SECTION_DESCRIPTIONS,
    SECTION_SPECS,
    compute_section_coverage,
    status_score,
)
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

# Tool-loop selection schema (Phase 5 shim). The analyst names the tool to run this turn plus its
# args. The analytic fields (confidence, gaps, section_assessment, active_mode, draft_update, ...) feed
# eval (active_mode), incremental draft (draft_update) and the coverage gate (section_assessment).
TOOL_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": [
                "ask_user", "respond", "write_draft", "finalize",
                "critique_note", "explore_note", "run_critique",
                "recommend_next_workflow", "run_readiness_check",
            ],
        },
        "message": {"type": "string"},
        "target": {"type": "string"},
        "mode": {"type": "string"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "summary": {"type": "string"},
        "content": {"type": "string"},
        "confidence": {"type": "number"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "answer_assessment": {"type": "string", "enum": ["complete", "partial", "none"]},
        "acknowledgment": {"type": "string"},
        "section_assessment": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["missing", "partial", "filled", "needs_review"],
            },
        },
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
                     "architecture_readiness", "epic_story_readiness"],
        },
        "planning_track": {"type": "string", "enum": ["quick", "standard", "enterprise"]},
    },
    "required": ["tool"],
}

# Valid BMAD workflow modes and planning tracks, plus aliases the LLM may report. analyze_node
# normalizes (alias -> lowercase/strip -> enum) and falls back to brainstorm / quick on miss.
_WORKFLOW_MODES = {"brainstorm", "brief", "prd", "readiness_check", "architecture_readiness", "epic_story_readiness"}
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
    scores = {section: status_score(cov.get(section)) for section in SECTION_SPECS}
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

async def analyze_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    project_id = uuid.UUID(cfg["project_id"])
    llm_client = cfg.get("strong_llm_client") or cfg["llm_client"]
    if llm_client is None:
        raise ValueError("Chưa cấu hình LLM provider. Vui lòng thêm API key trong phần cài đặt.")

    # Context for the analyst = artifacts of the current type (avoid duplicates)
    # plus its full transitive ancestry — the upstream sources it must derive
    # from (e.g. a `story` traces back through `epic` ... up to `intent`). Using
    # the closure (not just direct parents) makes provenance complete for every
    # type regardless of how ARTIFACT_PREDECESSORS declares it; dedup keeps it
    # token-light since read_artifacts returns title-only rows (no body).
    artifact_type = state["artifact_type"]
    context_types = [artifact_type, *ancestor_types(artifact_type)]
    async with session_factory() as db:
        artifacts: list[dict[str, Any]] = []
        for context_type in context_types:
            artifacts.extend(
                await read_artifacts(
                    db=db,
                    project_id=project_id,
                    artifact_type=context_type,
                    context={"workflow_area": state["workflow_area"]},
                )
            )
        # Load the current draft body for this artifact_type so the analyst can mine
        # the delta instead of re-asking what the draft already records (M7/M8).
        draft = await read_current_body(
            db=db, project_id=project_id, artifact_type=artifact_type
        )
    draft_body = draft["body"] if draft else None

    prompt = _build_tool_selection_prompt(state, artifacts, draft_body)
    response_format = TOOL_SELECTION_SCHEMA
    system_prompt = get_instruction(
        artifact_type=state["artifact_type"],
        workflow_area=state["workflow_area"],
        agent_role=cfg.get("agent_role"),
    )
    started_at = time.monotonic()
    analysis_result, usage = await llm_client.generate(
        messages=[{"role": "user", "content": prompt}],
        system=system_prompt,
        max_tokens=2000,
        response_format=response_format,
    )
    latency_ms = int((time.monotonic() - started_at) * 1000)
    section_assessment = analysis_result.get("section_assessment") if isinstance(analysis_result, dict) else None
    # section-coverage: deterministic, no LLM call. Keep "LLM did not report" (None -> fail-open)
    # separate from "reported empty {}" (evaluated as missing -> gate), so non-section-aware
    # turns such as derived artifacts or confident proposals continue normally.
    if section_assessment is None:
        coverage = {"section_coverage": None, "coverage_ratio": None, "coverage_complete": None}
    else:
        coverage = compute_section_coverage(section_assessment)
    # Stall counter: increment when a gated turn fails to raise coverage, reset otherwise.
    # route_node and the coverage hint read it to escape a non-advancing elicitation loop.
    prev_ratio = state.get("coverage_ratio")
    new_ratio = coverage["coverage_ratio"]
    if new_ratio is None or coverage["coverage_complete"] or prev_ratio is None or new_ratio > prev_ratio:
        coverage["section_coverage_stall_count"] = 0
    else:
        coverage["section_coverage_stall_count"] = (state.get("section_coverage_stall_count") or 0) + 1
    if isinstance(analysis_result, dict):
        analysis_result = {
            **analysis_result,
            "coverage_ratio": coverage["coverage_ratio"],
            "coverage_complete": coverage["coverage_complete"],
        }

    # Tool-loop shim: enforce the finalize hard-gate (and reject unknown picks) by coercing a
    # selection that names a tool not currently offered down to a safe ask_user (S4). Done before
    # persist so analysis_result records the tool actually dispatched.
    if isinstance(analysis_result, dict):
        gated_tool = _gate_selected_tool(state, analysis_result.get("tool"))
        analysis_result = {**analysis_result, "tool": gated_tool}
        # Derive active_mode from a note tool so S1 proactive coverage is tool-driven, not
        # dependent on the model also filling active_mode (which it reliably defaults to 'qa').
        if gated_tool in _NOTE_TOOL_MODE:
            analysis_result["active_mode"] = _NOTE_TOOL_MODE[gated_tool]
        # respond carries its own angle: clamp to a valid proactive mode (default critique) so a
        # user-facing assessment is always a proactive mode, never the discovery baseline.
        elif gated_tool == "respond":
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
    method_profile = dict(state.get("method_profile") or DEFAULT_METHOD_PROFILE)
    if reported.get("workflow_mode"):
        method_profile["current_workflow"] = _normalize_workflow_mode(reported.get("workflow_mode"))
    else:
        method_profile["current_workflow"] = _infer_workflow_mode(state)
    method_profile["planning_track"] = _normalize_planning_track(
        reported.get("planning_track") or method_profile.get("planning_track")
    )

    result = {
        "analysis_result": analysis_result,
        "turn_count": state["turn_count"] + 1,
        "last_agent_run_id": run_id,
        # Locale is detected on first contact and then sticky: once set it overrides later turns so the
        # output language lock stays stable even if the model omits the field mid-conversation.
        "locale": state.get("locale") or reported.get("locale"),
        # Persist the DB-loaded draft body so run_critique can target it next turn.
        "draft_body": draft_body,
        "working_draft": draft_update or state.get("working_draft"),
        "method_profile": method_profile,
        # Display/persistence snapshot; recommend_next_workflow re-derives inline to avoid staleness.
        "artifact_chain": _derive_artifact_chain(coverage.get("section_coverage")),
        # Multi-angle (S2): the mode_hint is a one-shot steer. It has already been folded into
        # this turn's prompt, so clear it now — the next turn returns to proactive default.
        "mode_hint": None,
        **coverage,
    }
    # Tool-loop shim: emit the selected tool as an AIMessage(tool_calls) so route_node dispatches it
    # to the ToolNode. tool_call.id = AgentRun.id keeps the tool idempotency keys aligned on resume.
    # tool=None means the analyst is done: emit a plain AIMessage (no tool_calls) so route_node ends.
    if isinstance(analysis_result, dict):
        tool = analysis_result.get("tool")
        if tool:
            args = {key: analysis_result.get(key, "") for key in _TOOL_ARG_KEYS[tool]}
            # A pick coerced to ask_user (gated-out tool) often carries no message — never ask a blank
            # question; fall back to the same prompt the coverage-gate uses.
            if tool == "ask_user" and not str(args.get("message") or "").strip():
                args["message"] = _COERCED_ASK_FALLBACK
            # respond needs its angle as a tool arg; take the active_mode just clamped above.
            if tool == "respond":
                args["mode"] = analysis_result.get("active_mode") or "critique"
            result["messages"] = [AIMessage(content="", tool_calls=[{"id": run_id, "name": tool, "args": args}])]
        else:
            done_message = str(analysis_result.get("summary") or analysis_result.get("message") or "")
            result["messages"] = [AIMessage(content=done_message)]
    return result


_COERCED_ASK_FALLBACK = (
    "Mình cần làm rõ thêm một ý trước khi có thể viết phần này chắc hơn. "
    "Bạn có thể chia sẻ thêm thông tin quan trọng nhất còn thiếu không?"
)


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
) -> str:
    """Persist one agent turn (idempotently), mark the session waiting, then interrupt.

    Shared by ask_user (kind="question") and respond (kind="assessment", mode set). Both use the
    ASK_HUMAN interrupt_type so the resume accepts a free-text reply; only the persisted message kind
    and the carried mode differ. Keying the idempotency guard on run_id is what makes an HTTP-resume —
    which re-executes the tool body from the top — skip the duplicate insert (R1). Returns the resumed
    user content so the caller can fold it back into the conversation.
    """
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
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.ASK_HUMAN
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


def _gate_selected_tool(state: WorkflowState, selected: str | None) -> str | None:
    """Clamp the analyst's tool pick to the currently offered set.

    - No pick (None/empty) → None: the loop is done, route_node ends the turn.
    - A pick outside get_available_tools (e.g. finalize before working_draft exists, an unknown name)
      → ask_user: a gated-out tool must not dispatch, so degrade to a safe clarifying question.
    - An offered pick → itself.
    """
    if not selected:
        return None
    from app.graphs.agent_tools import get_available_tools

    available = {t.name for t in get_available_tools(state)}
    return selected if selected in available else "ask_user"


def _build_tool_selection_prompt(state: WorkflowState, artifacts: list[dict], draft_body: str | None = None) -> str:
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

    return (
        f"Bạn là analyst cho loại artifact: {state['artifact_type']}.\n\n"
        f"Context hiện tại:\n{artifact_context}\n\n"
        f"Hội thoại gần đây:\n{messages_summary}\n\n"
        f"Công cụ khả dụng lượt này: {tool_menu}.\n"
        "Chọn đúng MỘT công cụ và điền các field của nó theo policy trong system prompt."
        f"{_build_section_coverage_hint(state)}"
        f"{draft_block}"
        f"{working_draft_block}"
        f"{_build_mode_hint_directive(state)}"
        f"{language_lock}"
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
    if (state.get("section_coverage_stall_count") or 0) >= COVERAGE_STALL_LIMIT:
        return (
            "\n\nĐộ phủ section không tăng qua nhiều lượt. Đừng lặp lại cùng hướng khai thác — hãy tổng "
            "hợp những gì đã có và cân nhắc propose, hoặc chuyển sang một angle hoàn toàn khác."
        )
    # Gap-inventory: list every weak section (missing first, then partial/needs_review) so the LLM
    # picks the angle that fits the conversation instead of being pinned to one scripted question.
    gap_lines = [
        f"- {SECTION_DESCRIPTIONS.get(section, section)} ({section_coverage.get(section)})"
        for status in ("missing", "partial", "needs_review")
        for section in SECTION_SPECS
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
