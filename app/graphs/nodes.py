import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from sqlalchemy import exists, select

from app.config import settings
from app.graphs.personas import get_persona
from app.graphs.policy import ancestor_types
from app.graphs.slot_schema import (
    BRD_SLOTS,
    COVERAGE_STALL_LIMIT,
    SLOT_DESCRIPTIONS,
    compute_coverage,
)
from app.graphs.state import WorkflowState
from app.graphs.tools import read_artifacts, read_current_body
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
)

# Tool-loop selection schema (Phase 5 shim). The analyst names the tool to run this turn plus its
# args. The analytic fields (confidence, gaps, slot_assessment, active_mode, draft_update, ...) feed
# eval (active_mode), incremental draft (draft_update) and the coverage gate (slot_assessment).
TOOL_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["ask_user", "write_draft", "finalize", "write_note"]},
        "message": {"type": "string"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "summary": {"type": "string"},
        "content": {"type": "string"},
        "confidence": {"type": "number"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "answer_assessment": {"type": "string", "enum": ["complete", "partial", "none"]},
        "acknowledgment": {"type": "string"},
        "slot_assessment": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": ["filled", "partial", "empty"]},
        },
        "active_mode": {"type": "string", "enum": ["qa", "critique", "explore", "draft"]},
        "draft_update": {"type": "string"},
    },
    "required": ["tool"],
}

# Per-tool arg names the shim copies from the selection dict into the tool_call args.
_TOOL_ARG_KEYS = {
    "ask_user": ["message"],
    "write_draft": ["title", "body"],
    "finalize": ["summary"],
    "write_note": ["content"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["greeting", "smalltalk", "task", "unclear"]},
        "locale": {"type": "string", "enum": ["vi", "en"]},
    },
    "required": ["intent", "locale"],
}

INTENT_SYSTEM = (
    "Bạn là bộ phân loại ý định cho một trợ lý phân tích yêu cầu sản phẩm. "
    "Chỉ phân loại, không trả lời người dùng."
)

# Hard-coded greeting templates per locale — stable for the live demo, no LLM call.
_GREETING_TEMPLATES = {
    "vi": (
        "Xin chào! Tôi là trợ lý phân tích yêu cầu. Tôi có thể giúp bạn làm rõ ý tưởng và "
        "xây dựng các artifact như mục tiêu, vấn đề, user story... Bạn muốn bắt đầu từ đâu?"
    ),
    "en": (
        "Hello! I'm your requirements analysis assistant. I can help you clarify ideas and "
        "build artifacts such as goals, problems, and user stories. Where would you like to start?"
    ),
}


SUMMARY_SYSTEM = (
    "Bạn là trợ lý tóm tắt hội thoại yêu cầu sản phẩm. "
    "Giữ nguyên các ràng buộc quan trọng, đặc biệt số liệu, tên riêng, deadline và phạm vi."
)

async def intent_router_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """Classify the user's intent (greeting/smalltalk/task/unclear) and lock the locale.

    Entry point of the graph. Runs only on the first invocation — on resume LangGraph re-enters
    the interrupted node directly, so this never re-runs mid-conversation.
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
        "Phân loại tin nhắn của người dùng và phát hiện ngôn ngữ.\n\n"
        f"Tin nhắn: {last_user!r}\n\n"
        "intent: 'greeting' nếu chỉ là chào hỏi; 'smalltalk' nếu tán gẫu không liên quan công việc; "
        "'task' nếu là yêu cầu phân tích/tạo artifact; 'unclear' nếu không rõ.\n"
        "locale: 'vi' nếu tiếng Việt, 'en' nếu tiếng Anh."
    )
    result, _usage = await llm_client.generate(
        messages=[{"role": "user", "content": prompt}],
        system=INTENT_SYSTEM,
        max_tokens=200,
        response_format=INTENT_SCHEMA,
    )
    if isinstance(result, dict):
        intent = result.get("intent") or "task"
        locale = result.get("locale") or "vi"
    else:
        intent, locale = "task", "vi"
    return {"intent": intent, "locale": locale}


def route_after_intent(state: WorkflowState) -> str:
    if state.get("intent") in ("greeting", "smalltalk"):
        return "greeting"
    return "analyze"


async def greeting_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """Greet the user with a locale-templated message and pause for their reply.

    Saves an agent message with payload.kind='greeting', sets WAITING_FOR_HUMAN/ASK_HUMAN, and
    interrupts — exactly like ask_human, so the turn never silently completes. No LLM call, no AgentRun.
    """
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])

    locale = state.get("locale") or "vi"
    message = _GREETING_TEMPLATES.get(locale, _GREETING_TEMPLATES["vi"])

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
    system_prompt = get_persona(
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
    slot_assessment = analysis_result.get("slot_assessment") if isinstance(analysis_result, dict) else None
    # Optimistic credit: last turn the hint pinned `last_asked_slot` and the user has now
    # answered that exact question. A fresh 'empty' grade for it is the model under-reporting
    # its own prior question — the root cause of the verbatim-repeated ask. Credit it 'partial'
    # so coverage advances and the hint rotates to the next slot instead of repeating.
    last_asked = state.get("last_asked_slot")
    if slot_assessment is not None and last_asked and slot_assessment.get(last_asked) == "empty":
        slot_assessment = {**slot_assessment, last_asked: "partial"}
    # slot-coverage: deterministic, no LLM call. Keep "LLM did not report" (None -> fail-open)
    # separate from "reported empty {}" (evaluated as missing -> gate), so non-slot-aware
    # turns such as non-BRD artifacts or confident proposals continue normally.
    if slot_assessment is None:
        coverage = {"slot_coverage": None, "coverage_ratio": None, "coverage_complete": None}
    else:
        coverage = compute_coverage(artifact_type, slot_assessment)
    # Stall counter: increment when a gated turn fails to raise coverage, reset otherwise.
    # route_node and the coverage hint read it to escape a non-advancing elicitation loop.
    prev_ratio = state.get("coverage_ratio")
    new_ratio = coverage["coverage_ratio"]
    if new_ratio is None or coverage["coverage_complete"] or prev_ratio is None or new_ratio > prev_ratio:
        coverage["coverage_stall_count"] = 0
    else:
        coverage["coverage_stall_count"] = (state.get("coverage_stall_count") or 0) + 1
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
        analysis_result = {**analysis_result, "tool": _gate_selected_tool(state, analysis_result.get("tool"))}

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

    result = {
        "analysis_result": analysis_result,
        "turn_count": state["turn_count"] + 1,
        "last_agent_run_id": run_id,
        # Record the slot this turn's hint pinned (the slot just asked) so the next turn's
        # hint can rotate off it — the deterministic guard against re-asking the same slot.
        "last_asked_slot": _coverage_hint_target(state),
        "working_draft": draft_update or state.get("working_draft"),
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
    # get_available_tools / _build_coverage_hint, not here.
    return "tools" if _last_message_has_tool_calls(state) else END


async def _save_and_interrupt_ask(state: WorkflowState, config: RunnableConfig, content: str, *, run_id) -> str:
    """Persist one agent question (idempotently), mark the session waiting, then interrupt.

    Used by the ask_user tool (run_id = ToolCall.id). Keying the idempotency guard on run_id is what
    makes an HTTP-resume — which re-executes the tool body from the top — skip the duplicate insert
    (R1). Returns the resumed user content so the caller can fold it back into the conversation.
    """
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    locale = state.get("locale") or "vi"

    async with session_factory() as db:
        already_saved = await _agent_message_already_saved(db, session_id, run_id, content)
        if not already_saved:
            db.add(
                AgentMessage(
                    session_id=session_id,
                    role=AgentMessageRole.AGENT,
                    content=content,
                    payload={
                        "kind": "question",
                        "locale": locale,
                        "options": [],
                        "blocks": [],
                        "run_id": run_id,
                    },
                )
            )
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == session_id))
        ).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.ASK_HUMAN
        await db.commit()

    user_response = interrupt({"type": "ask_human", "message": content})
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
    """Tool-loop variant of the analyst prompt: the model picks the next tool instead of next_action.

    Reuses every analytic directive of the enum prompt (synthesis/slot/coverage/draft/mode/locale);
    only the action instruction differs — it lists the currently available tools and asks for a
    selection. analyze_node converts the returned dict into an AIMessage(tool_calls).
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
        f"Bạn là BA/PM analyst. Phân tích và đề xuất artifact cho loại: {state['artifact_type']}.\n\n"
        f"Context hiện tại:\n{artifact_context}\n\n"
        f"Hội thoại gần đây:\n{messages_summary}\n\n"
        "Bạn điều phối hội thoại bằng cách CHỌN MỘT công cụ cho lượt này. Trả về JSON với field 'tool' "
        f"là một trong: {tool_menu}.\n"
        "- ask_user: hỏi người dùng một câu làm rõ — kèm field 'message'.\n"
        "- write_draft: đề xuất bản nháp artifact để duyệt — kèm 'title' và 'body'.\n"
        "- write_note: ghi chú phân tích/phản biện vào scratchpad (không cần duyệt) — kèm 'content'.\n"
        "- finalize: chốt phiên — kèm 'summary' (chỉ khả dụng khi đã có draft).\n"
        "Luôn kèm 'active_mode' (qa/critique/explore/draft) và 'confidence' (0-1). Khi đã rõ đủ, cập nhật "
        "draft qua field 'draft_update' (bồi đắp tăng dần, không bịa nội dung chưa có). Không lặp lại câu "
        "hỏi đã hỏi."
        f"{_build_synthesis_directive('dùng write_draft')}"
        f"{_build_slot_directive(state)}"
        f"{_build_coverage_hint(state)}"
        f"{draft_block}"
        f"{working_draft_block}"
        f"{_build_mode_directive(state)}"
        f"{language_lock}"
    )


def _build_mode_directive(state: WorkflowState) -> str:
    """Steer the analyst's operating angle (multi-angle S1/S2).

    A user-supplied `mode_hint` is an explicit "cướp lái" — switch to that mode now. With no
    hint, nudge the agent to proactively leave plain Q&A once enough is clarified, so it chains
    into critique/explore instead of only ever asking. The chosen angle is reported back via the
    optional `active_mode` field, which the eval layer counts as proactive coverage.
    """
    mode_hint = (state.get("mode_hint") or "").strip()
    if mode_hint:
        return (
            f"\n\nYÊU CẦU MODE: người dùng muốn chuyển sang chế độ '{mode_hint}'. Hãy chuyển ngay "
            f"trong lượt này, đặt active_mode='{mode_hint}' và phản hồi đúng theo chế độ đó."
        )
    return (
        "\n\nGỢI Ý CHẾ ĐỘ (chủ động): sau khi đã có ≥2 lượt làm rõ hoàn chỉnh, hãy chủ động chuyển "
        "active_mode sang 'critique' (soi điểm yếu/giả định) hoặc 'explore' (mở rộng góc nhìn) "
        "thay vì chỉ tiếp tục hỏi, và đặt tên chế độ trong field active_mode."
    )


def _build_synthesis_directive(trigger: str = "dùng write_draft") -> str:
    """Instruct the LLM to synthesize a rich artifact body when drafting.

    Without this the model emits a thin, one-paragraph body even after thorough elicitation. This
    block tells it to mine the whole conversation/context into detailed, structured content — while
    forbidding fabrication. `trigger` names the action that fires it (write_draft in the tool-loop).
    """
    return (
        f"\n\nĐỘ SÂU NỘI DUNG (khi {trigger}): body của mỗi proposal phải KHAI THÁC "
        "toàn bộ thông tin user đã cung cấp trong hội thoại và context, viết chi tiết và có cấu "
        "trúc rõ ràng phù hợp với loại artifact (các phần/mục liên quan, dữ kiện cụ thể, ràng buộc, "
        "tiêu chí, ví dụ user đã nêu). KHÔNG tóm tắt sơ sài thành một đoạn ngắn, KHÔNG lặp lại câu "
        "hỏi. Mỗi ý user đã trả lời phải được triển khai thành nội dung thực chất. Tuyệt đối KHÔNG "
        "bịa thông tin chưa được cung cấp — chỉ đào sâu từ những gì đã thu thập; phần thật sự thiếu "
        "thì để ngỏ, không bù bằng nội dung bịa."
    )


def _build_slot_directive(state: WorkflowState) -> str:
    """For BRD artifact types, instruct the LLM to always report slot_assessment.

    Without this the gate is a no-op in production: slot_assessment is optional in the
    schema, so an LLM that omits it never gets coverage computed (None -> fail-open).
    """
    artifact_type = state.get("artifact_type") or ""
    slot_spec = BRD_SLOTS.get(artifact_type)
    if not slot_spec:
        return ""
    slot_lines = "\n".join(
        f"- {slot}: {SLOT_DESCRIPTIONS.get(slot, slot)}"
        for slot in slot_spec["required"]
    )
    return (
        f"\n\nĐÁNH GIÁ ĐỘ PHỦ: với loại '{artifact_type}', LUÔN trả field slot_assessment — object chấm "
        "trạng thái TỪNG slot bắt buộc dưới đây dựa trên TOÀN BỘ thông tin user đã cung cấp:\n"
        f"{slot_lines}\n"
        "Quy tắc chấm: 'filled' khi user đã cung cấp thông tin rõ ràng cho slot đó; 'partial' khi mới có "
        "một phần hoặc còn mơ hồ; 'empty' khi chưa có gì. Nếu câu trả lời mới nhất của user đã đáp ứng "
        "một slot, PHẢI nâng slot đó lên 'filled' hoặc 'partial' — tuyệt đối không giữ nguyên 'empty'. "
        "Đây là rubric tham chiếu để bạn tự đánh giá độ đầy đủ, KHÔNG phải checklist hỏi tuần tự: bạn tự "
        "quyết slot nào nên khai thác tiếp theo dựa trên mạch hội thoại, và tự quyết khi nào đã đủ để propose."
    )


def _pick_weak_slot(slot_coverage: dict[str, str], required_slots: list[str], exclude: str | None = None) -> str | None:
    """First 'empty' (then 'partial') required slot, skipping `exclude`.

    `exclude` is the slot the previous turn already asked about; skipping it stops the
    hint re-pinning the same slot two turns running when the model under-grades.
    """
    for status in ("empty", "partial"):
        for slot in required_slots:
            if slot != exclude and slot_coverage.get(slot) == status:
                return slot
    return None


def _coverage_hint_target(state: WorkflowState) -> str | None:
    """Required slot to exclude from this turn's gap inventory, or None when none applies.

    analyze_node persists this as last_asked_slot; _build_coverage_hint drops it from the gap
    list so the same slot is not re-offered two turns running (best-effort anti-repeat — the
    LLM may still pick a different listed slot). None means coverage is OK/untracked, the loop
    has stalled, or the only weak slot left is the one asked last turn, in which case the hint
    steers the model to move on instead of repeating.
    """
    if state.get("coverage_complete") is not False:
        return None
    if (state.get("coverage_stall_count") or 0) >= COVERAGE_STALL_LIMIT:
        return None
    artifact_type = state.get("artifact_type") or ""
    slot_coverage = state.get("slot_coverage") or {}
    required_slots = (BRD_SLOTS.get(artifact_type) or {}).get("required") or []
    return _pick_weak_slot(slot_coverage, required_slots, exclude=state.get("last_asked_slot"))


def _build_coverage_hint(state: WorkflowState) -> str:
    if state.get("coverage_complete") is not False:
        return ""
    artifact_type = state.get("artifact_type")
    weak_slot = _coverage_hint_target(state)
    # weak_slot is None when the loop stalled OR the only slot still weak is the one we just
    # asked — either way re-pinning would reproduce the previous question verbatim, so steer
    # the model to synthesize what it has and move on or propose.
    if weak_slot is None:
        return (
            f"\n\nCoverage '{artifact_type}': độ phủ không tăng sau nhiều lượt hỏi. Đừng lặp lại cùng câu "
            "hỏi — hãy tổng hợp thông tin đã có và chuyển sang propose, hoặc hỏi một góc độ hoàn toàn khác."
        )
    # Gap-inventory: list every weak slot (empty first, then partial) so the LLM picks the angle
    # that fits the conversation instead of being pinned to one scripted question. last_asked_slot
    # is excluded for best-effort anti-repeat — see _coverage_hint_target.
    slot_coverage = state.get("slot_coverage") or {}
    required_slots = (BRD_SLOTS.get(artifact_type) or {}).get("required") or []
    last_asked = state.get("last_asked_slot")
    gap_lines = [
        f"- {SLOT_DESCRIPTIONS.get(slot, slot)} ({status})"
        for status in ("empty", "partial")
        for slot in required_slots
        if slot != last_asked and slot_coverage.get(slot) == status
    ]
    inventory = "\n".join(gap_lines)
    return (
        f"\n\nCoverage '{artifact_type}': các khía cạnh còn thiếu hoặc chưa rõ (rubric tham chiếu, "
        "không phải thứ tự bắt buộc):\n"
        f"{inventory}\n"
        "Tự chọn angle phù hợp nhất với mạch hội thoại để khai thác tiếp — không nhất thiết theo thứ tự "
        "trên. Không được trả lời cụt chỉ bằng câu hỏi; hãy có một câu dẫn dắt ngắn, rồi đặt một câu hỏi chính."
    )


def _build_summary_prompt(state: WorkflowState) -> str:
    current_summary = (state.get("conversation_summary") or "").strip() or "(chưa có)"
    recent_messages = "\n".join(
        f"{role}: {content}"
        for role, content in (_msg_role_content(m) for m in (state.get("messages") or [])[-settings.summary_trigger_every:])
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
