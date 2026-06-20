import time
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from sqlalchemy import exists, select

from app.config import settings
from app.graphs.personas import get_persona
from app.graphs.policy import ApprovalRequired, GovernanceDenied
from app.graphs.state import WorkflowState
from app.graphs.tools import create_artifact, read_artifacts
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "next_action": {"type": "string", "enum": ["ask", "propose", "done"]},
        "confidence": {"type": "number"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "message": {"type": "string"},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "artifact_type": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["artifact_type", "title", "body"],
            },
        },
    },
    "required": ["next_action", "confidence"],
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

    async with session_factory() as db:
        artifacts = await read_artifacts(
            db=db,
            project_id=project_id,
            artifact_type=state["artifact_type"],
            context={"workflow_area": state["workflow_area"]},
        )

    prompt = _build_analyst_prompt(state, artifacts)
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
        response_format=ANALYSIS_SCHEMA,
    )
    latency_ms = int((time.monotonic() - started_at) * 1000)

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

    return {
        "analysis_result": analysis_result,
        "turn_count": state["turn_count"] + 1,
        "last_agent_run_id": run_id,
    }


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

    action = (state.get("analysis_result") or {}).get("next_action", "done")
    if action == "ask":
        return "ask_human"
    if action == "propose":
        return "confirm"
    return END


async def ask_human_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])

    message = (state.get("analysis_result") or {}).get("message", "")
    if not message:
        raise ValueError("ask_human_node: LLM returned next_action='ask' but no message field")

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
            db.add(AgentMessage(session_id=session_id, role=AgentMessageRole.AGENT, content=message))
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


async def propose_artifacts_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])

    proposals = (state.get("analysis_result") or {}).get("proposals", [])
    if not state.get("last_agent_run_id"):
        raise RuntimeError(
            "propose_artifacts_node requires last_agent_run_id in state — analyze_node must run first"
        )
    run_id = uuid.UUID(state["last_agent_run_id"])
    tool_call_ids: list[str] = []

    session_artifact_type = state["artifact_type"]

    async with session_factory() as db:
        for proposal in proposals:
            # Dùng artifact_type của session, không phụ thuộc vào giá trị LLM trả về
            # vì LLM có thể trả về type không hợp lệ (vd: "brd" thay vì "goal")
            artifact_type = session_artifact_type
            try:
                await create_artifact(
                    artifact_type=artifact_type,
                    title=proposal.get("title", ""),
                    body=proposal.get("body", ""),
                    rationale=proposal.get("rationale", ""),
                    context={"allowed_types": [artifact_type]},
                )
            except ApprovalRequired as exc:
                tool_call = AgentToolCall(
                    run_id=run_id,
                    tool_name=exc.tool_name,
                    input_snapshot=exc.args_snapshot,
                    status=AgentToolCallStatus.PROPOSED,
                )
                db.add(tool_call)
                await db.flush()
                tool_call_ids.append(str(tool_call.id))
            except GovernanceDenied:
                continue

        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == session_id))
        ).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.PROPOSE_ARTIFACTS
        await db.commit()

    interrupt({"type": "propose_artifacts", "tool_call_ids": tool_call_ids})
    return {"pending_tool_call_ids": tool_call_ids}


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


def _build_analyst_prompt(state: WorkflowState, artifacts: list[dict]) -> str:
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

    return (
        f"Bạn là BA/PM analyst. Phân tích và đề xuất artifact cho loại: {state['artifact_type']}.\n\n"
        f"Context hiện tại:\n{artifact_context}\n\n"
        f"Hội thoại gần đây:\n{messages_summary}\n\n"
        "Trả về JSON với next_action (ask/propose/done), confidence (0-1), gaps, proposals (nếu propose). "
        "Nếu next_action='ask', bắt buộc có field message (string câu hỏi cụ thể gửi cho user). "
        "Lưu ý: nếu user vừa từ chối tạo artifact và yêu cầu khám phá thêm, hãy tiếp tục hỏi các góc độ "
        "chưa được đề cập thay vì đề xuất lại ngay."
        f"{language_lock}"
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


_YES_KEYWORDS = {"có", "yes", "đồng ý", "ok", "oke", "okay", "tạo", "được", "tạo đi", "create", "proceed", "go"}


def _is_affirmative(text: str) -> bool:
    tokens = set(text.lower().split())
    return bool(tokens & _YES_KEYWORDS)


async def confirm_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])

    artifact_type = state["artifact_type"]
    message = (
        f"Tôi đã có đủ thông tin để tạo **{artifact_type}**. "
        "Bạn có muốn tôi tiến hành tạo không?\n\n"
        "Nếu chưa, hãy cho tôi biết góc độ nào bạn muốn khám phá thêm."
    )

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
            db.add(AgentMessage(session_id=session_id, role=AgentMessageRole.AGENT, content=message))
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == session_id))
        ).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.ASK_HUMAN
        await db.commit()

    user_response = interrupt({"type": "ask_human", "message": message})
    user_content = user_response.get("content", "") if isinstance(user_response, dict) else str(user_response or "")
    confirmed = _is_affirmative(user_content)

    return {
        "messages": [
            {"role": "assistant", "content": message},
            {"role": "user", "content": user_content},
        ],
        "user_confirmed": confirmed,
    }


def route_after_confirm(state: WorkflowState) -> str:
    if state.get("user_confirmed"):
        return "propose_artifacts"
    return "analyze"
