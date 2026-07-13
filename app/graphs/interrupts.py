"""Interrupt-raising helpers shared by graph nodes and agent tools.

Neutral leaf module: imports only state, models, and LangGraph primitives — never nodes or
agent_tools — so both can depend on it without a circular import.
"""

import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from sqlalchemy import exists, select

from app.graphs.state import WorkflowState
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
)


async def _save_and_interrupt_ask(
    state: WorkflowState,
    config: RunnableConfig,
    content: str,
    *,
    run_id,
    kind: str = "question",
    mode: str | None = None,
    interrupt_kind: str = "ask_human",
    questions: list[dict[str, Any]] | None = None,
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
    # Batched question form: the structured list rides alongside `content`, which already
    # carries the joined-text fallback so a client that ignores `questions` still shows every facet.
    if questions:
        payload["questions"] = questions

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
    if questions:
        interrupt_payload["questions"] = questions
    user_response = interrupt(interrupt_payload)
    return _resume_answer_text(user_response, questions)


def _resume_answer_text(user_response: Any, questions: list[dict[str, Any]] | None) -> str:
    """Normalize a resume reply into the text the model reads.

    Free text (or a ``{"content": ...}`` dict) passes through unchanged — the legacy path. When the
    client answers a batched interrupt with a structured ``{"answers": [...]}`` list, pair each
    answer with its question prompt into one combined block so the model sees which answer maps to
    which question; anything unpairable falls back to the raw text.
    """
    if isinstance(user_response, dict):
        answers = user_response.get("answers")
        if isinstance(answers, list) and questions:
            paired = [
                f"- {str(q.get('prompt') or '').strip()}: {str(answer).strip()}"
                for q, answer in zip(questions, answers, strict=False)
                if str(answer).strip()
            ]
            if paired:
                return "\n".join(paired)
        return str(user_response.get("content", "") or "")
    return str(user_response or "")


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
