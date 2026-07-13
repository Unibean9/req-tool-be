"""Interaction tools — ask_user / confirm_intent / respond (interrupting, conversational).

These pause the tool loop for the user: ask_user gathers information, confirm_intent gates the
intent→artifact transition, respond delivers a proactive assessment. All three keep the session
ACTIVE with a stream_response interrupt (not the approval-gate WAITING state) and audit best-effort
to AgentToolCall.

The audit call is routed through the coordinator module reference (agent_tools._audit_...) so a test
that patches `app.graphs.agent_tools._audit_interaction_tool_call` still intercepts it after the
split. The interrupt itself goes through interrupts._save_and_interrupt_ask (a module attribute, so
patching app.graphs.interrupts._save_and_interrupt_ask keeps working unchanged).
"""

import logging
import uuid
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from sqlalchemy import exists, select

from app.graphs import agent_tools, interrupts
from app.graphs.agent_tools._shared import _missing_required_arg_update, _tool_not_available_update
from app.graphs.state import WorkflowState
from app.models.agent import AgentToolCall, AgentToolCallStatus

logger = logging.getLogger(__name__)


async def _audit_interaction_tool_call(
    state: WorkflowState,
    config: RunnableConfig,
    *,
    tool_name: str,
    message: str,
) -> None:
    """Best-effort AgentToolCall row for interaction tools (ask_user/respond/confirm_intent)."""
    run_id_raw = state.get("last_agent_run_id")
    if not run_id_raw:
        return
    try:
        session_factory = config["configurable"]["session_factory"]
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(
                        exists().where(
                            AgentToolCall.run_id == uuid.UUID(str(run_id_raw)),
                            AgentToolCall.tool_name == tool_name,
                        )
                    )
                )
            ).scalar()
            if not already:
                db.add(
                    AgentToolCall(
                        run_id=uuid.UUID(str(run_id_raw)),
                        tool_name=tool_name,
                        input_snapshot={"message": message},
                        status=AgentToolCallStatus.EXECUTED,
                    )
                )
                await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("interaction tool audit persist failed (%s): %s", tool_name, exc)


# Batched question form: up to 3 related, typed facets in one interrupt.
_MAX_BATCH_QUESTIONS = 3
_BATCH_QUESTION_TYPES = frozenset({"choice", "text", "confirm"})


def _normalize_batch_questions(questions: Any) -> list[dict[str, Any]]:
    """Keep at most 3 well-formed questions; drop malformed entries rather than erroring the turn.

    A valid entry has a non-empty `prompt` and a `type` in {choice, text, confirm} (defaulting to
    `text`); `choice` entries keep their string `options`. Anything else is silently dropped so a
    partially-malformed batch still asks its good questions instead of failing the whole tool call.
    """
    if not isinstance(questions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        q_type = str(item.get("type") or "text").strip().lower()
        if q_type not in _BATCH_QUESTION_TYPES:
            q_type = "text"
        entry: dict[str, Any] = {"prompt": prompt, "type": q_type}
        if q_type == "choice":
            entry["options"] = [str(opt).strip() for opt in (item.get("options") or []) if str(opt).strip()]
        normalized.append(entry)
        if len(normalized) >= _MAX_BATCH_QUESTIONS:
            break
    return normalized


def _render_batched_question_text(message: str, questions: list[dict[str, Any]]) -> str:
    """Joined-text fallback: the header (if any) followed by each numbered question, so a client
    that renders only free text still surfaces every facet."""
    lines: list[str] = []
    header = str(message or "").strip()
    if header:
        lines.append(header)
    for index, question in enumerate(questions, start=1):
        line = f"{index}. {question['prompt']}"
        if question["type"] == "choice" and question.get("options"):
            line += f" ({' / '.join(question['options'])})"
        lines.append(line)
    return "\n".join(lines)


async def _ask_user_impl(
    message: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
    questions: Any = None,
):
    batch = _normalize_batch_questions(questions)
    if not str(message or "").strip() and not batch:
        return _missing_required_arg_update("ask_user", "message", tool_call_id, state.get("locale"))
    content = _render_batched_question_text(message, batch) if batch else message

    # ToolCall.id is the correct idempotency key here: inside the ToolNode body
    # state["last_agent_run_id"] still belongs to the prior analyze_node, not this invocation.
    # interrupt_kind="stream_response": session stays ACTIVE so the conversation resume path applies
    # (not the approval-gate path). The graph still halts via interrupt() — only the DB fields differ.
    await agent_tools._audit_interaction_tool_call(state, config, tool_name=f"ask_user:{tool_call_id}", message=content)
    user_content = await interrupts._save_and_interrupt_ask(
        state, config, content, run_id=tool_call_id, interrupt_kind="stream_response", questions=batch or None
    )
    return Command(
        update={
            "messages": [
                ToolMessage(content=user_content, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ]
        }
    )


@tool
async def ask_user(
    message: Annotated[
        str,
        "A short header/lead-in, written in the user's locale. May be empty when `questions` is given.",
    ],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    questions: Annotated[
        list[dict] | None,
        "Optional batch of up to 3 RELATED questions asked in one turn. Each item: "
        "{prompt: str, type: 'choice'|'text'|'confirm', options?: [str] for choice}. Batch related "
        "facets of one topic; ask serially (single question, empty list) only when one answer "
        "determines the next question.",
    ] = None,
) -> Command:
    """Ask the user for information and pause for their reply.

    Use when you need information you do not have and cannot reasonably infer. Do NOT use to deliver
    an opinion or assessment (use respond) or to present a prepared draft (use write_draft /
    confirm_intent). Keep to ONE topic: either one focused question, or up to 3 related facets of
    the same topic via `questions` — never an open-ended checklist.
    """
    return await _ask_user_impl(message, state, config, tool_call_id, questions)


async def _confirm_intent_impl(
    summary: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
) -> Command:
    if not str(summary or "").strip():
        return _missing_required_arg_update("confirm_intent", "summary", tool_call_id, state.get("locale"))
    if state.get("user_confirmed") is not None:
        return _tool_not_available_update(
            "confirm_intent",
            "intent is already confirmed; use ask_user/respond/write_draft for the current phase.",
            tool_call_id,
            state.get("locale"),
        )

    # interrupt_kind="stream_response" keeps the session ACTIVE (D4): the user can reply, and the
    # next turn sees user_confirmed=True — which unlocks the artifact tool menu in get_available_tools.
    # kind="assessment": this is a surfaced intent summary, not a clarifying question.
    await agent_tools._audit_interaction_tool_call(
        state, config, tool_name=f"confirm_intent:{tool_call_id}", message=summary
    )
    user_content = await interrupts._save_and_interrupt_ask(
        state, config, summary, run_id=tool_call_id, kind="assessment", interrupt_kind="stream_response"
    )
    return Command(
        update={
            "user_confirmed": True,
            "messages": [
                ToolMessage(content=user_content, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ],
        }
    )


@tool
async def confirm_intent(
    summary: Annotated[
        str,
        "A short restatement of the user's goal/intent for them to confirm or correct, in their locale.",
    ],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Present a concise summary of the user's intent and pause for confirmation.

    Use in the intent phase, before any artifact work, to confirm you understood what they want to
    build. This is the one-shot gate into the artifact phase. Not for clarifying questions (use
    ask_user) and not for presenting a full draft (use write_draft).
    """
    return await _confirm_intent_impl(summary, state, config, tool_call_id)


async def _respond_impl(message: str, mode: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str):
    if not str(message or "").strip():
        return _missing_required_arg_update("respond", "message", tool_call_id, state.get("locale"))

    # Reuses the ask_user persist+interrupt path (idempotency keyed on ToolCall.id). respond is a
    # conversational pause like ask_user/confirm_intent, so it keeps the session ACTIVE with a
    # STREAM_RESPONSE interrupt instead of entering the approval-gate WAITING state.
    await agent_tools._audit_interaction_tool_call(state, config, tool_name=f"respond:{tool_call_id}", message=message)
    user_content = await interrupts._save_and_interrupt_ask(
        state, config, message, run_id=tool_call_id, kind="assessment", mode=mode, interrupt_kind="stream_response"
    )
    return Command(
        update={
            "messages": [
                ToolMessage(content=user_content, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ]
        }
    )


@tool
async def respond(
    message: Annotated[str, "The assessment to deliver, in the user's locale — a complete thought, not a question."],
    mode: Annotated[str, "Operating angle: 'critique' or 'structuring'."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Share an assessment with the user — a critique or exploration, NOT a question — and pause for their reaction.

    Use to deliver a proactive opinion or analysis instead of phrasing every turn as a question. Use
    ask_user when you actually need an answer; use the note tools to think without interrupting.
    """  # noqa: E501
    return await _respond_impl(message, mode, state, config, tool_call_id)
