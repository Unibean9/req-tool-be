"""Native tools wrapping the enum branches (Phase 3 parity wrap).

`ask_user`, `write_draft` and `finalize` mirror the `ask` / `propose` / `done` enum branches as
LangGraph tools dispatched by the parallel ToolNode. The enum branches stay live alongside them
(removed only in Phase 5). Each tool is a thin `@tool` over a plain async impl so the impls stay
unit-testable without a Runtime.

Idempotency on resume — LangGraph re-executes a ToolNode body from the top when its interrupt is
resumed: ask_user keys its message insert on the per-invocation ToolCall.id; write_draft keys its
proposal row on (run_id, tool_name), reusing the existing AgentToolCall.tool_name column (no
migration). finalize has no insert to dedup — its only DB write is an idempotent-by-value session
status update — so it needs no key.
"""

import uuid
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from sqlalchemy import exists, select

from app.graphs import nodes
from app.graphs.state import WorkflowState
from app.models.agent import (
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)


# ---------------------------------------------------------------------------
# ask_user — parity for the `ask` enum branch
# ---------------------------------------------------------------------------

async def _ask_user_impl(message: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str):
    # ToolCall.id is the correct idempotency key here: inside the ToolNode body
    # state["last_agent_run_id"] still belongs to the prior analyze_node, not this invocation.
    user_content = await nodes._save_and_interrupt_ask(state, config, message, run_id=tool_call_id)
    return Command(
        update={
            "messages": [
                ToolMessage(content=message, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ]
        }
    )


@tool
async def ask_user(
    message: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Hỏi người dùng một câu hỏi làm rõ và tạm dừng để chờ trả lời."""
    return await _ask_user_impl(message, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# write_draft — parity for the `propose` enum branch
# ---------------------------------------------------------------------------

async def _write_draft_impl(
    title: str, body: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str
):
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    if not state.get("last_agent_run_id"):
        raise RuntimeError("write_draft requires last_agent_run_id in state — analyze_node must run first")
    run_id = uuid.UUID(state["last_agent_run_id"])

    async with session_factory() as db:
        # Idempotency on (run_id, tool_name): a resume re-executes this body, so skip if the
        # proposed write already exists for this run. tool_name discriminates it from the enum
        # path's "create_artifact" rows — no new column, no migration (R3).
        already = (
            await db.execute(
                select(exists().where(
                    AgentToolCall.run_id == run_id,
                    AgentToolCall.tool_name == "write_draft",
                ))
            )
        ).scalar()
        if not already:
            db.add(
                AgentToolCall(
                    run_id=run_id,
                    tool_name="write_draft",
                    input_snapshot={
                        "artifact_type": state["artifact_type"],
                        "title": title,
                        "body": body,
                    },
                    status=AgentToolCallStatus.PROPOSED,
                )
            )
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == session_id))
        ).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.PROPOSE_ARTIFACTS
        await db.commit()

    interrupt({"type": "propose_artifacts", "tool_name": "write_draft"})
    return Command(update={"messages": [ToolMessage(content=title, tool_call_id=tool_call_id)]})


@tool
async def write_draft(
    title: str,
    body: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Đề xuất một bản nháp artifact và tạm dừng để người dùng duyệt."""
    return await _write_draft_impl(title, body, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# finalize — parity for the `done` enum branch, with a HITL confirmation gate
# ---------------------------------------------------------------------------

async def _finalize_impl(summary: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str):
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])

    async with session_factory() as db:
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == session_id))
        ).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.ASK_HUMAN
        await db.commit()

    interrupt({"type": "finalize", "message": summary})
    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})


@tool
async def finalize(
    summary: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Chốt phiên làm việc và tạm dừng để người dùng xác nhận kết thúc."""
    return await _finalize_impl(summary, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# write_note — explore/critique scratchpad (no interrupt, no DB, no approval)
# ---------------------------------------------------------------------------

async def _write_note_impl(content: str, tool_call_id: str):
    # The note lives in the message history (decision 3): no `notes` state field, no DB row. It
    # is pure working memory the analyst re-reads next turn, so there is no side-effect to dedup.
    return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]})


@tool
async def write_note(
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Ghi một note phân tích/phản biện vào scratchpad trước khi hỏi hoặc draft (không cần duyệt)."""
    return await _write_note_impl(content, tool_call_id)


# ---------------------------------------------------------------------------
# get_available_tools — state-driven gate over the tool-loop
# ---------------------------------------------------------------------------

# After this many consecutive write_note turns the loop must ask_user/write_draft instead of
# noting again — the only guard against an infinite note loop (S4). Tune via T8 once the loop is
# wired (Phase 5); start at 3.
NOTE_STEP_LIMIT = 3


def _tool_call_names(message) -> list[str]:
    """Tool names an AIMessage selected this turn; [] for any other message."""
    tool_calls = getattr(message, "tool_calls", None) or []
    return [tc["name"] for tc in tool_calls if isinstance(tc, dict) and tc.get("name")]


def _consecutive_write_notes(messages: list) -> int:
    """Count write_note turns since the last ask_user/write_draft — derived from history (N2).

    Counts per turn (per AIMessage), not per call: the limit is "N consecutive note turns", so a
    turn that batches two write_note calls is still one turn against the step-limit.
    """
    count = 0
    for message in reversed(messages or []):
        names = _tool_call_names(message)
        if not names:
            continue
        if any(name in ("ask_user", "write_draft") for name in names):
            break
        if "write_note" in names:
            count += 1
    return count


def get_available_tools(state: WorkflowState) -> list:
    """Tools the loop may pick this turn, gated on state.

    - `finalize` only once `working_draft` is non-empty (the single hard-gate; absent/None/blank
      → CLOSED, never crashes).
    - `write_note` dropped after NOTE_STEP_LIMIT consecutive notes, forcing ask_user/write_draft.
    """
    tools = [ask_user, write_draft, write_note]
    if (state.get("working_draft") or "").strip():
        tools.append(finalize)
    if _consecutive_write_notes(state.get("messages")) >= NOTE_STEP_LIMIT:
        tools = [t for t in tools if t.name != "write_note"]
    return tools
