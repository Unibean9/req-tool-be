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

import logging
import uuid
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from sqlalchemy import exists, select

from app.graphs import nodes
from app.graphs.note_parser import extract_structured_objects
from app.graphs.section_schema import SECTION_SPECS, status_score
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
    """Ask the user a clarifying question and pause to wait for their reply."""
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
    """Propose an artifact draft and pause for the user to review it."""
    return await _write_draft_impl(title, body, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# finalize — parity for the `done` enum branch, with a HITL confirmation gate
# ---------------------------------------------------------------------------

async def _finalize_impl(summary: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str):  # noqa: ARG001 — state kept for signature parity with sibling tool impls
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
    """Finalize the working session and pause for the user to confirm completion."""
    return await _finalize_impl(summary, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# critique_note / explore_note — mode-bearing scratchpad notes
# (no interrupt, no DB, no approval)
# ---------------------------------------------------------------------------
# Splitting the former single write_note into two named angles makes the analytical move a
# first-class menu choice and lets analyze_node derive `active_mode` from the tool picked, so
# proactive S1 coverage no longer depends on the model self-reporting active_mode.

async def _write_note_impl(content: str, state: WorkflowState, tool_call_id: str):
    # The note text lives in the message history (decision 3): no `notes` state field, no DB row.
    # Beyond that, tagged lines (ASSUMPTION:/RISK:/OPEN_QUESTION:) are parsed into structured state
    # objects and appended to the accumulating lists so validators and the finalize gate can query
    # them. Append (prior + new) since these channels have no reducer.
    extracted = extract_structured_objects(content)
    update: dict[str, Any] = {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    for bucket in ("assumptions", "risks", "open_questions"):
        if extracted[bucket]:
            update[bucket] = [*(state.get(bucket) or []), *extracted[bucket]]
    return Command(update=update)


@tool
async def critique_note(
    content: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Critique note: probe weaknesses, risky assumptions, or contradictions in the current information (no approval needed)."""  # noqa: E501
    return await _write_note_impl(content, state, tool_call_id)


@tool
async def explore_note(
    content: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Exploration note: broaden the perspective, raise angles or options not yet considered (no approval needed).

    Its active_mode maps to 'structuring' after the spec §7.1 migration (see phase-06).
    """
    return await _write_note_impl(content, state, tool_call_id)


# ---------------------------------------------------------------------------
# respond — user-facing critique/exploration (mode-bearing, interrupting)
# ---------------------------------------------------------------------------
# The note tools are silent scratchpad; respond is the outward voice for a non-question turn. It
# lets the analyst deliver a critique or an exploration TO the user and pause for their reaction,
# so the agent is not forced to phrase every proactive turn as an ask_user (the Q&A-bias fix).

async def _respond_impl(message: str, mode: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str):
    # Reuses the ask_user persist+interrupt path (idempotency keyed on ToolCall.id, ASK_HUMAN
    # interrupt_type so the resume accepts a free-text reply); only the message kind and the carried
    # mode differ, so the user sees an assessment rather than a question.
    user_content = await nodes._save_and_interrupt_ask(
        state, config, message, run_id=tool_call_id, kind="assessment", mode=mode
    )
    return Command(
        update={
            "messages": [
                ToolMessage(content=message, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ]
        }
    )


@tool
async def respond(
    message: str,
    mode: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Share an assessment with the user — a critique or exploration, not a question — and pause for their reaction."""  # noqa: E501
    return await _respond_impl(message, mode, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# run_critique — formal judge call over the current draft (mode-targeted, non-interrupting)
# ---------------------------------------------------------------------------
# Unlike critique_note (silent scratchpad), run_critique invokes the production judge in
# critique.py, records a quality_report, and increments critique_rounds. It does not interrupt —
# the analyst surfaces the result to the user via `respond` on a later turn.

async def _run_critique_impl(
    target: str,  # noqa: ARG001 — kept for schema parity; the judge scores the loaded draft body
    mode: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    from app.graphs.critique import _invoke_judge

    cfg = config["configurable"]
    llm_client = cfg.get("strong_llm_client") or cfg.get("llm_client")
    body = state.get("draft_body") or state.get("working_draft") or ""
    report = await _invoke_judge(body, mode, llm_client)
    summary = f"critique[{report['mode']}] score={report['score']:.2f}"
    return Command(
        update={
            "quality_report": report,
            "critique_rounds": (state.get("critique_rounds") or 0) + 1,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def run_critique(
    target: str,
    mode: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Run a formal quality critique over the current draft along one mode and record the report."""
    return await _run_critique_impl(target, mode, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# recommend_next_workflow — read-only analysis tool (no interrupt; audits to AgentToolCall)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# First four sections define a product brief; all seven define a PRD (addendum §6).
_BRIEF_SECTIONS = ("vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities")


def _compute_recommendation(section_coverage: dict[str, str] | None, planning_track: str) -> dict[str, Any]:
    """Pure workflow-selection rule over 7-section coverage. No DB, no state mutation.

    Derives the artifact chain inline from the latest section_coverage (never the possibly-stale
    state["artifact_chain"]). On the quick track, never escalates past readiness_check.
    """
    cov = section_coverage or {}
    scores = {section: status_score(cov.get(section)) for section in SECTION_SPECS}
    brief_score = sum(scores[s] for s in _BRIEF_SECTIONS) / len(_BRIEF_SECTIONS)
    prd_score = sum(scores.values()) / len(scores)
    missing = [section for section, score in scores.items() if score == 0.0]

    if prd_score >= 0.7:
        recommended, reason = "readiness_check", "PRD coverage is near-complete; assess readiness next."
    elif brief_score >= 0.6:
        recommended, reason = "prd", "Product-brief sections are solid; expand into a PRD."
    else:
        recommended, reason = "brief", "Early signal captured; consolidate into a product brief."

    # Quick track guards against over-planning a small idea.
    if planning_track == "quick" and recommended == "architecture_readiness":
        recommended = "readiness_check"

    missing_count = len(missing)
    confidence = "low" if missing_count >= 4 else ("medium" if missing_count >= 1 else "high")

    return {
        "recommended_next_workflow": recommended,
        "reason": reason,
        "required_inputs": list(missing),
        "blocking_gaps": list(missing),
        "confidence": confidence,
    }


async def _recommend_next_workflow_impl(
    current_artifact_type: str,  # noqa: ARG001 — kept for schema parity; recommendation is coverage-driven
    planning_track: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    result = _compute_recommendation(state.get("section_coverage"), planning_track or "quick")

    # Audit: reuse AgentToolCall.input_snapshot for the result blob (no output_snapshot column).
    # Best-effort — a DB failure must not deny the recommendation to the user.
    if not state.get("last_agent_run_id"):
        raise RuntimeError(
            "recommend_next_workflow requires last_agent_run_id in state — analyze_node must run first"
        )
    run_id = uuid.UUID(state["last_agent_run_id"])
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    try:
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(exists().where(
                        AgentToolCall.run_id == run_id,
                        AgentToolCall.tool_name == "recommend_next_workflow",
                    ))
                )
            ).scalar()
            if not already:
                db.add(
                    AgentToolCall(
                        run_id=run_id,
                        tool_name="recommend_next_workflow",
                        input_snapshot=result,
                        status=AgentToolCallStatus.PROPOSED,
                    )
                )
                await db.commit()
    except Exception as exc:  # noqa: BLE001 — audit is best-effort; never block the recommendation
        logger.warning("recommend_next_workflow audit persist failed: %s", exc)

    method_profile = dict(state.get("method_profile") or {})
    method_profile["recommended_next_workflow"] = result["recommended_next_workflow"]
    summary = f"recommend_next_workflow -> {result['recommended_next_workflow']} ({result['confidence']})"
    return Command(
        update={
            "method_profile": method_profile,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def recommend_next_workflow(
    current_artifact_type: str,
    planning_track: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Recommend the next planning workflow from current coverage; records an audit entry."""
    return await _recommend_next_workflow_impl(current_artifact_type, planning_track, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# run_readiness_check — 10-dimension readiness assessment (no interrupt; audits to AgentToolCall)
# ---------------------------------------------------------------------------

async def _run_readiness_check_impl(
    target: str,  # noqa: ARG001 — kept for schema parity; the check is coverage-driven
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    from app.graphs.readiness import compute_readiness_score

    report = compute_readiness_score(state.get("section_coverage"), state)

    if not state.get("last_agent_run_id"):
        raise RuntimeError(
            "run_readiness_check requires last_agent_run_id in state — analyze_node must run first"
        )
    run_id = uuid.UUID(state["last_agent_run_id"])
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    try:
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(exists().where(
                        AgentToolCall.run_id == run_id,
                        AgentToolCall.tool_name == "run_readiness_check",
                    ))
                )
            ).scalar()
            if not already:
                db.add(
                    AgentToolCall(
                        run_id=run_id,
                        tool_name="run_readiness_check",
                        input_snapshot=report,
                        status=AgentToolCallStatus.PROPOSED,
                    )
                )
                await db.commit()
    except Exception as exc:  # noqa: BLE001 — audit is best-effort; never block the check
        logger.warning("run_readiness_check audit persist failed: %s", exc)

    readiness = dict(state.get("readiness") or {})
    readiness["requirements_ready"] = report["ready"]
    readiness["blocking_gaps"] = report["blocking_gaps"]
    readiness["recommended_next_step"] = report["recommended_next_step"]
    summary = f"run_readiness_check -> ready={report['ready']} score={report['readiness_score']:.2f}"
    return Command(
        update={
            "readiness": readiness,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def run_readiness_check(
    target: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Assess readiness to advance the planning lifecycle across 10 dimensions; records an audit entry."""
    return await _run_readiness_check_impl(target, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# get_available_tools — state-driven gate over the tool-loop
# ---------------------------------------------------------------------------

# After this many consecutive note turns the loop must ask_user/write_draft instead of
# noting again — the only guard against an infinite note loop (S4). Tune via T8 once the loop is
# wired (Phase 5); start at 3.
NOTE_STEP_LIMIT = 3

# The scratchpad note tools, gated together as one family against the step-limit.
NOTE_TOOL_NAMES = ("critique_note", "explore_note")

# After this many run_critique calls the formal judge is gated off the menu so the loop cannot
# spin on critique forever (spec §5.5). write_draft / ask_user stay available regardless.
CRITIQUE_ROUNDS_MAX = 3


def _tool_call_names(message) -> list[str]:
    """Tool names an AIMessage selected this turn; [] for any other message."""
    tool_calls = getattr(message, "tool_calls", None) or []
    return [tc["name"] for tc in tool_calls if isinstance(tc, dict) and tc.get("name")]


def _consecutive_note_turns(messages: list) -> int:
    """Count note turns since the last ask_user/write_draft — derived from history (N2).

    Counts per turn (per AIMessage), not per call: the limit is "N consecutive note turns", so a
    turn that batches two note calls is still one turn against the step-limit.
    """
    count = 0
    for message in reversed(messages or []):
        names = _tool_call_names(message)
        if not names:
            continue
        if any(name in ("ask_user", "write_draft", "respond") for name in names):
            break
        if any(name in NOTE_TOOL_NAMES for name in names):
            count += 1
    return count


def get_available_tools(state: WorkflowState) -> list:
    """Tools the loop may pick this turn, gated on state.

    - `finalize` only once `working_draft` is non-empty AND critique_rounds > 0 (spec §15.1:
      a finalize requires at least one run_critique; human confirmation in _finalize_impl is the
      approval step, so no separate approval_status field).
    - `run_critique` only once a draft body exists (working_draft or DB-loaded draft_body) AND
      critique_rounds < CRITIQUE_ROUNDS_MAX. It is NOT a NOTE_TOOL, so the note step-limit never
      gates it.
    - the note tools are dropped after NOTE_STEP_LIMIT consecutive notes.
    - ask_user / write_draft are ALWAYS present (stall-escape), regardless of any cap.
    """
    tools = [ask_user, respond, write_draft, critique_note, explore_note]
    has_draft = bool((state.get("working_draft") or "").strip() or (state.get("draft_body") or "").strip())
    if (state.get("working_draft") or "").strip() and (state.get("critique_rounds") or 0) > 0:
        tools.append(finalize)
    if has_draft and (state.get("critique_rounds") or 0) < CRITIQUE_ROUNDS_MAX:
        tools.append(run_critique)
    # recommend_next_workflow: available once there is a draft, or once >= 2 sections have any
    # coverage (lets the quick track recommend early, before a draft exists).
    coverage = state.get("section_coverage") or {}
    sections_with_signal = sum(1 for v in coverage.values() if status_score(v) > 0.0)
    if has_draft or sections_with_signal >= 2:
        tools.append(recommend_next_workflow)
    # run_readiness_check needs an artifact to assess.
    if (state.get("working_draft") or "").strip():
        tools.append(run_readiness_check)
    if _consecutive_note_turns(state.get("messages")) >= NOTE_STEP_LIMIT:
        tools = [t for t in tools if t.name not in NOTE_TOOL_NAMES]
    return tools
