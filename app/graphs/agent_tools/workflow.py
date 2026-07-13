"""Workflow recommendation + readiness assessment — read-only analysis tools.

Both tools are side-effect-free over state and audit best-effort to AgentToolCall; neither
interrupts. Readiness scoring logic lives in app.graphs.readiness. Self-contained: no import
back into the coordinator.
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

from app.documents.registry import children_of, status_score
from app.graphs.gate_logging import log_gate_decision
from app.graphs.state import WorkflowState
from app.models.agent import AgentToolCall, AgentToolCallStatus

logger = logging.getLogger(__name__)

# First four sections define a product brief; all seven define a PRD (addendum §6).
_BRIEF_SECTIONS = ("problem_statement", "vision_objectives", "stakeholder_register", "scope_capabilities")


def _compute_recommendation(section_coverage: dict[str, str] | None, planning_track: str) -> dict[str, Any]:
    """Pure workflow-selection rule over 7-section coverage. No DB, no state mutation.

    Derives the artifact chain inline from the latest section_coverage (never the possibly-stale
    state["artifact_chain"]). On the quick track, never escalates past readiness_check.
    """
    cov = section_coverage or {}
    scores = {section: status_score(cov.get(section)) for section in children_of("brd")}
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
    # Read-only and side-effect-free, so no availability gate: calling it early just yields a weaker
    # recommendation, which is valid feedback. When to call is a prompt hint, not a safety invariant.
    result = _compute_recommendation(state.get("section_coverage"), planning_track or "quick")

    # Audit: reuse AgentToolCall.input_snapshot for the result blob (no output_snapshot column).
    # Best-effort — a DB failure must not deny the recommendation to the user.
    if not state.get("last_agent_run_id"):
        raise RuntimeError("recommend_next_workflow requires last_agent_run_id in state — analyze_node must run first")
    run_id = uuid.UUID(state["last_agent_run_id"])
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    try:
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(
                        exists().where(
                            AgentToolCall.run_id == run_id,
                            AgentToolCall.tool_name == "recommend_next_workflow",
                        )
                    )
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
    current_artifact_type: Annotated[
        str,
        "The artifact type currently in focus (cosmetic; recommendation is coverage-driven).",
    ],
    planning_track: Annotated[str, "Planning depth: 'quick' | 'standard' | 'enterprise'."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Recommend the next planning workflow from current coverage; records an audit entry.

    Use when the user asks what to do next, or when coverage suggests advancing the artifact chain.
    Read-only and non-interrupting.
    """
    return await _recommend_next_workflow_impl(current_artifact_type, planning_track, state, config, tool_call_id)


async def _run_readiness_check_impl(
    target: str,  # noqa: ARG001 — kept for schema parity; the check is coverage-driven
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    # No availability gate, same rationale as _recommend_next_workflow_impl: an early call returns a
    # low readiness score, which is valid feedback rather than a safety error.
    from app.graphs.readiness import compute_readiness_score

    report = compute_readiness_score(state.get("section_coverage"), state)

    if not state.get("last_agent_run_id"):
        raise RuntimeError("run_readiness_check requires last_agent_run_id in state — analyze_node must run first")
    run_id = uuid.UUID(state["last_agent_run_id"])
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    try:
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(
                        exists().where(
                            AgentToolCall.run_id == run_id,
                            AgentToolCall.tool_name == "run_readiness_check",
                        )
                    )
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
    log_gate_decision(
        "readiness",
        "ready" if report["ready"] else "not_ready",
        score=report["readiness_score"],
        reason=report["recommended_next_step"],
        session_id=cfg.get("thread_id"),
    )
    return Command(
        update={
            "readiness": readiness,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def run_readiness_check(
    target: Annotated[str, "Cosmetic label; the check is coverage-driven."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Assess readiness to advance the planning lifecycle across 10 dimensions; records an audit entry.

    Use after at least one critique round to check whether the draft is ready to progress. Read-only
    and non-interrupting.
    """
    return await _run_readiness_check_impl(target, state, config, tool_call_id)
