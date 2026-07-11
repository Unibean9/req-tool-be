"""run_critique — formal judge call over the current draft (mode-targeted, non-interrupting).

Unlike critique_note (silent scratchpad), run_critique invokes the production judge in
critique.py, records a quality_report, and increments critique_rounds. It does not interrupt —
the analyst surfaces the result to the user via `respond` on a later turn.

The draft-cache read (`current_draft_body`) and the finalize-gate hash check (`_draft_hash_stale`)
live in the coordinator; reach them through the module reference at call time to avoid an
import cycle.
"""

import hashlib
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.config import settings
from app.graphs import agent_tools
from app.graphs.agent_tools._shared import _missing_required_arg_update, _tool_not_available_update
from app.graphs.gate_logging import log_gate_decision
from app.graphs.state import QualityReport, WorkflowState

CRITIQUE_ROUNDS_MAX = settings.max_critique_rounds


async def _run_critique_impl(
    target: str,  # noqa: ARG001 — kept for schema parity; the judge scores the loaded draft body
    mode: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    from app.graphs.critique import _invoke_judge

    if not str(mode or "").strip():
        return _missing_required_arg_update("run_critique", "mode", tool_call_id)
    if (state.get("critique_rounds") or 0) >= CRITIQUE_ROUNDS_MAX and not agent_tools._draft_hash_stale(state):
        return _tool_not_available_update(
            "run_critique",
            "critique round limit reached; revise, respond/escalate, or finalize if the gate has passed.",
            tool_call_id,
        )

    cfg = config["configurable"]
    llm_client = cfg.get("strong_llm_client") or cfg.get("llm_client")
    # Source of truth for both the critique target and the hash: see current_draft_body. The
    # finalize gate reads the same helper, so the scored body and the gate body can never diverge.
    body = await agent_tools.current_draft_body(state, config)
    if not body.strip():
        return _tool_not_available_update(
            "run_critique",
            "no current draft to critique; write_draft or load an artifact first.",
            tool_call_id,
        )
    judged = await _invoke_judge(body, mode, llm_client)

    threshold = settings.critique_score_threshold
    score = judged["score"]
    findings = judged["findings"]
    suggestions = judged["suggestions"]
    # Gate result is derived from score, NOT from blocking_issues emptiness — the no-LLM degraded
    # path (score=0.0, findings=[]) must still "fail" so the loop can never finalize without a real
    # critique. This is fail-safe by design, not a bug.
    quality_gate_result = "fail" if score < threshold else "pass"
    blocking_issues = findings if quality_gate_result == "fail" else []
    non_blocking_warnings = findings if quality_gate_result == "pass" else []
    revision_plan = suggestions if quality_gate_result == "fail" else []

    rounds_after = (state.get("critique_rounds") or 0) + 1
    # A passing gate steers to finalize. A failing gate steers to revise while rounds remain; once
    # the rounds cap is reached and the gate still fails the loop has no auto-recovery (run_critique
    # is gated off, finalize is blocked), so it must escalate — hand the decision to the user rather
    # than revise silently forever. "re_critique" is never recommended (it would be a dead signal).
    if quality_gate_result == "pass":
        recommended_next_action = "finalize"
    elif rounds_after >= CRITIQUE_ROUNDS_MAX:
        recommended_next_action = "escalate"
    else:
        recommended_next_action = "revise"

    log_gate_decision(
        "critique",
        quality_gate_result,
        score=score,
        reason=recommended_next_action,
        session_id=cfg.get("thread_id"),
    )
    report: QualityReport = {
        "mode": judged["mode"],
        "score": score,
        "findings": findings,
        "suggestions": suggestions,
        "blocking_issues": blocking_issues,
        "non_blocking_warnings": non_blocking_warnings,
        "revision_plan": revision_plan,
        "quality_gate_result": quality_gate_result,
        "recommended_next_action": recommended_next_action,
    }
    draft_hash = hashlib.md5(body.encode()).hexdigest()[:8]
    summary = f"critique[{report['mode']}] score={report['score']:.2f} gate={quality_gate_result}"
    return Command(
        update={
            "quality_report": report,
            "last_critiqued_draft_hash": draft_hash,
            "critique_rounds": rounds_after,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def run_critique(
    target: Annotated[str, "Cosmetic label for the target; the judge always scores the current draft body."],
    mode: Annotated[str, "Critique dimension, e.g. 'completeness', 'clarity', 'feasibility'."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Run a formal quality critique over the current draft along one mode and record the report.

    Use after a draft exists to score it before finalizing. Does not interrupt — surface the result
    to the user with respond on a later turn. Gated off after the critique-rounds cap is reached.
    """
    return await _run_critique_impl(target, mode, state, config, tool_call_id)
