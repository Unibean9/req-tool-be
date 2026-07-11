"""Artifact-link graph tools — read the link graph, propose links/retirements, run impact analysis.

create_artifact_link and propose_retirement are approval-gated: they persist an AgentToolCall
proposal, move the session to WAITING_FOR_HUMAN, and interrupt — nothing is committed until a human
approves. read_artifact_graph and run_impact_analysis are read-only/non-interrupting. Self-contained:
no import back into the coordinator.
"""

import json
import uuid
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from sqlalchemy import select

from app.graphs import agent_tools
from app.graphs.agent_tools._shared import _missing_required_arg_update, _tool_not_available_update
from app.graphs.decision_graph import impact
from app.graphs.state import WorkflowState
from app.models.agent import (
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)
from app.models.artifact import ArtifactLink, RelationType
from app.schemas.artifact import ArtifactLinkCreateRequest


def _config_ids(config: RunnableConfig) -> tuple[uuid.UUID | None, uuid.UUID | None, Any]:
    cfg = config.get("configurable") or {}
    project_raw = cfg.get("project_id")
    thread_raw = cfg.get("thread_id")
    project_id = uuid.UUID(str(project_raw)) if project_raw is not None else None
    session_id = uuid.UUID(str(thread_raw)) if thread_raw is not None else None
    return project_id, session_id, cfg.get("session_factory")


async def _session_user_id(session_factory, session_id: uuid.UUID | None) -> uuid.UUID | None:
    if session_factory is None or session_id is None:
        return None
    async with session_factory() as db:
        return await db.scalar(select(AgentSession.created_by_id).where(AgentSession.id == session_id))


def _proposal_run_id(state: WorkflowState, tool_call_id: str, tool_name: str) -> uuid.UUID:
    run_id_raw = state.get("last_agent_run_id")
    if run_id_raw:
        return uuid.UUID(str(run_id_raw))
    # record_run_and_dispatch uses "{run_uuid}-{index}" ids. This fallback keeps proposal tools
    # idempotent even if a legacy caller did not thread last_agent_run_id into state.
    try:
        return uuid.UUID(str(tool_call_id)[:36])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{tool_name} requires last_agent_run_id in state — analyze_node must run first") from exc


async def _save_approval_proposal(
    *,
    config: RunnableConfig,
    run_id: uuid.UUID,
    session_id: uuid.UUID,
    tool_name: str,
    input_snapshot: dict[str, Any],
) -> None:
    session_factory = config["configurable"]["session_factory"]
    async with session_factory() as db:
        existing = (
            await db.execute(
                select(AgentToolCall).where(
                    AgentToolCall.run_id == run_id,
                    AgentToolCall.tool_name == tool_name,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            db.add(
                AgentToolCall(
                    run_id=run_id,
                    tool_name=tool_name,
                    input_snapshot=input_snapshot,
                    status=AgentToolCallStatus.PROPOSED,
                )
            )
        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.PROPOSE_ARTIFACTS
        await db.commit()


async def _load_artifact_links(config: RunnableConfig) -> list[dict[str, str]]:
    project_id, _session_id, session_factory = _config_ids(config)
    if project_id is None or session_factory is None:
        return []
    async with session_factory() as db:
        rows = (
            (
                await db.execute(
                    select(ArtifactLink).where(ArtifactLink.project_id == project_id).order_by(ArtifactLink.created_at)
                )
            )
            .scalars()
            .all()
        )
    return [
        {
            "source_id": str(row.source_artifact_id),
            "target_id": str(row.target_artifact_id),
            "relation_type": row.relation_type.value,
        }
        for row in rows
    ]


async def _read_artifact_graph_impl(config: RunnableConfig, tool_call_id: str) -> Command:
    links = await _load_artifact_links(config)
    return Command(update={"messages": [ToolMessage(content=json.dumps({"links": links}), tool_call_id=tool_call_id)]})


@tool("read_artifact_graph")
async def read_artifact_graph_tool(
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read artifact-link graph for the current project. Read-only, non-interrupting."""
    return await _read_artifact_graph_impl(config, tool_call_id)


async def _create_artifact_link_impl(
    source_artifact_id: str,
    target_artifact_id: str,
    relation_type: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
) -> Command:
    project_id, session_id, session_factory = _config_ids(config)
    if project_id is None or session_id is None or session_factory is None:
        return _tool_not_available_update("create_artifact_link", "missing project/session context", tool_call_id)
    try:
        body = ArtifactLinkCreateRequest(
            source_artifact_id=uuid.UUID(str(source_artifact_id)),
            target_artifact_id=uuid.UUID(str(target_artifact_id)),
            relation_type=RelationType(str(relation_type)),
        )
    except (ValueError, TypeError) as exc:
        return _tool_not_available_update("create_artifact_link", f"invalid input: {exc}", tool_call_id)
    run_id = _proposal_run_id(state, tool_call_id, "create_artifact_link")
    proposal_tool_name = (
        f"create_artifact_link:{body.source_artifact_id}:{body.target_artifact_id}:{body.relation_type.value}"
    )
    await _save_approval_proposal(
        config=config,
        run_id=run_id,
        session_id=session_id,
        tool_name=proposal_tool_name,
        input_snapshot={
            "source_artifact_id": str(body.source_artifact_id),
            "target_artifact_id": str(body.target_artifact_id),
            "relation_type": body.relation_type.value,
            "metadata": body.metadata,
        },
    )
    agent_tools.interrupt({"type": "propose_artifacts", "tool_name": "create_artifact_link"})
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        "create_artifact_link proposal is waiting for human approval; "
                        "the link is not committed until approved."
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool("create_artifact_link")
async def create_artifact_link_tool(
    source_artifact_id: Annotated[str, "Source artifact UUID."],
    target_artifact_id: Annotated[str, "Target artifact UUID."],
    relation_type: Annotated[str, "RelationType value, e.g. derives_from, depends_on, satisfies."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Propose an artifact dependency link and pause for human approval.

    The link is not committed until the proposal is approved, so do not expect read_artifact_graph to
    show it in the same turn.
    """
    return await _create_artifact_link_impl(
        source_artifact_id,
        target_artifact_id,
        relation_type,
        state,
        config,
        tool_call_id,
    )


async def _propose_retirement_impl(
    artifact_id: str,
    reason: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
    superseded_by_artifact_id: str | None = None,
) -> Command:
    project_id, session_id, session_factory = _config_ids(config)
    if project_id is None or session_id is None or session_factory is None:
        return _tool_not_available_update("propose_retirement", "missing project/session context", tool_call_id)
    if not str(reason or "").strip():
        return _missing_required_arg_update("propose_retirement", "reason", tool_call_id)
    try:
        retired_id = uuid.UUID(str(artifact_id))
        superseded_by_id = (
            uuid.UUID(str(superseded_by_artifact_id)) if str(superseded_by_artifact_id or "").strip() else None
        )
    except (TypeError, ValueError) as exc:
        return _tool_not_available_update("propose_retirement", f"invalid input: {exc}", tool_call_id)
    run_id = _proposal_run_id(state, tool_call_id, "propose_retirement")
    proposal_tool_name = f"propose_retirement:{retired_id}"
    await _save_approval_proposal(
        config=config,
        run_id=run_id,
        session_id=session_id,
        tool_name=proposal_tool_name,
        input_snapshot={
            "artifact_id": str(retired_id),
            "reason": str(reason).strip(),
            "superseded_by_artifact_id": str(superseded_by_id) if superseded_by_id else None,
        },
    )
    agent_tools.interrupt({"type": "propose_artifacts", "tool_name": "propose_retirement"})
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        "propose_retirement is waiting for human approval; "
                        "the artifact is not archived until approved."
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool("propose_retirement")
async def propose_retirement_tool(
    artifact_id: Annotated[str, "Artifact UUID to archive/retire."],
    reason: Annotated[str, "Concise audit reason for retiring this artifact."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    superseded_by_artifact_id: Annotated[str | None, "Optional replacement artifact UUID."] = None,
) -> Command:
    """Propose retiring an artifact and pause for human approval.

    Approval archives the artifact, records the optional superseded_by reference, and rejects the
    proposal if live downstream dependents still use the artifact.
    """
    return await _propose_retirement_impl(
        artifact_id,
        reason,
        state,
        config,
        tool_call_id,
        superseded_by_artifact_id,
    )


async def _run_impact_analysis_impl(
    change_description: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
    changed_artifact_id: str | None = None,
) -> Command:
    if not str(change_description or "").strip():
        return _missing_required_arg_update("run_impact_analysis", "change_description", tool_call_id)
    nodes_state = state.get("decision_nodes") or {}
    links = await _load_artifact_links(config)
    result = impact(change_description, nodes_state, links, changed_artifact_id=changed_artifact_id)
    affected = result["affected_node_ids"]
    feedback = dict(state.get("feedback_summary") or {})
    if affected:
        feedback["stale_warning"] = f"{len(affected)} node need reconfirmation due to change: {', '.join(affected)}"
        feedback["impact_result"] = {
            "affected_node_ids": affected,
            "stale_artifact_ids": result["stale_artifact_ids"],
        }
    return Command(
        update={
            "decision_nodes": result["decision_nodes"],
            "feedback_summary": feedback,
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "affected_node_ids": affected,
                            "stale_artifact_ids": result["stale_artifact_ids"],
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool("run_impact_analysis")
async def run_impact_analysis(
    change_description: Annotated[str, "User-described change that may affect existing nodes."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    changed_artifact_id: Annotated[str | None, "Artifact UUID where the change originated, if known."] = None,
) -> Command:
    """Mark exactly affected decision nodes stale; do not rewrite them silently."""
    return await _run_impact_analysis_impl(change_description, state, config, tool_call_id, changed_artifact_id)
