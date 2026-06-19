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


async def analyze_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    project_id = uuid.UUID(cfg["project_id"])
    llm_client = cfg["llm_client"]

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
    analysis_result = await llm_client.generate(
        messages=[{"role": "user", "content": prompt}],
        system=system_prompt,
        max_tokens=2000,
        response_format=ANALYSIS_SCHEMA,
    )

    async with session_factory() as db:
        run = AgentRun(session_id=session_id, analysis_result=analysis_result)
        db.add(run)
        await db.commit()
        run_id = str(run.id)

    return {
        "analysis_result": analysis_result,
        "turn_count": state["turn_count"] + 1,
        "last_agent_run_id": run_id,
    }


def route_node(state: WorkflowState) -> str:
    if state["turn_count"] >= settings.max_agent_turns:
        return END

    action = (state.get("analysis_result") or {}).get("next_action", "done")
    if action == "ask":
        return "ask_human"
    if action == "propose":
        return "propose_artifacts"
    return END


async def ask_human_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])

    message = (state.get("analysis_result") or {}).get("message", "")

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

    interrupt({"type": "ask_human", "message": message})
    return {}


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

    async with session_factory() as db:
        for proposal in proposals:
            artifact_type = proposal.get("artifact_type", "")
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

    messages_summary = "\n".join(
        f"{role}: {content}"
        for role, content in (_msg_role_content(m) for m in (state.get("messages") or [])[-5:])
    ) or "(chưa có hội thoại)"

    return (
        f"Bạn là BA/PM analyst. Phân tích và đề xuất artifact cho loại: {state['artifact_type']}.\n\n"
        f"Context hiện tại:\n{artifact_context}\n\n"
        f"Hội thoại gần đây:\n{messages_summary}\n\n"
        "Trả về JSON với next_action (ask/propose/done), confidence (0-1), gaps, message (nếu ask), proposals (nếu propose)."
    )
