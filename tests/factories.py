import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

from app.graphs.decision_graph import create_node
from app.graphs.state import WorkflowState, build_initial_workflow_state
from app.models.agent import AgentRun, AgentSession
from app.models.artifact import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    ChangeSource,
    VersionStatus,
)
from tests.conftest import TestSessionFactory
from tests.helpers import create_org, create_project, make_auth_headers


def _session_factory():
    @asynccontextmanager
    async def factory():
        async with TestSessionFactory() as db:
            yield db

    return factory


def _state(
    artifact_type: str = "goal",
    turn_count: int = 0,
    analysis_result: dict[str, Any] | None = None,
) -> WorkflowState:
    state = build_initial_workflow_state(
        artifact_type=artifact_type,
        workflow_area="analysis",
        step_key=None,
    )
    state["turn_count"] = turn_count
    state["analysis_result"] = analysis_result
    return state


def _config(session_id: str, project_id: str, llm_client=None) -> dict:
    return {
        "configurable": {
            "thread_id": session_id,
            "project_id": project_id,
            "llm_client": llm_client or AsyncMock(),
            "session_factory": _session_factory(),
        }
    }


async def _make_agent_session(client, db_session, project_id: uuid.UUID) -> AgentSession:
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.commit()
    return session


async def _accept_predecessor(db_session, project_id: uuid.UUID, item_type: str, body: str = "seed") -> Artifact:
    """Seed an accepted artifact of ``item_type`` with a current version so a downstream artifact
    resolves out of BLOCKED (its predecessor is accepted) in lifecycle-aware tests."""
    art = Artifact(
        project_id=project_id, type=item_type, status=ArtifactStatus.ACCEPTED, title=item_type, extra_metadata={}
    )
    db_session.add(art)
    await db_session.flush()
    ver = ArtifactVersion(
        artifact_id=art.id,
        version_number=1,
        title=item_type,
        body=body,
        status=VersionStatus.ACCEPTED,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(ver)
    await db_session.flush()
    art.current_version_id = ver.id
    await db_session.commit()
    return art


async def _make_agent_run(db_session, agent_session: AgentSession) -> AgentRun:
    run = AgentRun(session_id=agent_session.id, analysis_result={})
    db_session.add(run)
    await db_session.commit()
    return run


async def _project(client) -> uuid.UUID:
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


def _draft_state(statement: str = "Increase retention by 30%.") -> WorkflowState:
    """A BRD workflow state carrying one confirmed objective node — enough to make a
    draft view renderable for critique/gate tests.
    """
    state = _state(artifact_type="brd")
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement=statement,
            origin={"source": "test"},
            status="confirmed",
        )
    }
    return state


def _scripted_client(score: float, findings: list[str], suggestions: list[str]):
    """A judge LLM stub returning a fixed critique score payload."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=(
        {"score": score, "findings": findings, "suggestions": suggestions}, None
    ))
    return client


async def _focused_items(db_session, project_id: uuid.UUID, *item_types: ArtifactType):
    parent = Artifact(
        project_id=project_id,
        type=ArtifactType.BRD,
        status=ArtifactStatus.DRAFT,
        title="BRD",
        extra_metadata={},
    )
    db_session.add(parent)
    await db_session.flush()
    items = [
        Artifact(
            project_id=project_id,
            parent_id=parent.id,
            type=item_type,
            status=ArtifactStatus.DRAFT,
            title=item_type.value.replace("_", " ").title(),
            extra_metadata={},
        )
        for item_type in item_types
    ]
    db_session.add_all(items)
    await db_session.commit()
    return items
