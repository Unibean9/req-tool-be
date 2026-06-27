"""Enum-to-Tool Parity Wrap.

Wraps the three enum branches as parallel tools (ask→ask_user, propose→write_draft,
done→finalize) without removing the enum branches. Guards the R1 (duplicate-message on
HTTP-resume) and R3 (idempotency-key collision) risks.

Unit tests (T1–T5) call the tool impls directly with `interrupt` patched. T6 exercises the
real ToolNode dispatch + interrupt/resume through a minimal compiled graph (analyze_node
cannot emit native tool_calls without bind_tools, so the HTTP driver path cannot
reach the tools yet — a seeded AIMessage seeds tool-path coverage).
"""

import hashlib
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.graphs.decision_graph import create_node, render_view
from app.models.agent import (
    AgentMessage,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
)
from app.models.artifact import Artifact, ArtifactStatus, ArtifactType
from tests.conftest import TestSessionFactory
from tests.helpers import create_org, create_project, make_auth_headers
from tests.integration.test_graph_nodes import (
    _config,
    _make_agent_run,
    _make_agent_session,
    _session_factory,
    _state,
)


def _set_graph_draft(state: dict, statement: str = "draft") -> str:
    state["artifact_type"] = "brd"
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement=statement,
            origin={"source": "test"},
            status="confirmed",
        )
    }
    return render_view(state["decision_nodes"], state["artifact_type"])


async def _project(client) -> uuid.UUID:
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


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


# ---------------------------------------------------------------------------
# T1 — ask_user idempotency on resume
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")  # ask_user delegates to nodes._save_and_interrupt_ask
async def test_ask_user_tool_idempotent_on_resume(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _ask_user_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    # Resume re-executes the tool body from the top: same ToolCall.id + content twice.
    await _ask_user_impl("Ban muon xay gi?", state, config, "call_abc")
    await _ask_user_impl("Ban muon xay gi?", state, config, "call_abc")

    async with TestSessionFactory() as db:
        msgs = (
            await db.execute(
                select(AgentMessage).where(AgentMessage.session_id == agent_session.id)
            )
        ).scalars().all()
        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# T2 — both paths delegate to the shared helper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_user_tool_uses_shared_helper(client, db_session):
    from app.graphs import nodes
    from app.graphs.agent_tools import _ask_user_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    with patch.object(nodes, "_save_and_interrupt_ask", new=AsyncMock(return_value="ok")) as helper:
        command = await _ask_user_impl("Ban muon xay gi?", state, config, "call_1")

    helper.assert_awaited_once()
    assert command.update["messages"][0].content == "ok"


# ---------------------------------------------------------------------------
# T3 — write_draft idempotency key (run_id, tool_name)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")  # write_draft calls interrupt directly (not via nodes)
async def test_write_draft_tool_idempotency_key_run_id_tool_name(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _write_draft_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(
        db_session,
        project_id,
        ArtifactType.VISION_OBJECTIVES,
    )
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives", analysis_result={"next_action": "propose"})
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _write_draft_impl("Title", "Than bai", state, config, "call_1")
    await _write_draft_impl("Title", "Than bai", state, config, "call_1")

    async with TestSessionFactory() as db:
        rows = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].tool_name == f"write_draft:{focused.id}"
        assert rows[0].input_snapshot["focused_artifact_id"] == str(focused.id)
        assert rows[0].input_snapshot["synthesis_metadata"]["synthesis_source"] == "bmad_synthesis"


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_scopes_body_and_idempotency_to_focused_artifact(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _write_draft_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    focused_a, focused_b = await _focused_items(
        db_session,
        project_id,
        ArtifactType.VISION_OBJECTIVES,
        ArtifactType.PROBLEM_STATEMENT,
    )
    agent_session.focused_artifact_id = focused_a.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(analysis_result={"next_action": "propose"})
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused_a.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _write_draft_impl("Title", "Than bai section", state, config, "call_1")
    await _write_draft_impl("Title", "Than bai section", state, config, "call_1")
    state["focused_artifact_id"] = str(focused_b.id)
    await _write_draft_impl("Title 2", "Than bai section 2", state, config, "call_2")

    assert command.update["draft_body"] == "Than bai section"
    async with TestSessionFactory() as db:
        rows = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        assert {row.tool_name for row in rows} == {
            f"write_draft:{focused_a.id}",
            f"write_draft:{focused_b.id}",
        }
        snapshots = {row.tool_name: row.input_snapshot for row in rows}
        assert snapshots[f"write_draft:{focused_a.id}"]["focused_artifact_id"] == str(focused_a.id)
        assert snapshots[f"write_draft:{focused_b.id}"]["focused_artifact_id"] == str(focused_b.id)


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_snapshot_keeps_single_canonical_body_when_graph_render_wins(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _write_draft_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    model_body = "\n\n".join(
        [
            "## Vision\nModel-authored vision.",
            "## Objectives\n| Objective | Rationale |\n| --- | --- |\n| Improve onboarding | Reduce drop-off |",
            "## Success Metrics\n| Metric | Target |\n| --- | --- |\n| Activation | 25% |",
        ]
    )
    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    state["decision_nodes"] = {
        "N1": create_node(
            kind="fact",
            statement="Students need a clearer first-session path.",
            origin={"source": "test"},
            status="confirmed",
            section="Vision",
        )
    }
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _write_draft_impl("Vision", model_body, state, config, "call_1")

    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        assert row.input_snapshot["body"] == render_view(state["decision_nodes"], "vision_objectives")
        assert "model_body" not in row.input_snapshot
        assert "body_source" not in row.input_snapshot


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_snapshot_records_base_version_and_assumptions(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _write_draft_impl
    from app.models.artifact import ArtifactVersion, ChangeSource, VersionStatus

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    current = ArtifactVersion(
        artifact_id=focused.id,
        version_number=1,
        title="Vision cu",
        body="Body cu",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(current)
    await db_session.flush()
    focused.current_version_id = current.id
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    state["assumptions"] = [{"statement": "Retention metric confirmed by the user", "source": "user", "status": "confirmed"}]
    state["open_questions"] = [{"question": "Specific target needs confirmation", "domain": "metrics"}]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _write_draft_impl("Vision", "## Vision\n...", state, config, "call_1")

    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        metadata = row.input_snapshot["synthesis_metadata"]
        assert row.input_snapshot["base_version_id"] == str(current.id)
        assert metadata["base_version_id"] == str(current.id)
        assert metadata["confirmed_assumptions"] == ["Retention metric confirmed by the user"]
        assert metadata["pending_assumptions"] == ["Specific target needs confirmation"]
        assert "## Assumptions" in row.input_snapshot["body"]
        assert "- Retention metric confirmed by the user" in row.input_snapshot["body"]
        assert "- Specific target needs confirmation ⚠️ needs confirmation" in row.input_snapshot["body"]


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_snapshot_records_candidate_readiness(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _write_draft_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    state["open_questions"] = [{"question": "Specific target needs confirmation", "domain": "metrics"}]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()
    body = "\n\n".join(
        [
            "## Vision\nTang retention.",
            "## Objectives\n- Cai thien activation.",
            "## Success Metrics\n- Retention target is missing.",
        ]
    )

    command = await _write_draft_impl("Vision", body, state, config, "call_1")

    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        readiness = row.input_snapshot["candidate_readiness"]
        assert readiness["state"] == "needs_confirmation"
        assert readiness["can_persist"] is True
        assert readiness["blocking_reasons"] == []
        assert "- Specific target needs confirmation ⚠️ needs confirmation" in row.input_snapshot["body"]
        assert command.update["candidate_readiness"]["state"] == "needs_confirmation"
        assert command.update["tool_errors"] == []


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_write_draft_graph_render_marker_makes_pending_assumption_persistable(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _write_draft_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    state["decision_nodes"] = {
        "V1": create_node(
            kind="fact",
            statement="Students need faster group scheduling.",
            origin={"source": "test"},
            status="confirmed",
            section="## Vision",
        ),
        "O1": create_node(
            kind="objective",
            statement="Reduce scheduling coordination time.",
            origin={"source": "test"},
            status="confirmed",
            section="## Objectives",
        ),
        "M1": create_node(
            kind="objective",
            statement="Successful scheduling rate target needs confirmation.",
            origin={"source": "test"},
            status="needs_confirmation",
            section="## Success Metrics",
            fields={
                "goal": "Schedule study groups",
                "user/business value": "Students coordinate faster",
                "metric": "Successful scheduling rate",
                "target": "80%",
                "timeframe": "First semester",
            },
        )
    }
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _write_draft_impl("Vision", "## Vision\nModel body without marker.", state, config, "call_1")

    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        assert row.input_snapshot["body"].startswith(render_view(state["decision_nodes"], "vision_objectives"))
        assert "body_source" not in row.input_snapshot
        assert "⚠️ needs confirmation" in row.input_snapshot["body"]
        assert "## Assumptions" in row.input_snapshot["body"]
        assert row.input_snapshot["candidate_readiness"]["state"] == "needs_confirmation"
        assert row.input_snapshot["candidate_readiness"]["can_persist"] is True
        assert command.update["candidate_readiness"]["can_persist"] is True


@pytest.mark.asyncio
async def test_write_draft_missing_focus_returns_recoverable_observation(client, db_session):
    from app.graphs.agent_tools import _write_draft_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _write_draft_impl("Vision", "## Vision\nContent", state, config, "call_1")

    assert command.update["tool_errors"][0]["classification"] == "recoverable"
    assert command.update["tool_errors"][0]["code"] == "missing_focused_artifact"
    assert "focused_artifact_id" in command.update["messages"][0].content


# ---------------------------------------------------------------------------
# T4 — finalize interrupt gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")  # finalize calls interrupt directly (not via nodes)
async def test_finalize_tool_raises_interrupt(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _finalize_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state(artifact_type="brd")
    body = _set_graph_draft(state, "draft")
    state["critique_rounds"] = 1
    state["last_critiqued_draft_hash"] = hashlib.md5(body.encode()).hexdigest()[:8]
    state["quality_report"] = {"quality_gate_result": "pass"}  # finalize now requires a passing gate
    state["candidate_readiness"] = {"state": "sufficient", "can_persist": True}
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _finalize_impl("Da hoan tat.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))
        ).scalar_one()
        assert session_row.status == AgentSessionStatus.WAITING_FOR_HUMAN


def test_finalize_not_available_when_candidate_readiness_is_not_sufficient():
    from app.graphs.agent_tools import get_available_tools

    state = _state(artifact_type="brd")
    body = _set_graph_draft(state, "draft")
    state["critique_rounds"] = 1
    state["last_critiqued_draft_hash"] = hashlib.md5(body.encode()).hexdigest()[:8]
    state["quality_report"] = {"quality_gate_result": "pass"}
    state["candidate_readiness"] = {
        "state": "well_structured_but_incomplete",
        "can_persist": False,
        "blocking_reasons": ["Missing target needing confirmation"],
    }

    tool_names = {tool.name for tool in get_available_tools(state)}

    assert "finalize" not in tool_names


# ---------------------------------------------------------------------------
# T5 — ask_user uses ToolCall.id (not state's last_agent_run_id) as idempotency key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_user_uses_tool_call_id_not_state_run_id(client, db_session):
    from app.graphs import nodes
    from app.graphs.agent_tools import _ask_user_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["last_agent_run_id"] = "old-run-id"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    with patch.object(nodes, "_save_and_interrupt_ask", new=AsyncMock(return_value="ok")) as helper:
        await _ask_user_impl("Ban muon xay gi?", state, config, "new-tool-call-id")

    run_id = helper.await_args.kwargs["run_id"]
    assert "new-tool-call-id" in str(run_id)
    assert run_id != "old-run-id"


# ---------------------------------------------------------------------------
# T6 — end-to-end tool path through a real ToolNode dispatch (interrupt/resume)
# ---------------------------------------------------------------------------

def _tool_graph():
    """Minimal compiled graph: START → tools (real ToolNode) → END, with a checkpointer.

    The compiled graph injects the Runtime that ToolNode + interrupt() need, so we exercise the
    real dispatch path. WorkflowState carries the fields the tools read (last_agent_run_id, etc.).
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from app.graphs.agent_tools import ask_user, finalize, write_draft
    from app.graphs.state import WorkflowState

    builder = StateGraph(WorkflowState)
    builder.add_node("tools", ToolNode([ask_user, write_draft, finalize]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=MemorySaver())


def _ai_tool_call(name: str, args: dict, call_id: str = "c1"):
    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


@pytest.mark.asyncio
async def test_ask_user_tool_call_scenario(client, db_session):
    from langgraph.types import Command

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    graph = _tool_graph()
    state = _state()
    state["messages"] = [_ai_tool_call("ask_user", {"message": "Ban muon xay gi?"})]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out  # paused for the human

    async with TestSessionFactory() as db:
        from app.models.agent import AgentSession

        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))
        ).scalar_one()
        # D4: ask_user keeps session ACTIVE (conversational Q&A) with STREAM_RESPONSE interrupt type.
        assert session_row.status == AgentSessionStatus.ACTIVE
        assert session_row.interrupt_type == AgentSessionInterruptType.STREAM_RESPONSE

    # Resume round-trip: a second invoke with the user's reply must complete, no crash.
    resumed = await graph.ainvoke(Command(resume={"content": "Mot app lich nhom"}), config)
    assert "__interrupt__" not in resumed


@pytest.mark.asyncio
async def test_write_draft_tool_call_scenario(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(
        db_session,
        project_id,
        ArtifactType.VISION_OBJECTIVES,
    )
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    graph = _tool_graph()
    state = _state()
    state["user_confirmed"] = True  # artifact phase mo: write_draft new dispatch thay vi self-reject
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    state["messages"] = [_ai_tool_call("write_draft", {"title": "Goal", "body": "Content"})]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out

    # Resume re-executes the tool body: the idempotency guard must keep it at one row, and the
    # Command(update={messages:[ToolMessage]}) return path (only reached on resume) must complete.
    from langgraph.types import Command

    resumed = await graph.ainvoke(Command(resume={"decision": "approve"}), config)
    assert "__interrupt__" not in resumed

    async with TestSessionFactory() as db:
        rows = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].tool_name == f"write_draft:{focused.id}"
        assert rows[0].input_snapshot["focused_artifact_id"] == str(focused.id)


@pytest.mark.asyncio
async def test_finalize_tool_call_scenario(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    graph = _tool_graph()
    state = _state()
    body = _set_graph_draft(state, "draft")
    state["critique_rounds"] = 1
    state["last_critiqued_draft_hash"] = hashlib.md5(body.encode()).hexdigest()[:8]
    state["quality_report"] = {"quality_gate_result": "pass"}  # finalize now requires a passing gate
    state["candidate_readiness"] = {"state": "sufficient", "can_persist": True}
    state["messages"] = [_ai_tool_call("finalize", {"summary": "Da hoan tat."})]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out

    # Resume must complete via the Command(update=...) return path (only reached on resume).
    from langgraph.types import Command

    resumed = await graph.ainvoke(Command(resume={"content": "ok"}), config)
    assert "__interrupt__" not in resumed


# ---------------------------------------------------------------------------
# M2 — read_artifact: side-effect-free body read by id
# ---------------------------------------------------------------------------

async def _artifact_with_body(db_session, project_id: uuid.UUID, body: str, title: str = "Vision"):
    from app.models.artifact import ArtifactVersion, ChangeSource, VersionStatus

    artifact = Artifact(
        project_id=project_id,
        type=ArtifactType.VISION_OBJECTIVES,
        status=ArtifactStatus.DRAFT,
        title=title,
        extra_metadata={},
    )
    db_session.add(artifact)
    await db_session.flush()
    version = ArtifactVersion(
        artifact_id=artifact.id,
        version_number=1,
        title=title,
        body=body,
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    artifact.current_version_id = version.id
    await db_session.commit()
    return artifact


@pytest.mark.asyncio
async def test_read_artifact_returns_current_body(client, db_session):
    from app.graphs.agent_tools import _read_artifact_impl

    project_id = await _project(client)
    artifact = await _artifact_with_body(db_session, project_id, "## Vision\nContent goc.")
    config = _config(str(uuid.uuid4()), str(project_id))

    command = await _read_artifact_impl(str(artifact.id), config, "call_1")

    msg = command.update["messages"][0]
    assert "Content goc" in msg.content
    assert msg.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_read_artifact_not_found_returns_observation(client):
    from app.graphs.agent_tools import _read_artifact_impl

    project_id = await _project(client)
    config = _config(str(uuid.uuid4()), str(project_id))

    command = await _read_artifact_impl(str(uuid.uuid4()), config, "call_1")

    assert "artifact not found" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_read_artifact_invalid_id_returns_observation(client):
    from app.graphs.agent_tools import _read_artifact_impl

    project_id = await _project(client)
    config = _config(str(uuid.uuid4()), str(project_id))

    command = await _read_artifact_impl("not-a-uuid", config, "call_1")

    assert "invalid id" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_read_artifact_truncates_large_body(client, db_session):
    from app.graphs.agent_tools import READ_ARTIFACT_MAX_CHARS, _read_artifact_impl

    project_id = await _project(client)
    artifact = await _artifact_with_body(db_session, project_id, "x" * (READ_ARTIFACT_MAX_CHARS + 500))
    config = _config(str(uuid.uuid4()), str(project_id))

    command = await _read_artifact_impl(str(artifact.id), config, "call_1")

    assert "remaining content truncated" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_read_artifact_scoped_to_project(client, db_session):
    """An artifact in another project is invisible — the project_id filter is the scope boundary."""
    from app.graphs.agent_tools import _read_artifact_impl

    project_a = await _project(client)
    project_b = await _project(client)
    artifact = await _artifact_with_body(db_session, project_b, "body bi mat")
    config = _config(str(uuid.uuid4()), str(project_a))

    command = await _read_artifact_impl(str(artifact.id), config, "call_1")

    assert "artifact not found" in command.update["messages"][0].content
