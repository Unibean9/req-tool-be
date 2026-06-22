"""Foundation DB & Config.

Verify the schema foundation for the agent UX feature: the JSONB payload column on agent_messages,
the agent_turn_timeout_seconds config, the locale/intent fields on WorkflowState, and the payload
on AgentMessageResponse. No business logic here — schema only.
"""
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.graphs.state import WorkflowState
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentSession,
    AgentSessionStatus,
)
from app.schemas.agent import AgentMessageResponse


async def _make_session(db_session) -> AgentSession:
    session = AgentSession(
        project_id=uuid.uuid4(),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.mark.asyncio
async def test_agent_message_payload_column_nullable(db_session):
    session = await _make_session(db_session)

    msg_none = AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content="hi", payload=None)
    msg_set = AgentMessage(
        session_id=session.id,
        role=AgentMessageRole.AGENT,
        content="chào bạn",
        payload={"kind": "greeting", "locale": "vi"},
    )
    db_session.add_all([msg_none, msg_set])
    await db_session.flush()
    await db_session.refresh(msg_none)
    await db_session.refresh(msg_set)

    assert msg_none.payload is None
    assert msg_set.payload["kind"] == "greeting"


@pytest.mark.asyncio
async def test_agent_message_content_still_required(db_session):
    session = await _make_session(db_session)

    db_session.add(AgentMessage(session_id=session.id, role=AgentMessageRole.USER))
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


def test_settings_agent_turn_timeout_default():
    assert settings.agent_turn_timeout_seconds == 90.0


def test_workflow_state_accepts_locale_and_intent():
    assert "locale" in WorkflowState.__annotations__
    assert "intent" in WorkflowState.__annotations__


def test_agent_message_response_payload_nullable():
    base = {
        "id": uuid.uuid4(),
        "session_id": uuid.uuid4(),
        "role": AgentMessageRole.AGENT,
        "content": "nội dung",
        "created_at": None,
        "updated_at": None,
    }

    resp_none = AgentMessageResponse.model_validate({**base, "payload": None})
    resp_set = AgentMessageResponse.model_validate({**base, "payload": {"kind": "info"}})

    assert resp_none.payload is None
    assert resp_set.payload["kind"] == "info"


def test_agent_message_response_payload_defaults_to_none():
    """Old clients send no payload → field defaults to None, does not raise."""
    resp = AgentMessageResponse.model_validate(
        {
            "id": uuid.uuid4(),
            "session_id": uuid.uuid4(),
            "role": AgentMessageRole.USER,
            "content": "hello",
            "created_at": None,
            "updated_at": None,
        }
    )
    assert resp.payload is None
