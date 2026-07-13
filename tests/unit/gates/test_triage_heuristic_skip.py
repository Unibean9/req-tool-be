"""Mid-session (draft/review/finalize) turns with unambiguous work content skip the LLM triage
call entirely; ambiguous/short messages and turns outside those phases keep calling it."""

import uuid
from unittest.mock import AsyncMock

import pytest

from app.graphs.nodes import triage_node
from tests.factories import _config


def _state_with_phase(phase: str | None, message: str, locale: str | None = "vi") -> dict:
    return {
        "messages": [{"role": "user", "content": message}],
        "session_phase": phase,
        "locale": locale,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["draft", "review", "finalize"])
async def test_heuristic_skips_llm_for_unambiguous_work_message(phase):
    llm = AsyncMock()
    state = _state_with_phase(phase, "Them field email bat buoc vao form dang ky nguoi dung", locale="vi")

    result = await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["turn_type"] == "work"
    assert result["triage_reply"] is None
    llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_heuristic_preserves_locale_from_state_without_redetecting():
    llm = AsyncMock()
    state = _state_with_phase("draft", "Update the acceptance criteria for this user story", locale="en")

    result = await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["locale"] == "en"
    llm.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["ok", "chao", "cam on nhe", "thanks"])
async def test_short_or_greeting_message_still_calls_llm_even_in_draft_phase(message):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=({"turn_type": "converse", "locale": "vi"}, {}))
    state = _state_with_phase("draft", message)

    await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    llm.generate.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [None, "intent", "elicit"])
async def test_phases_outside_draft_review_finalize_still_call_llm(phase):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=({"turn_type": "work", "locale": "vi"}, {}))
    state = _state_with_phase(phase, "Xay dung tinh nang dang nhap bang so dien thoai")

    await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    llm.generate.assert_called_once()
