"""Finalize gate: working_draft non-empty AND critique_rounds > 0 (spec §15.1)."""

from unittest.mock import patch

import pytest

from app.graphs.agent_tools import _finalize_impl, get_available_tools


def _names(state):
    return {t.name for t in get_available_tools(state)}


def test_finalize_not_available_when_critique_rounds_zero():
    state = {"messages": [], "working_draft": "Một bản nháp", "critique_rounds": 0}
    assert "finalize" not in _names(state)


def test_finalize_available_when_critique_rounds_positive():
    state = {"messages": [], "working_draft": "Một bản nháp", "critique_rounds": 1}
    assert "finalize" in _names(state)


def test_finalize_not_available_without_working_draft():
    state = {"messages": [], "working_draft": None, "critique_rounds": 1}
    assert "finalize" not in _names(state)


@pytest.mark.asyncio
async def test_finalize_interrupt_triggers_when_available():
    """_finalize_impl interrupts for human confirmation (the approval step) rather than erroring."""

    class _Session:
        status = None
        interrupt_type = None

    session_row = _Session()

    class _Result:
        def scalar_one(self_inner):
            return session_row

    class _DB:
        async def execute(self_inner, *a, **k):
            return _Result()

        async def commit(self_inner):
            return None

    def _factory():
        class _Ctx:
            async def __aenter__(self_inner):
                return _DB()

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    config = {"configurable": {"session_factory": _factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {"working_draft": "draft", "critique_rounds": 1}

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert session_row.status is not None
