"""Shared tool-error helpers and cross-family primitives.

Kept free of any import back into the coordinator so it can be imported first without a cycle:
it depends only on stdlib, langchain_core, and the state type.
"""

import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.graphs.state import WorkflowState

logger = logging.getLogger(__name__)


class RecoverableToolError(Exception):
    def __init__(
        self, *, code: str, message: str, user_fixable: bool = False, recovery: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.user_fixable = user_fixable
        # A short, code-specific imperative telling the model how to recover. Optional: absent
        # recovery keeps the legacy behavior (the message alone), so old checkpoints are unchanged.
        self.recovery = recovery


def _recoverable_tool_update(
    exc: RecoverableToolError, tool_call_id: str, extra_update: dict[str, Any] | None = None
) -> Command:
    logger.info(
        "tool-error code=%s classification=recoverable user_fixable=%s message=%s",
        exc.code,
        exc.user_fixable,
        exc.message,
    )
    entry: dict[str, Any] = {
        "code": exc.code,
        "classification": "recoverable",
        "user_fixable": exc.user_fixable,
        "message": exc.message,
    }
    tool_message = exc.message
    if exc.recovery:
        entry["recovery"] = exc.recovery
        tool_message = f"{exc.message} {exc.recovery}"
    return Command(
        update={
            "tool_errors": [entry],
            "messages": [ToolMessage(content=tool_message, tool_call_id=tool_call_id, status="error")],
            **(extra_update or {}),
        }
    )


def _missing_required_arg_update(tool_name: str, arg_name: str, tool_call_id: str) -> Command:
    return _recoverable_tool_update(
        RecoverableToolError(
            code="missing_required_arg",
            message=f"Cannot {tool_name}: missing required field '{arg_name}'.",
            recovery=f"Provide '{arg_name}' and call {tool_name} again.",
        ),
        tool_call_id,
    )


def _tool_not_available_update(tool_name: str, message: str, tool_call_id: str) -> Command:
    return _recoverable_tool_update(
        RecoverableToolError(
            code="tool_not_available",
            message=f"Cannot {tool_name}: {message}",
            recovery="This tool is not offered in the current phase; pick from the tools offered this turn.",
        ),
        tool_call_id,
    )


def _node_origin(state: WorkflowState, technique: str | None) -> dict[str, Any]:
    return {"turn": state.get("turn_count") or 0, "by": "agent", "technique": technique, "source": None}
