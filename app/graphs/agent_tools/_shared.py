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


_CANNOT_TOOL_MESSAGE_BY_LOCALE = {
    "vi": "Không thể {tool_name}: {detail}",
    "en": "Cannot {tool_name}: {detail}",
}
_MISSING_REQUIRED_ARG_DETAIL_BY_LOCALE = {
    "vi": "thiếu trường bắt buộc '{arg_name}'.",
    "en": "missing required field '{arg_name}'.",
}
_MISSING_REQUIRED_ARG_RECOVERY_BY_LOCALE = {
    "vi": "Cung cấp '{arg_name}' rồi gọi lại {tool_name}.",
    "en": "Provide '{arg_name}' and call {tool_name} again.",
}
_TOOL_NOT_AVAILABLE_RECOVERY_BY_LOCALE = {
    "vi": "Tool này không được cung cấp ở giai đoạn hiện tại; chọn tool khác trong danh sách của lượt này.",
    "en": "This tool is not offered in the current phase; pick from the tools offered this turn.",
}


def _missing_required_arg_update(
    tool_name: str, arg_name: str, tool_call_id: str, locale: str | None = None
) -> Command:
    detail_by_locale = _MISSING_REQUIRED_ARG_DETAIL_BY_LOCALE.get(locale, _MISSING_REQUIRED_ARG_DETAIL_BY_LOCALE["en"])
    message_template = _CANNOT_TOOL_MESSAGE_BY_LOCALE.get(locale, _CANNOT_TOOL_MESSAGE_BY_LOCALE["en"])
    recovery_template = _MISSING_REQUIRED_ARG_RECOVERY_BY_LOCALE.get(
        locale, _MISSING_REQUIRED_ARG_RECOVERY_BY_LOCALE["en"]
    )
    return _recoverable_tool_update(
        RecoverableToolError(
            code="missing_required_arg",
            message=message_template.format(
                tool_name=tool_name, detail=detail_by_locale.format(arg_name=arg_name)
            ),
            recovery=recovery_template.format(arg_name=arg_name, tool_name=tool_name),
        ),
        tool_call_id,
    )


def _tool_not_available_update(
    tool_name: str, message: str, tool_call_id: str, locale: str | None = None
) -> Command:
    message_template = _CANNOT_TOOL_MESSAGE_BY_LOCALE.get(locale, _CANNOT_TOOL_MESSAGE_BY_LOCALE["en"])
    recovery = _TOOL_NOT_AVAILABLE_RECOVERY_BY_LOCALE.get(locale, _TOOL_NOT_AVAILABLE_RECOVERY_BY_LOCALE["en"])
    return _recoverable_tool_update(
        RecoverableToolError(
            code="tool_not_available",
            message=message_template.format(tool_name=tool_name, detail=message),
            recovery=recovery,
        ),
        tool_call_id,
    )


def _node_origin(state: WorkflowState, technique: str | None) -> dict[str, Any]:
    return {"turn": state.get("turn_count") or 0, "by": "agent", "technique": technique, "source": None}
