"""Deterministic steps that do not need the model to pick a tool, and a guard against a model that
answers "done" without having done anything.

A weaker model (gpt-oss, DeepSeek on Bedrock) often replies in text -- "Đã tạo Constraints ... như
yêu cầu" -- instead of calling a tool, and the loop used to show that text as if the work was done.

- scripted_tool_call: when the next step is fully determined by state, it is taken without an LLM
  call (faster for every model, and a weak model cannot skip it): right after the user approves the
  intent summary, an artifact with a parallel plan is drafted with draft_in_parallel; right after
  that draft is written, it is proposed with write_draft.
- needs_tool_retry / claims_completion: a text-only reply before any draft exists, or a reply that
  claims something was created when no write tool ran this turn, is retried once with a tool call
  required. If the retry still only claims success, analyze_node replaces the claim with an honest
  message (see honest_fallback).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

from app.documents.registry import get_config
from app.graphs.agent_tools.draft_lifecycle import _assembled_draft_sections
from app.graphs.agent_tools.parallel_draft import has_parallel_plan
from app.graphs.analysis.prompt_assembly import _is_human_turn, _message_tool_call_id, _message_tool_calls
from app.graphs.analysis.tool_gating import _looks_like_question
from app.graphs.lifecycle_context import lifecycle_tool_block_reason

# A short approval of the intent summary. Anything longer or different (a correction, a question)
# is left to the model.
_APPROVAL_RE = re.compile(
    r"^\s*(ok(ay|e)?|xác nhận|xac nhan|đồng ý|dong y|đúng( rồi)?|chuẩn|được|duoc|yes|yep|y|confirm(ed)?|"
    r"go( ahead)?|tiếp tục|làm đi|tạo đi|viết đi|bắt đầu|start|👍)(\s+(nhé|nha|luôn|đi|ạ|thôi|rồi))*[\s!.,]*$",
    re.IGNORECASE,
)

_CLAIM_RE = re.compile(
    r"\b(đã|vừa)\s+(tạo|viết|soạn|lập|hoàn thành|hoàn tất|cập nhật|xong)"
    r"|như yêu cầu"
    r"|\b(i have|i've|has been|have been|was|were)\s+(created|drafted|written|generated|completed|updated)\b"
    r"|\b(created|generated|drafted)\b.*\bas requested\b",
    re.IGNORECASE,
)

_WRITE_TOOLS = frozenset({"write_draft", "write_draft_section", "draft_in_parallel"})
AUTO_BRIEF = "(none beyond the approved intent summary and the user's messages)"


def _turn_messages(messages: list[Any]) -> list[Any]:
    """Messages since the latest genuine human turn (inclusive)."""
    for index in range(len(messages) - 1, -1, -1):
        if _is_human_turn(messages[index]):
            return messages[index:]
    return messages


def _text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, list):
        content = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return str(content or "").strip()


def _call_names_by_id(messages: list[Any]) -> dict[str, str]:
    return {str(call.get("id")): str(call.get("name") or "") for m in messages for call in _message_tool_calls(m)}


def _tool_result_ok(message: Any) -> bool:
    status = message.get("status") if isinstance(message, dict) else getattr(message, "status", None)
    return status != "error"


def _write_tool_succeeded_this_turn(messages: list[Any]) -> bool:
    names = _call_names_by_id(messages)
    return any(
        names.get(_message_tool_call_id(message)) in _WRITE_TOOLS and _tool_result_ok(message)
        for message in _turn_messages(messages)
        if _message_tool_call_id(message)
    )


def _has_draft(state: dict[str, Any]) -> bool:
    return bool(str(state.get("draft_body") or "").strip() or state.get("draft_sections"))


def _just_approved_intent(messages: list[Any]) -> bool:
    """The last two messages are the confirm_intent result and the user's short approval."""
    if len(messages) < 2 or not _is_human_turn(messages[-1]):
        return False
    call_id = _message_tool_call_id(messages[-2])
    if not call_id or _call_names_by_id(messages).get(call_id) != "confirm_intent":
        return False
    return bool(_APPROVAL_RE.match(_text(messages[-1])))


def _last_tool_result(messages: list[Any], tool_name: str) -> Any | None:
    if not messages or not _message_tool_call_id(messages[-1]):
        return None
    last = messages[-1]
    return last if _call_names_by_id(messages).get(_message_tool_call_id(last)) == tool_name else None


def _label(artifact_type: str) -> str:
    try:
        return get_config(artifact_type).label
    except ValueError:
        return artifact_type.replace("_", " ")


def scripted_tool_call(state: dict[str, Any], available_tool_names: set[str]) -> dict[str, Any] | None:
    """The next tool call when state alone determines it, else None (the model decides)."""
    artifact_type = str(state.get("artifact_type") or "")
    messages = list(state.get("messages") or [])
    if not has_parallel_plan(artifact_type) or lifecycle_tool_block_reason(state, "write_draft") is not None:
        return None

    finished = _last_tool_result(messages, "draft_in_parallel")
    if (
        finished is not None
        and _tool_result_ok(finished)
        and "write_draft" in available_tool_names
        and _assembled_draft_sections(state, artifact_type) is not None
    ):
        return {"name": "write_draft", "args": {"title": _label(artifact_type), "body": "(assembled from sections)"}}

    if (
        "draft_in_parallel" in available_tool_names
        and state.get("user_confirmed")
        and not _has_draft(state)
        and _just_approved_intent(messages)
    ):
        return {"name": "draft_in_parallel", "args": {"brief": AUTO_BRIEF}}
    return None


def claims_completion(text: str) -> bool:
    return bool(_CLAIM_RE.search(text or ""))


def _reply_text(ai_message: AIMessage) -> str | None:
    """The user-facing text of a reply that did no work: plain text, or a lone respond call."""
    calls = list(getattr(ai_message, "tool_calls", None) or [])
    if not calls:
        return _text(ai_message)
    if len(calls) == 1 and calls[0].get("name") == "respond":
        return str((calls[0].get("args") or {}).get("message") or "")
    return None


def needs_tool_retry(state: dict[str, Any], ai_message: AIMessage) -> bool:
    """A reply that should have been a tool call: text-only before any draft exists (the work of this
    phase is done through tools), or any reply claiming something was created that no write tool
    produced this turn."""
    messages = list(state.get("messages") or [])
    text = _reply_text(ai_message)
    if text is None:
        return False
    if claims_completion(text) and not _write_tool_succeeded_this_turn(messages):
        return True
    # A plain-text question is already turned into ask_user (tool_gating._plain_response_tool), which
    # is the right move before a draft exists -- only a plain statement is a skipped tool call.
    return (
        not getattr(ai_message, "tool_calls", None)
        and bool(text)
        and not _looks_like_question(text)
        and not _has_draft(state)
    )


def retry_nudge(available_tool_names: set[str]) -> str:
    tools = ", ".join(sorted(available_tool_names))
    return (
        "Your previous reply did not call a tool, so nothing has been created or saved yet. Do the "
        f"work with a tool now (available: {tools}). Never say something was created, written or "
        "updated unless a tool in this turn did it."
    )


_HONEST_FALLBACK = {
    "vi": (
        "Mình chưa tạo được nội dung nào ở lượt này: model đang dùng không gọi được công cụ cần thiết. "
        "Bạn gửi lại yêu cầu, hoặc chọn một model mạnh hơn trong Settings."
    ),
    "en": (
        "Nothing was created in this turn: the current model did not call the tool it needed. "
        "Please send the request again, or pick a stronger model in Settings."
    ),
}


def honest_fallback(state: dict[str, Any], ai_message: AIMessage, locale: str) -> AIMessage:
    """Replace a reply that still claims unperformed work with a truthful one; otherwise unchanged."""
    text = _reply_text(ai_message)
    if (
        text is None
        or not claims_completion(text)
        or _write_tool_succeeded_this_turn(list(state.get("messages") or []))
    ):
        return ai_message
    message = _HONEST_FALLBACK.get(locale, _HONEST_FALLBACK["en"])
    calls = list(getattr(ai_message, "tool_calls", None) or [])
    if calls:
        call = dict(calls[0])
        call["args"] = {**(call.get("args") or {}), "message": message}
        return AIMessage(content="", tool_calls=[call])
    return AIMessage(content=message)


def with_nudge(messages: list[dict[str, Any]], nudge: str) -> list[dict[str, Any]]:
    """The analyzer thread with `nudge` added to its final user message (roles must keep alternating)."""
    copied = [dict(message) for message in messages]
    if copied and copied[-1].get("role") == "user":
        content = copied[-1].get("content")
        if isinstance(content, list):
            copied[-1]["content"] = [*content, {"type": "text", "text": nudge}]
        else:
            copied[-1]["content"] = f"{content}\n\n{nudge}"
    else:
        copied.append({"role": "user", "content": nudge})
    return copied
