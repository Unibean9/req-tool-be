"""Recoverable tool-error messages/recovery hints are translated by the locale stored in state,
with English as the fallback when a locale has no translation."""

from app.graphs.agent_tools import _missing_required_arg_update, _tool_not_available_update


def test_missing_required_arg_message_in_vietnamese():
    cmd = _missing_required_arg_update("write_draft", "body", "tc1", "vi")
    entry = cmd.update["tool_errors"][0]
    assert "Không thể write_draft" in entry["message"]
    assert "body" in entry["message"]
    assert "Cung cấp 'body'" in entry["recovery"]


def test_missing_required_arg_message_in_english():
    cmd = _missing_required_arg_update("write_draft", "body", "tc1", "en")
    entry = cmd.update["tool_errors"][0]
    assert entry["message"] == "Cannot write_draft: missing required field 'body'."
    assert entry["recovery"] == "Provide 'body' and call write_draft again."


def test_missing_required_arg_falls_back_to_english_for_unknown_locale():
    cmd = _missing_required_arg_update("write_draft", "body", "tc1", "fr")
    entry = cmd.update["tool_errors"][0]
    assert entry["message"] == "Cannot write_draft: missing required field 'body'."


def test_tool_not_available_message_in_vietnamese():
    cmd = _tool_not_available_update("finalize", "draft has unresolved violations", "tc1", "vi")
    entry = cmd.update["tool_errors"][0]
    assert entry["message"] == "Không thể finalize: draft has unresolved violations"
    assert "giai đoạn hiện tại" in entry["recovery"]


def test_tool_not_available_message_in_english():
    cmd = _tool_not_available_update("finalize", "draft has unresolved violations", "tc1", "en")
    entry = cmd.update["tool_errors"][0]
    assert entry["message"] == "Cannot finalize: draft has unresolved violations"
    assert entry["recovery"] == "This tool is not offered in the current phase; pick from the tools offered this turn."


def test_tool_not_available_falls_back_to_english_when_locale_unset():
    cmd = _tool_not_available_update("finalize", "draft has unresolved violations", "tc1", None)
    entry = cmd.update["tool_errors"][0]
    assert entry["message"] == "Cannot finalize: draft has unresolved violations"
