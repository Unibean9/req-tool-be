"""Render scenario transcripts into readable conversation markdown.

Reads JSON transcript files, extracts conversation messages + tool call proposals,
and renders a human-readable conversation trace with session status annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TRANSCRIPTS_DIR = Path(__file__).parents[2] / "tests" / "integration" / "scenarios" / "transcripts"

_KIND_LABEL = {
    "question": "❓ Agent (câu hỏi)",
    "assessment": "📋 Agent (xác nhận intent)",
}

_STATUS_LABEL = {
    "active": "🟢 ACTIVE",
    "waiting_for_human": "🔵 WAITING_FOR_HUMAN",
    "completed": "✅ COMPLETED",
    "failed": "❌ FAILED",
}

_INTERRUPT_LABEL = {
    "stream_response": "STREAM_RESPONSE",
    "propose_artifacts": "PROPOSE_ARTIFACTS",
    "ask_human": "ASK_HUMAN",
    None: "—",
}


def _render_message(msg: dict[str, Any]) -> str:
    role = msg["role"]
    content = msg["content"]
    payload = msg.get("payload") or {}
    kind = payload.get("kind")

    if role == "user":
        return f"> 👤 **User:** {content}"
    label = _KIND_LABEL.get(kind, "🤖 Agent")
    return f"> {label}: {content}"


def _render_tool_call(tc: dict[str, Any]) -> list[str]:
    snap = tc.get("input_snapshot") or {}
    status = tc.get("status", "unknown")
    title = snap.get("title", "")
    body = snap.get("body", "")
    tool = tc.get("tool_name", "").split(":")[0]
    verdict = "✅ EXECUTED" if status == "executed" else "⏳ PROPOSED"

    lines = [
        f"**[{tool}]** {verdict}: *{title}*",
        "",
    ]
    if body:
        lines += [
            "```markdown",
            body[:600] + ("..." if len(body) > 600 else ""),
            "```",
        ]
    return lines


def render_transcript(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = data["scenario"]
    summary = data.get("summary", {})
    steps = data.get("steps", [])
    evals = data.get("eval", [])

    lines: list[str] = [
        f"## Scenario: `{scenario}`",
        "",
    ]

    # Summary box
    status_str = _STATUS_LABEL.get(summary.get("final_status", ""), summary.get("final_status", ""))
    lines += [
        "| Field | Value |",
        "| --- | --- |",
        f"| Artifact type | `{summary.get('artifact_type', '—')}` |",
        f"| Final status | {status_str} |",
        f"| Brain turns consumed | {summary.get('brain_turns_consumed', '—')} |",
        f"| Artifacts produced | {summary.get('artifacts_produced', '—')} |",
    ]
    if summary.get("mean_overall") is not None:
        lines.append(f"| Artifact quality (mean) | {summary['mean_overall']:.2f} |")
    lines.append("")

    # Conversation trace
    lines.append("### Conversation trace")
    lines.append("")

    # Track seen message IDs to avoid duplicates from cumulative snapshots
    seen_msg_ids: set[str] = set()
    seen_tc_ids: set[str] = set()

    for step in steps:
        action = step["action"]
        snap = step.get("snapshot", {})
        sess = snap.get("session", {})
        status = sess.get("status", "")
        interrupt = sess.get("interrupt_type")

        action_type = action.get("type", "")

        if action_type == "create_session":
            lines.append(f"*Session created — artifact: `{action.get('artifact_type')}`*")
            lines.append("")
            continue

        if action_type == "send":
            content = action.get("content", "")
            lines.append(f"> 👤 **User:** {content}")
            lines.append("")

        # New agent messages this step
        new_msgs = [m for m in snap.get("messages", []) if m["id"] not in seen_msg_ids]
        for msg in new_msgs:
            seen_msg_ids.add(msg["id"])
            if msg["role"] == "agent":
                lines.append(_render_message(msg))
                lines.append("")

        # New tool calls this step
        new_tcs = [tc for tc in snap.get("tool_calls", []) if tc["id"] not in seen_tc_ids]
        for tc in new_tcs:
            seen_tc_ids.add(tc["id"])
            for line in _render_tool_call(tc):
                lines.append(line)
            lines.append("")

        if action_type == "approve_all":
            lines.append("*[User: approved draft]*")
            lines.append("")
        elif action_type == "reject_all":
            lines.append("*[User: rejected draft]*")
            lines.append("")

        # Session status annotation after each step
        sl = _STATUS_LABEL.get(status, status)
        il = _INTERRUPT_LABEL.get(interrupt, interrupt or "—")
        lines.append(f"*Session: {sl} | interrupt: `{il}`*")
        lines.append("")

    # Eval scores
    if evals:
        lines.append("### Artifact quality scores")
        lines.append("")
        lines.append("| Criterion | Score |")
        lines.append("| --- | ---: |")
        ev = evals[0]
        for k, v in (ev.get("score", {}).get("scores") or {}).items():
            val = f"{v:.2f}" if v is not None else "—"
            lines.append(f"| {k} | {val} |")
        overall = ev.get("score", {}).get("overall")
        if overall is not None:
            lines.append(f"| **overall** | **{overall:.2f}** |")
        rationale = ev.get("score", {}).get("rationale")
        if rationale:
            lines.append("")
            lines.append(f"*{rationale}*")
        lines.append("")

    return "\n".join(lines)


def render_conversation_file(
    transcript_names: list[str],
    title: str,
    plan_notes: str,
    output_path: Path,
) -> None:
    parts = [
        f"# {title}",
        "",
        plan_notes,
        "",
        "---",
        "",
    ]
    for name in transcript_names:
        path = _TRANSCRIPTS_DIR / f"{name}.json"
        if not path.exists():
            parts.append(f"## Scenario: `{name}`")
            parts.append("")
            parts.append(f"*Transcript not found at `{path}`*")
            parts.append("")
            continue
        parts.append(render_transcript(path))
        parts.append("---")
        parts.append("")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Written: {output_path}")


def main() -> None:
    # harness-conversation-fluency: multi-turn Q&A (D4 ACTIVE status + D1 multi-tool) +
    # reject-then-explore (D1 composite dispatch gating).
    render_conversation_file(
        transcript_names=["multi-turn-qna", "reject-then-explore"],
        title="Conversation Trace: Harness Conversation Fluency (D1–D5)",
        plan_notes=(
            "Các kịch bản này chứng minh D4 (session ACTIVE sau ask_user), D1 (composite dispatch).\n"
            "Mỗi bước hiển thị message thực tế từ API + trạng thái session.\n\n"
            "**Claim D4:** Sau `ask_user`, session status = `ACTIVE` (không phải `WAITING_FOR_HUMAN`).\n"
            "**Claim D1:** Nhiều tool_call có thể dispatch trong một turn (gate pass-through)."
        ),
        output_path=Path("plans/harness-conversation-fluency/conversation-trace.md"),
    )

    # intent-phase-gate: intent-propose-approve (D6 confirm_intent → analysis_frame → write_draft)
    # + multi-turn-qna (D6 multi-turn flow).
    render_conversation_file(
        transcript_names=["intent-propose-approve", "multi-turn-qna"],
        title="Conversation Trace: Intent Phase Gate (D6)",
        plan_notes=(
            "Kịch bản chứng minh D6 intent gate: `confirm_intent` mở artifact phase, "
            "`analysis_frame` trình khung phân tích, rồi mới unblock `write_draft`.\n"
            "Session giữ `ACTIVE` sau `confirm_intent` (STREAM_RESPONSE interrupt);\n"
            "`write_draft` chỉ xuất hiện sau khi intent đã confirm và frame đã được trình.\n\n"
            "**Claim D6:** `confirm_intent` → `interrupt_type=STREAM_RESPONSE`, session `ACTIVE`.\n"
            "**Claim D6-flow:** Sau confirm + analysis_frame, `write_draft` propose thành công → artifact."
        ),
        output_path=Path("plans/intent-phase-gate/conversation-trace.md"),
    )


if __name__ == "__main__":
    main()
