"""Quality gate node — reflection critic using an edge-loop design.

Each `quality_gate_node` run = ONE critic round (validator -> LLM critic ->
one re-generate if it does not pass). `critique_rounds` commits at the node
boundary; `route_after_gate` decides whether to loop back to `quality_gate`
or advance to `propose_artifacts`. The `max_critique_rounds` cap is enforced
even across crash/resume.

Imports from `app.*` only — never from `tests/`.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.graphs.nodes import ANALYSIS_SCHEMA
from app.graphs.rubric import render_criteria_block
from app.graphs.state import WorkflowState
from app.graphs.validators import validate_proposal

# Bare schema (like ANALYSIS_SCHEMA) for the session llm_client; adds `suggestions`.
CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "unambiguous": {"type": "number"},
                "verifiable": {"type": "number"},
                "complete": {"type": "number"},
                "consistent": {"type": "number"},
                "traceable": {"type": "number"},
                "feasible": {"type": "number"},
                "invest": {"type": ["number", "null"]},
                "smart": {"type": ["number", "null"]},
            },
        },
        "overall": {"type": "number"},
        "rationale": {"type": "string"},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overall", "suggestions"],
}

_CRITIC_SYSTEM = (
    "Bạn là chuyên gia kỹ nghệ yêu cầu (requirements engineering). "
    "Chấm các proposal artifact theo rubric, mỗi tiêu chí 0.0–1.0, và đề xuất "
    "các sửa đổi ngắn gọn (suggestions) để cải thiện chất lượng. "
    "overall là điểm tổng hợp 0.0–1.0; rationale giải thích ngắn gọn bằng tiếng Việt."
)

_REGENERATE_SYSTEM = (
    "Bạn là BA/PM analyst. Hãy chỉnh sửa lại các proposal artifact dựa trên "
    "phản hồi chất lượng để khắc phục điểm yếu, giữ nguyên ý định ban đầu."
)


def _proposal_block(proposals: list[dict]) -> str:
    lines = []
    for i, p in enumerate(proposals, 1):
        lines.append(f"{i}. [{p.get('artifact_type', '')}] {p.get('title', '')}\n   {p.get('body', '')}")
    return "\n".join(lines) or "(không có proposal)"


def _extract_request_edit_notes(messages: list) -> list[str]:
    """Scan messages and return notes from entries with type='request_edit'."""
    notes: list[str] = []
    for m in messages or []:
        if isinstance(m, dict) and m.get("type") == "request_edit" and m.get("note"):
            notes.append(str(m["note"]))
    return notes


def _build_critic_prompt(
    proposals: list[dict],
    validation_result: dict,
    rubric_block: str,
    request_edit_notes: list[str],
) -> str:
    notes_block = "\n".join(f"- {n}" for n in request_edit_notes) or "(không có)"
    return (
        "Chấm chất lượng các proposal sau theo rubric kỹ nghệ yêu cầu.\n\n"
        f"RUBRIC:\n{rubric_block}\n\n"
        f"KẾT QUẢ VALIDATOR:\nviolations: {validation_result.get('violations')}\n"
        f"warnings: {validation_result.get('warnings')}\n\n"
        f"YÊU CẦU CHỈNH SỬA TỪ NGƯỜI DÙNG:\n{notes_block}\n\n"
        f"PROPOSALS:\n{_proposal_block(proposals)}"
    )


def _build_regenerate_prompt(
    proposals: list[dict],
    suggestions: list[str],
    request_edit_notes: list[str],
) -> str:
    feedback = request_edit_notes + suggestions
    feedback_block = "\n".join(f"- {f}" for f in feedback) or "(không có)"
    return (
        "Chỉnh sửa lại các proposal artifact dưới đây để khắc phục các điểm yếu.\n\n"
        f"PHẢN HỒI CẦN XỬ LÝ:\n{feedback_block}\n\n"
        f"PROPOSALS HIỆN TẠI:\n{_proposal_block(proposals)}\n\n"
        "Trả về JSON theo schema phân tích (next_action, confidence, proposals đã chỉnh sửa)."
    )


def _merge_proposals(old_result: dict, new_proposals: list[dict]) -> dict:
    """Replace only proposals; keep the previous round's next_action, confidence, gaps."""
    return {**old_result, "proposals": new_proposals}


async def quality_gate_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config["configurable"]
    llm_client = cfg["llm_client"]

    analysis_result = state.get("analysis_result") or {}
    proposals = analysis_result.get("proposals") or []
    artifact_type = state.get("artifact_type", "")
    current_rounds = state.get("critique_rounds", 0)

    # Step 1: validate each proposal; drop proposals missing title/body (hard block)
    all_violations: list[str] = []
    all_warnings: list[str] = []
    valid_proposals: list[dict] = []
    for proposal in proposals:
        result = validate_proposal(artifact_type, proposal)
        all_violations.extend(result.violations)
        all_warnings.extend(result.warnings)
        if result.passed:
            valid_proposals.append(proposal)

    critic_scores: dict[str, Any] = {}
    critic_overall: float | None = None
    critic_suggestions: list[str] = []
    critic_error: str | None = None
    merged = analysis_result

    # Steps 2 + 3: run critic/re-generate only when valid proposals remain
    if valid_proposals:
        notes = _extract_request_edit_notes(state.get("messages") or [])
        validation_summary = {"violations": all_violations, "warnings": all_warnings}
        try:
            critic_prompt = _build_critic_prompt(
                valid_proposals, validation_summary, render_criteria_block(), notes
            )
            critic_result, _usage = await llm_client.generate(
                messages=[{"role": "user", "content": critic_prompt}],
                system=_CRITIC_SYSTEM,
                max_tokens=1024,
                response_format=CRITIC_SCHEMA,
            )
            critic_scores = critic_result.get("scores") or {}
            critic_overall = float(critic_result.get("overall"))
            critic_suggestions = critic_result.get("suggestions") or []
        except Exception as exc:  # parse/LLM error -> skip the critic this round
            critic_error = str(exc)

        threshold = settings.critique_score_threshold
        below_threshold = critic_overall is not None and critic_overall < threshold
        if below_threshold and current_rounds < settings.max_critique_rounds:
            try:
                regen_prompt = _build_regenerate_prompt(valid_proposals, critic_suggestions, notes)
                new_result, _usage = await llm_client.generate(
                    messages=[{"role": "user", "content": regen_prompt}],
                    system=_REGENERATE_SYSTEM,
                    max_tokens=2000,
                    response_format=ANALYSIS_SCHEMA,
                )
                new_proposals = new_result.get("proposals") or []
                if new_proposals:
                    merged = _merge_proposals(analysis_result, new_proposals)
            except Exception:  # re-generate error -> keep the previous round's proposals
                pass

    passed = (
        len(all_violations) == 0
        and critic_overall is not None
        and critic_overall >= settings.critique_score_threshold
    )

    quality_report: dict[str, Any] = {
        "passed": passed,
        "overall": critic_overall,
        "scores": critic_scores,
        "violations": all_violations,
        "warnings": all_warnings,
        "suggestions": critic_suggestions,
    }
    if critic_error:
        quality_report["critic_error"] = critic_error

    return {
        "analysis_result": merged,
        "critique_rounds": current_rounds + 1,
        "quality_report": quality_report,
    }


def route_after_gate(state: WorkflowState) -> str:
    report = state.get("quality_report") or {}
    rounds = state.get("critique_rounds", 0)
    if report.get("passed") or rounds >= settings.max_critique_rounds:
        return "propose_artifacts"
    return "quality_gate"
