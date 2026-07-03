"""Batched ask_user question form (plan 260702 Phase 3).

ask_user accepts up to 3 related, typed questions in one interrupt. The joined-text `message`
fallback always carries every facet, so a client that ignores the structured `questions` still
shows them. Legacy single-question calls are unchanged.
"""

from app.graphs.agent_tools import (
    _MAX_BATCH_QUESTIONS,
    _normalize_batch_questions,
    _render_batched_question_text,
)
from app.graphs.interrupts import _resume_answer_text


# --- normalization -----------------------------------------------------------


def test_normalize_clamps_to_three_questions():
    raw = [{"prompt": f"Q{i}", "type": "text"} for i in range(5)]
    assert len(_normalize_batch_questions(raw)) == _MAX_BATCH_QUESTIONS == 3


def test_normalize_drops_malformed_entries():
    raw = [
        {"prompt": "  ", "type": "text"},  # empty prompt -> dropped
        "not a dict",  # wrong shape -> dropped
        {"prompt": "Budget?", "type": "text"},  # kept
    ]
    normalized = _normalize_batch_questions(raw)
    assert [q["prompt"] for q in normalized] == ["Budget?"]


def test_normalize_defaults_unknown_type_to_text_and_keeps_choice_options():
    raw = [
        {"prompt": "Priority?", "type": "weird"},
        {"prompt": "Platform?", "type": "choice", "options": ["web", " mobile ", ""]},
    ]
    normalized = _normalize_batch_questions(raw)
    assert normalized[0]["type"] == "text"
    assert normalized[1]["type"] == "choice"
    assert normalized[1]["options"] == ["web", "mobile"]


def test_normalize_non_list_returns_empty():
    assert _normalize_batch_questions(None) == []
    assert _normalize_batch_questions("nope") == []


# --- joined-text fallback ----------------------------------------------------


def test_render_numbers_questions_under_header():
    text = _render_batched_question_text(
        "Let's scope the goal.",
        [
            {"prompt": "What is the deadline?", "type": "text"},
            {"prompt": "Which platform?", "type": "choice", "options": ["web", "mobile"]},
        ],
    )
    assert text == (
        "Let's scope the goal.\n"
        "1. What is the deadline?\n"
        "2. Which platform? (web / mobile)"
    )


def test_render_without_header_starts_at_first_question():
    text = _render_batched_question_text("", [{"prompt": "Budget?", "type": "text"}])
    assert text == "1. Budget?"


# --- resume answer mapping ---------------------------------------------------


def test_resume_pairs_structured_answers_with_questions():
    questions = [
        {"prompt": "Deadline?", "type": "text"},
        {"prompt": "Platform?", "type": "choice", "options": ["web", "mobile"]},
    ]
    text = _resume_answer_text({"answers": ["End of Q3", "web"]}, questions)
    assert text == "- Deadline?: End of Q3\n- Platform?: web"


def test_resume_falls_back_to_content_for_free_text_reply():
    questions = [{"prompt": "Deadline?", "type": "text"}]
    assert _resume_answer_text({"content": "sometime next month"}, questions) == "sometime next month"


def test_resume_plain_string_passthrough():
    assert _resume_answer_text("just text", None) == "just text"


def test_resume_ignores_empty_structured_answers():
    questions = [{"prompt": "Deadline?", "type": "text"}]
    # No paired answers -> fall back to content rather than an empty string.
    assert _resume_answer_text({"answers": ["  "], "content": "fallback"}, questions) == "fallback"
