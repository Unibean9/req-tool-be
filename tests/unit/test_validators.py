"""Deterministic validator (pure Python). Tests written before implementation."""

from app.graphs.validators import ValidationResult, validate_proposal

# --- Group 1: Required fields (violations, hard block) ---

def test_missing_title_is_violation():
    r = validate_proposal("story", {"body": "A complete description"})
    assert not r.passed
    assert any("title" in v for v in r.violations)


def test_missing_body_is_violation():
    r = validate_proposal("story", {"title": "Title"})
    assert not r.passed
    assert any("body" in v for v in r.violations)


def test_empty_title_is_violation():
    r = validate_proposal("story", {"title": "   ", "body": "A complete description"})
    assert not r.passed
    assert any("title" in v for v in r.violations)


def test_both_present_no_violation():
    r = validate_proposal(
        "story",
        {"title": "Login", "body": "Given registered, when entering the correct password, then enter the system"},
    )
    assert r.passed
    assert r.violations == []


# --- Group 2: Weasel words (warnings, non-blocking) ---

def test_weasel_word_vi_is_warning():
    r = validate_proposal("goal", {"title": "Goal", "body": "The system runs fast within 1 month"})
    assert r.passed
    assert any("weasel" in w for w in r.warnings)


def test_weasel_word_en_is_warning():
    r = validate_proposal("goal", {"title": "Goal", "body": "Make it easy to use, increase 20% within 2 months"})
    assert r.passed
    assert any("weasel" in w for w in r.warnings)


def test_multiple_weasel_words():
    r = validate_proposal(
        "goal",
        {"title": "Goal", "body": "fast fast fast and robust, reaches 50% within 1 month"},
    )
    weasel = [w for w in r.warnings if "weasel" in w]
    # "fast" repeated 3 times -> only 1 warning; "robust" -> 1 warning
    assert len(weasel) == 2


def test_clean_text_no_warnings():
    r = validate_proposal("goal", {"title": "Goal", "body": "Increase revenue by 20% within 6 months"})
    assert r.passed
    assert r.warnings == []


# --- Group 3: Precision/recall on a labeled set ---

def _categories(r: ValidationResult) -> dict:
    return {
        "violation": len(r.violations) > 0,
        "weasel": any("weasel" in w for w in r.warnings),
    }


def test_validator_precision_recall_on_labeled_samples():
    samples = [
        (
            "story",
            {"title": "X", "body": "Given registered, when entering the correct password, then enter the system"},
            {"violation": False, "weasel": False},
        ),
        (
            "story",
            {"title": "", "body": "Given A, when B, then C"},
            {"violation": True, "weasel": False},
        ),
            (
                "goal",
                {"title": "Goal", "body": "Improve effective system behavior"},
                {"violation": False, "weasel": True},
            ),
        (
            "goal",
            {"title": "Goal", "body": "Increase revenue by 20% within 6 months"},
            {"violation": False, "weasel": False},
        ),
        (
            "story",
            {"title": "Story", "body": "Users want fast login"},
            {"violation": False, "weasel": True},
        ),
    ]
    for artifact_type, proposal, expected in samples:
        assert _categories(validate_proposal(artifact_type, proposal)) == expected


# --- Group: business rule / open question / assumption checks (spec §9.4) ---

def test_business_rule_missing_condition_raises_violation():
    r = validate_proposal("business_rule", {"title": "T", "body": "The system sends email to users."})
    assert not r.passed
    assert any("condition" in v for v in r.violations)


def test_business_rule_with_condition_and_outcome_passes():
    r = validate_proposal(
        "business_rule",
        {"title": "T", "body": "If an invoice is overdue then the system will lock the account."},
    )
    assert not any("condition" in v or "outcome" in v for v in r.violations)


def test_open_question_missing_status_raises_warning():
    r = validate_proposal("open_question", {"title": "T", "body": "Who approves the budget?"})
    assert any("status" in w for w in r.warnings)


def test_open_question_with_status_no_warning():
    r = validate_proposal("open_question", {"title": "T", "body": "Who approves?", "status": "unresolved"})
    assert not any("status" in w for w in r.warnings)


def test_assumption_missing_confidence_raises_warning():
    proposal = {
        "title": "T", "body": "Content",
        "assumptions": [{"statement": "users have phones", "confidence": "", "owner": "PM"}],
    }
    r = validate_proposal("assumption", proposal)
    assert any("confidence" in w for w in r.warnings)


def test_assumption_missing_owner_raises_warning():
    proposal = {
        "title": "T", "body": "Content",
        "assumptions": [{"statement": "users have phones", "confidence": "high", "owner": ""}],
    }
    r = validate_proposal("assumption", proposal)
    assert any("owner" in w for w in r.warnings)
