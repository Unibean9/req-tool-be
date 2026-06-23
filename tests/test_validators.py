"""Deterministic validator (pure Python). Tests written before implementation."""

from app.graphs.validators import ValidationResult, validate_proposal

# --- Group 1: Required fields (violations, hard block) ---

def test_missing_title_is_violation():
    r = validate_proposal("story", {"body": "Một mô tả đầy đủ"})
    assert not r.passed
    assert any("title" in v for v in r.violations)


def test_missing_body_is_violation():
    r = validate_proposal("story", {"title": "Tiêu đề"})
    assert not r.passed
    assert any("body" in v for v in r.violations)


def test_empty_title_is_violation():
    r = validate_proposal("story", {"title": "   ", "body": "Một mô tả đầy đủ"})
    assert not r.passed
    assert any("title" in v for v in r.violations)


def test_both_present_no_violation():
    r = validate_proposal(
        "story",
        {"title": "Đăng nhập", "body": "Given đã đăng ký, when nhập đúng mật khẩu, then vào hệ thống"},
    )
    assert r.passed
    assert r.violations == []


# --- Group 2: Weasel words (warnings, non-blocking) ---

def test_weasel_word_vi_is_warning():
    r = validate_proposal("goal", {"title": "Mục tiêu", "body": "Hệ thống chạy nhanh 30% trong vòng 1 tháng"})
    assert r.passed
    assert any("weasel" in w for w in r.warnings)


def test_weasel_word_en_is_warning():
    r = validate_proposal("goal", {"title": "Goal", "body": "Make it easy to use, tăng 20% trong vòng 2 tháng"})
    assert r.passed
    assert any("weasel" in w for w in r.warnings)


def test_multiple_weasel_words():
    r = validate_proposal(
        "goal",
        {"title": "Mục tiêu", "body": "nhanh nhanh nhanh và mạnh mẽ, đạt 50% trong vòng 1 tháng"},
    )
    weasel = [w for w in r.warnings if "weasel" in w]
    # 'nhanh' repeated 3 times → only 1 warning; 'mạnh mẽ' → 1 warning
    assert len(weasel) == 2


def test_clean_text_no_warnings():
    r = validate_proposal("goal", {"title": "Mục tiêu", "body": "Tăng doanh thu 20% trong vòng 6 tháng"})
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
            {"title": "X", "body": "Given đã đăng ký, when nhập đúng mật khẩu, then vào hệ thống"},
            {"violation": False, "weasel": False},
        ),
        (
            "story",
            {"title": "", "body": "Given A, when B, then C"},
            {"violation": True, "weasel": False},
        ),
        (
            "goal",
            {"title": "Mục tiêu", "body": "Cải thiện hiệu quả hệ thống"},
            {"violation": False, "weasel": True},
        ),
        (
            "goal",
            {"title": "Mục tiêu", "body": "Tăng doanh thu 20% trong vòng 6 tháng"},
            {"violation": False, "weasel": False},
        ),
        (
            "story",
            {"title": "Story", "body": "Người dùng muốn đăng nhập nhanh"},
            {"violation": False, "weasel": True},
        ),
    ]
    for artifact_type, proposal, expected in samples:
        assert _categories(validate_proposal(artifact_type, proposal)) == expected


# --- Group: business rule / open question / assumption checks (spec §9.4) ---

def test_business_rule_missing_condition_raises_violation():
    r = validate_proposal("business_rule", {"title": "T", "body": "Hệ thống gửi email cho người dùng."})
    assert not r.passed
    assert any("condition" in v for v in r.violations)


def test_business_rule_with_condition_and_outcome_passes():
    r = validate_proposal(
        "business_rule",
        {"title": "T", "body": "Nếu đơn quá hạn thanh toán thì hệ thống sẽ khóa tài khoản."},
    )
    assert not any("condition" in v or "outcome" in v for v in r.violations)


def test_open_question_missing_status_raises_warning():
    r = validate_proposal("open_question", {"title": "T", "body": "Ai approve ngân sách?"})
    assert any("status" in w for w in r.warnings)


def test_open_question_with_status_no_warning():
    r = validate_proposal("open_question", {"title": "T", "body": "Ai approve?", "status": "unresolved"})
    assert not any("status" in w for w in r.warnings)


def test_assumption_missing_confidence_raises_warning():
    proposal = {
        "title": "T", "body": "Nội dung",
        "assumptions": [{"statement": "users have phones", "confidence": "", "owner": "PM"}],
    }
    r = validate_proposal("assumption", proposal)
    assert any("confidence" in w for w in r.warnings)


def test_assumption_missing_owner_raises_warning():
    proposal = {
        "title": "T", "body": "Nội dung",
        "assumptions": [{"statement": "users have phones", "confidence": "high", "owner": ""}],
    }
    r = validate_proposal("assumption", proposal)
    assert any("owner" in w for w in r.warnings)
