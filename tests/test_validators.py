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
    # 'nhanh' lặp 3 lần → chỉ 1 cảnh báo; 'mạnh mẽ' → 1 cảnh báo
    assert len(weasel) == 2


def test_clean_text_no_warnings():
    r = validate_proposal("goal", {"title": "Mục tiêu", "body": "Tăng doanh thu 20% trong vòng 6 tháng"})
    assert r.passed
    assert r.warnings == []


# --- Group 3: INVEST for story ---

def test_invest_missing_testable_for_story():
    r = validate_proposal("story", {"title": "Story", "body": "Người dùng muốn đăng nhập vào hệ thống"})
    assert any("INVEST" in w for w in r.warnings)


def test_invest_ok_for_story():
    r = validate_proposal(
        "story",
        {"title": "Story", "body": "Given đã đăng ký, when nhập đúng mật khẩu, then vào được hệ thống"},
    )
    assert not any("INVEST" in w for w in r.warnings)


def test_invest_not_applied_to_goal():
    r = validate_proposal("goal", {"title": "Goal", "body": "Tăng doanh thu 20% trong vòng 6 tháng"})
    assert not any("INVEST" in w for w in r.warnings)


# --- Group 4: SMART for goal ---

def test_smart_missing_measurable_for_goal():
    r = validate_proposal("goal", {"title": "Goal", "body": "Cải thiện trải nghiệm người dùng"})
    assert any("SMART" in w for w in r.warnings)


def test_smart_ok_for_goal():
    r = validate_proposal("goal", {"title": "Goal", "body": "Tăng tỷ lệ giữ chân lên 30% trong vòng 3 tháng"})
    assert not any("SMART" in w for w in r.warnings)


def test_smart_not_applied_to_story():
    r = validate_proposal(
        "story",
        {"title": "Story", "body": "Given đã đăng ký, when nhập đúng, then vào hệ thống"},
    )
    assert not any("SMART" in w for w in r.warnings)


# --- Group 5: Precision/recall on a labeled set ---

def _categories(r: ValidationResult) -> dict:
    return {
        "violation": len(r.violations) > 0,
        "weasel": any("weasel" in w for w in r.warnings),
        "invest": any("INVEST" in w for w in r.warnings),
        "smart": any("SMART" in w for w in r.warnings),
    }


def test_validator_precision_recall_on_labeled_samples():
    samples = [
        (
            "story",
            {"title": "X", "body": "Given đã đăng ký, when nhập đúng mật khẩu, then vào hệ thống"},
            {"violation": False, "weasel": False, "invest": False, "smart": False},
        ),
        (
            "story",
            {"title": "", "body": "Given A, when B, then C"},
            {"violation": True, "weasel": False, "invest": False, "smart": False},
        ),
        (
            "goal",
            {"title": "Mục tiêu", "body": "Cải thiện hiệu quả hệ thống"},
            {"violation": False, "weasel": True, "invest": False, "smart": True},
        ),
        (
            "goal",
            {"title": "Mục tiêu", "body": "Tăng doanh thu 20% trong vòng 6 tháng"},
            {"violation": False, "weasel": False, "invest": False, "smart": False},
        ),
        (
            "story",
            {"title": "Story", "body": "Người dùng muốn đăng nhập nhanh"},
            {"violation": False, "weasel": True, "invest": True, "smart": False},
        ),
    ]
    for artifact_type, proposal, expected in samples:
        assert _categories(validate_proposal(artifact_type, proposal)) == expected
