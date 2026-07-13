"""turn_audit calibrates per-component token breakdown against the real provider input count
(system/history/tools/draft are all prompt-side content), falling back to the raw char-based
estimate when no real input figure is available.
"""

from app.graphs.analysis.turn_audit import (
    _calibrate_breakdown,
    _estimate_token_breakdown,
    annotate_token_usage,
)

# ---------------------------------------------------------------------------
# breakdown calibration against a real provider total
# ---------------------------------------------------------------------------

def test_calibrate_breakdown_scales_to_match_real_total():
    raw = {"system": 100, "history": 300, "tools": 50, "draft": 50}  # total = 500

    calibrated = _calibrate_breakdown(raw, real_total=1000)

    assert sum(calibrated.values()) == 1000
    # Relative ratio is preserved: history (300/500) is still the largest component after scaling.
    assert calibrated["history"] == max(calibrated.values())
    assert calibrated["system"] == 200
    assert calibrated["tools"] == 100
    assert calibrated["draft"] == 100


def test_calibrate_breakdown_fallback_when_real_total_zero():
    raw = {"system": 10, "history": 20, "tools": 5, "draft": 5}

    calibrated = _calibrate_breakdown(raw, real_total=0)

    assert calibrated == raw


def test_calibrate_breakdown_fallback_when_raw_sum_zero():
    raw = {"system": 0, "history": 0, "tools": 0, "draft": 0}

    calibrated = _calibrate_breakdown(raw, real_total=1000)

    assert calibrated == raw


# ---------------------------------------------------------------------------
# annotate_token_usage end-to-end
# ---------------------------------------------------------------------------

def _kwargs(**overrides):
    base = dict(
        system_prompt="system prompt text " * 10,
        messages=[{"content": "hello"}, {"content": "world " * 20}],
        tool_schemas=[{"name": "ask_user"}],
        draft_body="draft body " * 5,
    )
    base.update(overrides)
    return base


def test_annotate_token_usage_calibrates_by_component_to_input_when_present():
    usage = {"input": 100, "output": 50, "total": 150}

    result = annotate_token_usage(usage, **_kwargs())

    assert result["input"] == 100
    assert result["output"] == 50
    assert result["total"] == 150
    # by_component covers only prompt-side content, so it sums to input, not total.
    assert sum(result["by_component"].values()) == 100


def test_annotate_token_usage_uses_raw_estimate_when_usage_is_none():
    result = annotate_token_usage(None, **_kwargs())

    assert result is None


def test_annotate_token_usage_uses_raw_estimate_when_input_missing():
    usage = {"output": 50, "total": 150}
    kwargs = _kwargs()

    result = annotate_token_usage(usage, **kwargs)

    assert result["output"] == 50
    assert result["total"] == 150
    # No real input figure -> keep the raw character estimate (no calibration), matching
    # _estimate_token_breakdown computed directly from the same input.
    assert result["by_component"] == _estimate_token_breakdown(
        system_prompt=kwargs["system_prompt"],
        messages=kwargs["messages"],
        tool_schemas=kwargs["tool_schemas"],
        draft_body=kwargs["draft_body"],
    )
