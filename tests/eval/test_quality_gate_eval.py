"""Phase 5 — Eval surface for the quality gate.

Measures the before/after delta when a low-quality proposal passes through the
gate. The mock test (default) asserts the gate transforms content (content-based,
not a tautology); real judge-score comparison lives only in the integration test.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.graphs.critic import quality_gate_node
from app.graphs.validators import WEASEL_WORDS, _has_word

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _weak_fixtures() -> list[Path]:
    return sorted(_FIXTURES_DIR.glob("*_weak.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_weasel(text: str) -> int:
    return sum(1 for w in WEASEL_WORDS if _has_word(text, w))


# --- Group 1: Fixture loading ---

def test_weak_fixtures_exist():
    assert len(_weak_fixtures()) >= 2


def test_weak_fixtures_have_required_fields():
    for path in _weak_fixtures():
        fixture = _load(path)
        for key in ("artifact_type", "content", "expected_violations", "expected_warnings"):
            assert key in fixture, f"{path.name} thiếu key {key}"


# --- Helper: run the gate on a weak fixture ---

async def run_gate_eval(weak_fixture: dict, llm_client, judge_client=None) -> dict:
    proposal_before = {
        "artifact_type": weak_fixture["artifact_type"],
        "title": "Bản nháp",
        "body": weak_fixture["content"],
    }
    state = {
        "artifact_type": weak_fixture["artifact_type"],
        "messages": [],
        "analysis_result": {"next_action": "propose", "confidence": 0.8, "proposals": [proposal_before]},
        "critique_rounds": 0,
    }
    out = await quality_gate_node(state, {"configurable": {"llm_client": llm_client}})
    proposal_after = out["analysis_result"]["proposals"][0]

    result = {"before_proposal": proposal_before, "after_proposal": proposal_after}
    if judge_client is not None:
        from tests.eval import rubric
        from tests.eval.judge import judge_artifact

        before_text = f"{proposal_before['title']} {proposal_before['body']}"
        after_text = f"{proposal_after.get('title', '')} {proposal_after.get('body', '')}"
        result["before"] = await judge_artifact(before_text, rubric.RUBRIC_CRITERIA, judge_client)
        result["after"] = await judge_artifact(after_text, rubric.RUBRIC_CRITERIA, judge_client)
    return result


def _critic_low() -> tuple:
    return ({"scores": {}, "overall": 0.35, "rationale": "kém", "suggestions": ["thêm tiêu chí đo lường"]}, None)


def _regen_clean(artifact_type: str, body: str) -> tuple:
    return ({"next_action": "propose", "proposals": [{"artifact_type": artifact_type, "title": "Bản nháp", "body": body}]}, None)


# --- Group 2: Eval delta with mock (runs by default) ---

@pytest.mark.asyncio
async def test_mock_gate_changes_content():
    fixture = _load(_FIXTURES_DIR / "goal_weak.json")
    clean_body = "Tăng tỷ lệ hài lòng của người dùng lên 90% trong vòng 3 tháng."
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=[_critic_low(), _regen_clean(fixture["artifact_type"], clean_body)])

    out = await run_gate_eval(fixture, llm)
    before, after = out["before_proposal"], out["after_proposal"]

    # (a) content changed; (b) fewer weasel words — a real signal, not a tautology
    assert after != before
    before_text = f"{before['title']} {before['body']}"
    after_text = f"{after['title']} {after['body']}"
    assert _count_weasel(after_text) < _count_weasel(before_text)


@pytest.mark.asyncio
async def test_zero_weasel_words_after_gate():
    fixture = _load(_FIXTURES_DIR / "story_weak.json")
    clean_body = "Given đã đăng nhập, when chọn xuất báo cáo, then tải file CSV trong vòng 3 giây."
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=[_critic_low(), _regen_clean(fixture["artifact_type"], clean_body)])

    out = await run_gate_eval(fixture, llm)
    after = out["after_proposal"]
    after_text = f"{after.get('title', '')} {after.get('body', '')}"

    assert _count_weasel(after_text) == 0


# --- Group 3: Integration (runs only when JUDGE_API_KEY is set) ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_judge_delta_on_weak_fixtures():
    from tests.eval.config import judge_settings

    if not judge_settings.judge_api_key:
        pytest.skip("Cần JUDGE_API_KEY trong .env.test để chạy judge thật")

    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory

    judge_client = LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.judge_provider),
        api_key=judge_settings.judge_api_key,
        model=judge_settings.judge_model,
        region=judge_settings.judge_region,
        secret_key=judge_settings.judge_secret_key or None,
    )

    for path in _weak_fixtures():
        fixture = _load(path)
        clean_body = "Tăng tỷ lệ hoàn thành quy trình lên 95% trong vòng 2 tháng, đo bằng log hệ thống."
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=[_critic_low(), _regen_clean(fixture["artifact_type"], clean_body)])

        out = await run_gate_eval(fixture, llm, judge_client=judge_client)
        before, after = out["before"]["overall"], out["after"]["overall"]
        print(f"[{fixture['artifact_type']}] before={before:.3f} after={after:.3f}")
        assert after >= before
