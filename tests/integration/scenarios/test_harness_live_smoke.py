"""Live smoke — confirm the thinned prompt still drives correct tool intent end-to-end.

The prompt-as-workflow → harness-as-workflow refactor moved process/contract rules out of the
system prompt into the graph (gates), the tool schemas, and per-turn context. Two risk classes:

- **Gate-enforced** rules (force-critique-before-finalize, human gate, resume idempotency) run in
  graph/code regardless of which model is used — already proven deterministically by the offline
  scenario + unit suite, so a live run cannot add signal there.
- **Model-judgment** rules (greeting ≠ artifact, ambiguity → confirm/ask, enough context → draft)
  depend on the model reading the now-thinner prompt. THIS is what a live run must confirm.

So this smoke drives a real analyst through the judgment arc and records the per-turn tool
trajectory. It hard-asserts only the model-independent structural fact (a greeting never drafts);
the trajectory itself is the signal a human reads to confirm the thinned prompt behaves.

Integration-only: needs a real LLM (reuses the judge credentials from .env.test). Skipped by the
default suite; run with `pytest -m integration -s tests/integration/scenarios/test_harness_live_smoke.py`.
"""

import uuid

import pytest

from tests.conftest import BASE
from tests.eval.config import judge_settings

_TURNS = [
    ("greeting", "Chào bạn nhé!"),
    ("ambiguous", "Mình muốn xây một công cụ gì đó cho sinh viên, ý tưởng còn mơ hồ lắm."),
    ("context-1", "Đối tượng là sinh viên học nhóm 3–6 người; pain lớn nhất là trùng lịch và quên buổi, gần như mỗi tuần."),
    ("context-2", "Phạm vi MVP: nhắc lịch + tìm khung giờ chung. Ngân sách nhỏ, làm trong 1 tháng."),
    ("draft-push", "Tôi xác nhận intent và frame. Hãy viết draft đầu tiên; phần nào chưa rõ thì đánh dấu cần xác nhận."),
]


async def _ensure_vision_item(client, headers, project_id: str) -> str:
    container = await client.post(
        f"{BASE}/projects/{project_id}/documents/brd",
        headers=headers,
    )
    assert container.status_code in {200, 201, 409}, container.text

    existing = await client.get(
        f"{BASE}/projects/{project_id}/documents/brd/vision_objectives",
        headers=headers,
    )
    if existing.status_code == 200:
        return existing.json()["data"]["artifact_id"]

    created = await client.post(
        f"{BASE}/projects/{project_id}/documents/brd/vision_objectives",
        json={
            "title": "Vision Objectives",
            "body": "Chưa có nội dung.",
            "status": "draft",
        },
        headers=headers,
    )
    assert created.status_code == 201, created.text
    return created.json()["data"]["artifact_id"]


async def _create_session(client, headers, project_id: str) -> uuid.UUID:
    focused_artifact_id = await _ensure_vision_item(client, headers, project_id)
    resp = await client.post(
        f"{BASE}/projects/{project_id}/agent-sessions",
        json={"artifact_type": "vision_objectives", "focused_artifact_id": focused_artifact_id},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["data"]["session_id"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_thinned_prompt_drives_correct_tool_intent(client, scenario_env, scenario_project):
    if not judge_settings.judge_api_key:
        pytest.skip("Cần JUDGE_API_KEY trong .env.test để chạy analyst LLM thật")

    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory

    analyst = LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.judge_provider),
        api_key=judge_settings.judge_api_key,
        model=judge_settings.judge_model,
        region=judge_settings.judge_region,
        secret_key=judge_settings.judge_secret_key or None,
    )
    scenario_env.set_llm(analyst)

    headers, proj = scenario_project
    session_id = await _create_session(client, headers, proj["id"])

    trajectory: list[dict] = []
    for label, content in _TURNS:
        await client.post(
            f"{BASE}/projects/{proj['id']}/agent-sessions/{session_id}/messages",
            json={"content": content},
            headers=headers,
        )
        status = await scenario_env.drain(session_id)
        analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
        tools = [t.get("name") for t in (analysis or {}).get("tools", [])] if isinstance(analysis, dict) else []
        errors = await scenario_env.get_checkpoint_field(session_id, "tool_errors")
        trajectory.append({
            "turn": label,
            "tools": tools,
            "active_mode": (analysis or {}).get("active_mode") if isinstance(analysis, dict) else None,
            "status": status,
            "tool_errors": [e.get("code") for e in (errors or [])] if isinstance(errors, list) else errors,
        })

    print(f"\n=== LIVE SMOKE TRAJECTORY (model={judge_settings.judge_model}) ===")
    for row in trajectory:
        print(
            f"  [{row['turn']:>13}] tools={row['tools']!s:<26} mode={row['active_mode']!s:<12} "
            f"status={row['status']} errors={row['tool_errors']}"
        )

    greeting = trajectory[0]
    assert "write_draft" not in greeting["tools"], (
        f"greeting must not draft an artifact; got tools={greeting['tools']}"
    )
    assert "finalize" not in greeting["tools"], (
        f"greeting must not finalize; got tools={greeting['tools']}"
    )
    assert any("analysis_frame" in row["tools"] for row in trajectory), (
        f"live analyst must present analysis_frame before drafting; trajectory={trajectory}"
    )
    first_draft_idx = next(
        (idx for idx, row in enumerate(trajectory) if "write_draft" in row["tools"]),
        None,
    )
    assert first_draft_idx is not None, (
        f"live analyst must reach write_draft to prove the pre-draft flow; trajectory={trajectory}"
    )
    assert any("analysis_frame" in row["tools"] for row in trajectory[:first_draft_idx]), (
        f"write_draft must be preceded by analysis_frame; trajectory={trajectory}"
    )
    assert all("analysis_frame_required" not in (row["tool_errors"] or []) for row in trajectory), (
        f"model tried to bypass analysis_frame gate; trajectory={trajectory}"
    )
