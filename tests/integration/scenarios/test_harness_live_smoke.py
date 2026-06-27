"""Live smoke — confirm the thinned prompt still drives correct tool intent end-to-end.

The prompt-as-workflow → harness-as-workflow refactor moved process/contract rules out of the
system prompt into the graph (gates), the tool schemas, and per-turn context. Two risk classes:

- **Gate-enforced** rules (force-critique-before-finalize, human gate, resume idempotency) run in
  graph/code regardless of which model is used — already proven deterministically by the offline
  scenario + unit suite, so a live run cannot add signal there.
- **Model-judgment** rules (greeting = artifact, ambiguity → confirm/ask, enough context → draft)
  depend on the model reading the now-thinner prompt. THIS is what a live run must confirm.

So this smoke drives a real analyst through the judgment arc and records the per-turn tool
trajectory. It hard-asserts only the model-independent structural fact (a greeting never drafts);
the trajectory itself is the signal a human reads to confirm the thinned prompt behaves.

Integration-only: needs a real LLM (reuses shared credentials from .env.test). Skipped by the
default suite; run with `pytest -m integration -s tests/integration/scenarios/test_harness_live_smoke.py`.
"""

import uuid

import pytest

from tests.conftest import BASE
from tests.eval.config import judge_settings

_TURNS = [
    ("greeting", "Hello there!"),
    ("ambiguous", "Minh muon xay mot cong cu gi do cho sinh vien, y tuong con mo ho lam."),
    ("context-1", "Audience is students in groups of 3-6; biggest pain is schedule conflicts and forgotten sessions almost every week."),
    ("context-2", "Pham vi MVP: nhac lich + tim khung gio chung. Ngan sach nho, lam trong 1 thang."),
    ("draft-push", "I confirm the intent and frame. Write the first draft; mark unclear parts as needing confirmation."),
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
            "body": "Chua co content.",
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
    if not judge_settings.llm_api_key:
        pytest.skip("LLM_API_KEY is required in .env.test to run the real analyst LLM")

    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory

    analyst = LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.llm_provider_type),
        api_key=judge_settings.llm_api_key,
        model=judge_settings.llm_model_name,
        region=judge_settings.llm_region,
        secret_key=judge_settings.llm_secret_key or None,
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

    print(f"\n=== LIVE SMOKE TRAJECTORY (model={judge_settings.llm_model_name}) ===")
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
    first_draft_idx = next(
        (idx for idx, row in enumerate(trajectory) if "write_draft" in row["tools"]),
        None,
    )
    assert first_draft_idx is not None, (
        f"live analyst must reach write_draft to prove the pre-draft flow; trajectory={trajectory}"
    )
