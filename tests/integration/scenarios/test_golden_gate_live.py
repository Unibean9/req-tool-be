"""Golden gate tests with a real analyst LLM.

These tests replace ScriptedLLM with a real model for the assertion turns.
They assert harness *invariants* and behavioral *outcomes*, not tool-call sequences.

Three scenarios from golden-conversation.md:
  B1 — Cold-start judgment: thin input must not trigger write_draft immediately.
  B2 — Decision reversal non-destructive: ScriptedLLM sets up N1; real LLM handles reversal;
       if supersede is called, old node must be preserved (never deleted).
  B3 — Sufficient context leads to draft: real LLM reaches write_draft when primed.

Run: pytest -m "integration and live" -s tests/integration/scenarios/test_golden_gate_live.py
Needs LLM_API_KEY (or equivalent) in .env.test.
"""

import sys
import uuid

import pytest

from tests.conftest import BASE

# Windows console (cp1252) cannot print Vietnamese + arrows. Reconfigure stdout so
# debug prints in live tests don't fail the assertions themselves.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from tests.eval.config import judge_settings
from tests.integration.scenarios.scripted_llm import ScriptedLLM, tool_select

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.asyncio]


def _real_analyst():
    """Build a real analyst client from the shared test credentials in .env.test."""
    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory

    return LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.llm_provider_type),
        api_key=judge_settings.llm_api_key,
        model=judge_settings.llm_model_name,
        region=judge_settings.llm_region,
        secret_key=judge_settings.llm_secret_key or None,
    )


def _skip_without_key():
    if not judge_settings.llm_api_key:
        pytest.skip("LLM_API_KEY is required in .env.test to run the live LLM")


# ---------------------------------------------------------------------------
# Shared HTTP helpers (mirrors test_harness_live_smoke.py pattern)
# ---------------------------------------------------------------------------

async def _ensure_brd_vision_item(client, headers, project_id: str) -> str:
    container = await client.post(
        f"{BASE}/projects/{project_id}/documents/brd", headers=headers
    )
    assert container.status_code in {200, 201, 409}, container.text

    existing = await client.get(
        f"{BASE}/projects/{project_id}/documents/brd/vision_objectives", headers=headers
    )
    if existing.status_code == 200:
        return existing.json()["data"]["artifact_id"]

    created = await client.post(
        f"{BASE}/projects/{project_id}/documents/brd/vision_objectives",
        json={"title": "Vision Objectives", "body": "Chua co content.", "status": "draft"},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    return created.json()["data"]["artifact_id"]


async def _open_session(client, headers, project_id: str) -> uuid.UUID:
    artifact_id = await _ensure_brd_vision_item(client, headers, project_id)
    resp = await client.post(
        f"{BASE}/projects/{project_id}/agent-sessions",
        json={"artifact_type": "vision_objectives", "focused_artifact_id": artifact_id},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["data"]["session_id"])


async def _send(client, headers, project_id: str, session_id: uuid.UUID, content: str) -> None:
    resp = await client.post(
        f"{BASE}/projects/{project_id}/agent-sessions/{session_id}/messages",
        json={"content": content},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text


def _tools_last_turn(analysis: dict | None) -> list[str]:
    if not isinstance(analysis, dict):
        return []
    return [t.get("name") for t in analysis.get("tools", [])]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_flag_on(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)


# ---------------------------------------------------------------------------
# B1 — Cold-start: thin message must NOT trigger write_draft
# ---------------------------------------------------------------------------

async def test_live_cold_start_does_not_draft_on_thin_input(
    client, scenario_env, scenario_project, graph_flag_on
):
    """Real LLM must explore (elicit/ask_user/respond) on a vague cold-start; not draft immediately."""
    _skip_without_key()
    analyst = _real_analyst()
    scenario_env.set_llm(analyst)

    headers, proj = scenario_project
    session_id = await _open_session(client, headers, proj["id"])

    await _send(client, headers, proj["id"], session_id, "I want to build a coffee shop management app.")
    await scenario_env.drain(session_id)

    analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
    tools = _tools_last_turn(analysis)
    print(f"\n[B1 cold-start] tools={tools}")

    assert "write_draft" not in tools, (
        f"Real LLM drafted on thin cold-start input — expected elicit/ask_user/respond; got {tools}"
    )
    assert "finalize" not in tools, f"Real LLM finalized with no draft; got {tools}"


def _dump_transcript(label: str, messages, decision_nodes, analysis) -> None:
    """Print the full message thread + tool calls + decision graph for a live turn."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    print(f"\n========== {label} — FULL TRANSCRIPT ==========")
    for i, m in enumerate(messages or []):
        if isinstance(m, HumanMessage):
            print(f"[{i}] 🧑 USER: {m.content}")
        elif isinstance(m, AIMessage):
            text = (m.content or "").strip()
            calls = getattr(m, "tool_calls", None) or []
            if text:
                print(f"[{i}] 🤖 AGENT(text): {text}")
            for c in calls:
                print(f"[{i}] 🛠  TOOL_CALL: {c.get('name')}  args={c.get('args')}")
            if not text and not calls:
                print(f"[{i}] 🤖 AGENT(empty)")
        elif isinstance(m, ToolMessage):
            content = str(m.content or "")
            preview = content if len(content) <= 300 else content[:300] + "..."
            print(f"[{i}] ◀ TOOL_RESULT[{m.tool_call_id}]: {preview}")
        else:
            role = getattr(m, "type", type(m).__name__)
            print(f"[{i}] {role}: {getattr(m, 'content', '')}")

    print(f"\n--- analysis_result.tools (last turn): {_tools_last_turn(analysis)}")
    print(f"--- decision_nodes ({len(decision_nodes or {})}):")
    for nid, n in (decision_nodes or {}).items():
        print(f"    {nid} [{n.get('kind')}/{n.get('status')}] {n.get('statement', '')}")
    print("=" * 52)


async def test_live_cold_start_full_transcript(
    client, scenario_env, scenario_project, graph_flag_on
):
    """Scenario 1 (golden Part 1): empty project + thin input → agent explores, does not draft.

    Dumps the full message thread + every tool call/result + the decision graph, then asserts the
    cold-start invariant (no write_draft / finalize on the first thin turn).
    """
    _skip_without_key()
    analyst = _real_analyst()
    scenario_env.set_llm(analyst)

    headers, proj = scenario_project
    session_id = await _open_session(client, headers, proj["id"])

    await _send(client, headers, proj["id"], session_id, "I want to build a coffee shop management app.")
    await scenario_env.drain(session_id)

    messages = await scenario_env.get_checkpoint_field(session_id, "messages")
    decision_nodes = await scenario_env.get_checkpoint_field(session_id, "decision_nodes")
    analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
    _dump_transcript("SCENARIO 1 COLD-START", messages, decision_nodes, analysis)

    # User-facing chat output (AgentMessage table) — what the user actually sees, not the tool thread.
    resp = await client.get(
        f"{BASE}/projects/{proj['id']}/agent-sessions/{session_id}/messages", headers=headers
    )
    assert resp.status_code == 200, resp.text
    print("\n========== USER-FACING MESSAGES ==========")
    for m in resp.json()["data"]:
        print(f"[{m.get('role')}] {m.get('content')}")
        if m.get("payload"):
            print(f"      payload={m['payload']}")
    print("=" * 42)

    tools = _tools_last_turn(analysis)
    assert "write_draft" not in tools, f"Drafted on thin cold-start; got {tools}"
    assert "finalize" not in tools, f"Finalized with no draft; got {tools}"


async def _user_facing(client, headers, project_id, session_id) -> list[dict]:
    resp = await client.get(
        f"{BASE}/projects/{project_id}/agent-sessions/{session_id}/messages", headers=headers
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]


async def test_live_cold_start_through_draft_full(
    client, scenario_env, scenario_project, graph_flag_on
):
    """Full golden Part 1→2: cold-start exploration primed across turns until the agent drafts.

    Dumps the user-facing chat each turn, then the full tool thread + decision graph + draft body.
    Lenient on whether the draft lands on the exact turn (real LLM is non-deterministic); the point
    is to show the complete block through to the draft.
    """
    _skip_without_key()
    analyst = _real_analyst()
    scenario_env.set_llm(analyst)

    headers, proj = scenario_project
    session_id = await _open_session(client, headers, proj["id"])

    for idx, content in enumerate(_PRIME_TURNS):
        await _send(client, headers, proj["id"], session_id, content)
        await scenario_env.drain(session_id)
        analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
        print(f"\n----- TURN {idx + 1}: user={content[:60]!r} -> tools={_tools_last_turn(analysis)}")
        for m in await _user_facing(client, headers, proj["id"], session_id):
            if m.get("role") == "agent":
                print(f"   [agent] {m.get('content')}")

    messages = await scenario_env.get_checkpoint_field(session_id, "messages")
    decision_nodes = await scenario_env.get_checkpoint_field(session_id, "decision_nodes")
    analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
    draft_body = await scenario_env.get_checkpoint_field(session_id, "draft_body")
    _dump_transcript("COLD-START → DRAFT", messages, decision_nodes, analysis)
    print("\n========== DRAFT BODY (rendered from graph) ==========")
    print(draft_body or "(chua co draft_body — agent chua write_draft)")
    print("=" * 54)


# ---------------------------------------------------------------------------
# B2 — Decision reversal: harness preserves old node (non-destructive supersede invariant)
# ---------------------------------------------------------------------------

def _setup_brain_for_reversal() -> ScriptedLLM:
    """Create N1 (operations-first) confirmed + N3/N4 as dependents, then wait for user."""
    return ScriptedLLM(tool_brain=[
        tool_select("elicit_tool", technique="5_whys", seed="operations"),
        tool_select("create_decision_node", node_id="N1", kind="decision",
                    statement="operations-first: kiem soat nguyen lieu theo cong thuc", technique="5_whys"),
        tool_select("create_decision_node", node_id="N3", kind="objective",
                    statement="Reduce loss nguyen lieu 20%", depends_on=["N1"]),
        tool_select("create_decision_node", node_id="N4", kind="scope",
                    statement="Realtime kitchen screen", depends_on=["N1"]),
        tool_select("update_decision_node", node_id="N1", status="confirmed"),
        tool_select("ask_user", message="Chot huong operations-first nhe?"),
    ])


async def test_live_reversal_old_node_preserved(
    client, scenario_env, scenario_project, graph_flag_on
):
    """Invariant: supersede never deletes the old node — it stays in decision_nodes as superseded.

    Setup phase: ScriptedLLM creates N1/N3/N4 deterministically.
    Assertion phase: real LLM responds to the reversal message.
    The harness must ensure N1 is not removed from decision_nodes regardless of LLM choice.
    """
    _skip_without_key()

    headers, proj = scenario_project
    session_id = await _open_session(client, headers, proj["id"])

    # --- Setup: ScriptedLLM creates nodes ---
    # The brain ends with ask_user, leaving the session in waiting-for-user interrupt state.
    # Do NOT send a second message here — the brain would be exhausted and end the session.
    scenario_env.set_llm(_setup_brain_for_reversal())
    await _send(client, headers, proj["id"], session_id,
                "I want to build a coffee shop management app - mainly inventory control.")
    await scenario_env.drain(session_id)

    nodes_before = await scenario_env.get_checkpoint_field(session_id, "decision_nodes") or {}
    assert "N1" in nodes_before, "Setup failed — N1 not created by ScriptedLLM"
    assert nodes_before["N1"]["status"] == "confirmed"

    # --- Assertion: real LLM handles reversal ---
    analyst = _real_analyst()
    scenario_env.set_llm(analyst)

    await _send(client, headers, proj["id"], session_id,
                "I reconsidered. Switch to loyalty-first - customer retention is more important.")
    await scenario_env.drain(session_id)

    nodes_after = await scenario_env.get_checkpoint_field(session_id, "decision_nodes") or {}
    analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
    tools = _tools_last_turn(analysis)
    print(f"\n[B2 reversal] tools={tools}")
    print(f"[B2 reversal] nodes_after keys={list(nodes_after)}")
    print(f"[B2 reversal] N1 status={nodes_after.get('N1', {}).get('status')}")

    # Core invariant: N1 must still exist in the graph (never deleted)
    assert "N1" in nodes_after, (
        "HARNESS VIOLATION: N1 was removed from decision_nodes. "
        "supersede_decision_node must never delete nodes — non-destructive invariant."
    )

    # If the real LLM did call supersede, verify the invariant holds structurally
    if nodes_after["N1"]["status"] == "superseded":
        new_id = nodes_after["N1"].get("superseded_by")
        assert new_id, "N1 is superseded but has no superseded_by pointer"
        assert new_id in nodes_after, f"Replacement node {new_id} missing from decision_nodes"
        assert nodes_after[new_id].get("supersedes") == "N1", (
            f"New node {new_id} does not point back to N1 via supersedes"
        )
        print(f"[B2 reversal] supersede invariant holds — N1→{new_id}")


# ---------------------------------------------------------------------------
# B3 — Sufficient context: real LLM eventually calls write_draft
# ---------------------------------------------------------------------------

_PRIME_TURNS = [
    "I want to build a coffee shop management app.",
    ("Main problem: inventory often runs short mid-shift, staff do not know how much remains. "
     "Moi tuan xay ra 3–4 lan, anh huong truc tiep doanh thu."),
    ("MVP scope: realtime inventory tracking, minimum threshold reminders, end-of-shift reports. "
     "Customers are owners or shift managers. Deadline: 6 weeks."),
    "I confirm this direction. Write the first draft for Vision & Objectives.",
]


async def test_live_sufficient_context_reaches_write_draft(
    client, scenario_env, scenario_project, graph_flag_on
):
    """Real LLM must reach write_draft when given enough context + explicit request.

    This is the behavioral contract from golden-conversation.md:
    once the analyst has sufficient signal and the user asks to draft, the model drafts.
    """
    _skip_without_key()
    analyst = _real_analyst()
    scenario_env.set_llm(analyst)

    headers, proj = scenario_project
    session_id = await _open_session(client, headers, proj["id"])

    trajectory: list[dict] = []
    for content in _PRIME_TURNS:
        await _send(client, headers, proj["id"], session_id, content)
        await scenario_env.drain(session_id)
        analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
        tools = _tools_last_turn(analysis)
        trajectory.append({"content": content[:40], "tools": tools})

    print("\n[B3 sufficient-context] trajectory:")
    for row in trajectory:
        print(f"  {row['content']!r} -> {row['tools']}")

    draft_turn = next(
        (row for row in trajectory if "write_draft" in row["tools"]), None
    )
    assert draft_turn is not None, (
        "Real LLM never called write_draft after explicit draft request with full context. "
        f"Trajectory: {trajectory}"
    )


# ---------------------------------------------------------------------------
# R3 — Mixed-initiative: user confirms agent proposal → agent must process it
# ---------------------------------------------------------------------------

def _propose_brain() -> ScriptedLLM:
    """Create N1 as proposed and ask user for confirmation."""
    return ScriptedLLM(tool_brain=[
        tool_select("create_decision_node", node_id="N1", kind="decision",
                    statement="operations-first: kiem soat nguyen lieu theo cong thuc",
                    technique="5_whys"),
        tool_select("ask_user", message="Do you agree with the operations-first direction?"),
    ])


async def test_live_user_agreement_triggers_agent_advance(
    client, scenario_env, scenario_project, graph_flag_on
):
    """R3 mixed-initiative: when user agrees with agent proposal, real LLM must process
    the agreement — confirm the node or advance the artifact (not silently ignore input).
    """
    _skip_without_key()

    headers, proj = scenario_project
    session_id = await _open_session(client, headers, proj["id"])

    # ScriptedLLM creates N1 as proposed and hands control to user
    scenario_env.set_llm(_propose_brain())
    await _send(client, headers, proj["id"], session_id,
                "I want to build a coffee shop management app - mainly inventory control.")
    await scenario_env.drain(session_id)

    nodes_before = await scenario_env.get_checkpoint_field(session_id, "decision_nodes") or {}
    assert "N1" in nodes_before, "Setup failed — N1 not created"
    assert nodes_before["N1"]["status"] == "proposed"

    # Real LLM: user explicitly agrees
    analyst = _real_analyst()
    scenario_env.set_llm(analyst)
    await _send(client, headers, proj["id"], session_id,
                "Dong y huong operations-first, di theo huong do.")
    await scenario_env.drain(session_id)

    nodes_after = await scenario_env.get_checkpoint_field(session_id, "decision_nodes") or {}
    analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
    tools = _tools_last_turn(analysis)
    print(f"\n[R3 mixed-initiative] tools={tools}")
    print(f"[R3 mixed-initiative] N1 status before={nodes_before['N1']['status']} "
          f"after={nodes_after.get('N1', {}).get('status')}")

    n1_confirmed = nodes_after.get("N1", {}).get("status") == "confirmed"
    agent_advanced = any(t in tools for t in ("write_draft", "create_decision_node", "update_decision_node"))
    assert n1_confirmed or agent_advanced, (
        f"R3 FAIL: real LLM ignored user agreement. "
        f"N1.status={nodes_after.get('N1', {}).get('status')}, tools={tools}. "
        "Expected N1 confirmed OR agent advanced (new node / draft)."
    )


# ---------------------------------------------------------------------------
# R4 — Artifact grows across turns (decision_nodes accumulate, never reset)
# ---------------------------------------------------------------------------

def _artifact_build_brain() -> ScriptedLLM:
    """Build a 4-node graph over 2 scripted turns: confirmed decision + 3 dependents."""
    return ScriptedLLM(tool_brain=[
        # Turn 1 of user message — creates the root direction
        tool_select("elicit_tool", technique="5_whys", seed="inventory shortage"),
        tool_select("create_decision_node", node_id="N1", kind="decision",
                    statement="operations-first: quantification cong thuc + stock deduction", technique="5_whys"),
        tool_select("update_decision_node", node_id="N1", status="confirmed"),
        tool_select("ask_user", message="Toi da ghi nhan huong operations-first. Tiep tuc xay scope nhe?"),
    ])


def _artifact_extend_brain() -> ScriptedLLM:
    """Second scripted turn: add 3 dependent nodes."""
    return ScriptedLLM(tool_brain=[
        tool_select("create_decision_node", node_id="N3", kind="objective",
                    statement="Reduce loss NL <4%", depends_on=["N1"]),
        tool_select("create_decision_node", node_id="N4", kind="scope",
                    statement="Canh bao ton thap realtime", depends_on=["N1"]),
        tool_select("create_decision_node", node_id="N5", kind="assumption",
                    statement="Staff use the app on tablets", depends_on=["N1"]),
        tool_select("ask_user", message="3 dependent nodes recorded. Can you review?"),
    ])


async def test_live_artifact_nodes_persist_across_turns(
    client, scenario_env, scenario_project, graph_flag_on
):
    """R4: nodes built in early turns must persist unchanged when real LLM handles later turns.

    ScriptedLLM builds a 4-node graph (N1+N3+N4+N5) over 2 turns.
    Real LLM then handles a follow-up turn.
    Assert: all 4 nodes still present and statuses not regressed (non-destructive persistence).
    """
    _skip_without_key()

    headers, proj = scenario_project
    session_id = await _open_session(client, headers, proj["id"])

    # Turn 1: ScriptedLLM creates root node N1
    scenario_env.set_llm(_artifact_build_brain())
    await _send(client, headers, proj["id"], session_id,
                "I want to build a coffee shop management app - main pain is inventory shortage.")
    await scenario_env.drain(session_id)
    nodes_t1 = dict(await scenario_env.get_checkpoint_field(session_id, "decision_nodes") or {})
    assert "N1" in nodes_t1 and nodes_t1["N1"]["status"] == "confirmed", "Setup T1 failed"

    # Turn 2: ScriptedLLM adds N3/N4/N5
    scenario_env.set_llm(_artifact_extend_brain())
    await _send(client, headers, proj["id"], session_id, "Ok, tiep tuc xay scope di.")
    await scenario_env.drain(session_id)
    nodes_t2 = dict(await scenario_env.get_checkpoint_field(session_id, "decision_nodes") or {})
    assert {"N1", "N3", "N4", "N5"} <= set(nodes_t2), "Setup T2 failed"

    # Turn 3: real LLM handles next input — assert nodes from T1/T2 are NOT removed or regressed
    analyst = _real_analyst()
    scenario_env.set_llm(analyst)
    await _send(client, headers, proj["id"], session_id,
                "Trong ok roi. Bay gio muon them chuc nang bao cao cuoi ca.")
    await scenario_env.drain(session_id)
    nodes_t3 = dict(await scenario_env.get_checkpoint_field(session_id, "decision_nodes") or {})
    analysis = await scenario_env.get_checkpoint_field(session_id, "analysis_result")
    tools = _tools_last_turn(analysis)

    print(f"\n[R4 persistence] T2={len(nodes_t2)} nodes -> T3={len(nodes_t3)} nodes")
    print(f"[R4 persistence] tools at T3={tools}")
    for nid, n in nodes_t3.items():
        print(f"  {nid} [{n.get('status')}] {n.get('statement', '')[:55]}")

    # Core invariant: all nodes from T2 must still be present in T3 (graph never shrinks)
    for nid in nodes_t2:
        assert nid in nodes_t3, (
            f"R4 FAIL: node {nid} present at T2 disappeared by T3. "
            "decision_nodes must never shrink — graph is non-destructive across real-LLM turns."
        )
    # N1 must remain confirmed (real LLM must not auto-revert a user-confirmed decision)
    assert nodes_t3["N1"]["status"] == "confirmed", (
        f"R4 FAIL: N1 status regressed from confirmed to {nodes_t3['N1']['status']} "
        "during a real-LLM turn."
    )
