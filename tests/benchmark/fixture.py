"""Fixed, repeatable multi-turn fixture for the baseline/final latency+token benchmark.

Drives `triage_node` and `analyze_node` directly (bypassing the LangGraph ToolNode/interrupt
wiring) across 8 turns spanning triage, analyze, a same-turn critique round-trip, and a
summary-trigger turn. Baseline and final benchmark runs must import and run this same fixture
unchanged so the before/after comparison is apples-to-apples.

Token counts are a deterministic size proxy (chars // 4) attached by `BenchmarkLLM`, not real
provider-billed tokens — see evidence/benchmark-baseline.md for why (no real API key in this
environment; `AgentRun.token_usage` is only ever non-null when a client reports real usage).
"""

import uuid
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import AIMessage

from app.graphs.critique import JUDGE_SCHEMA, _invoke_judge
from app.graphs.nodes import TRIAGE_SCHEMA, analyze_node, triage_node
from app.graphs.state import (
    DEFAULT_ARTIFACT_CHAIN,
    DEFAULT_METHOD_PROFILE,
    DEFAULT_READINESS,
    WorkflowState,
)
from app.models.agent import AgentSession
from app.models.artifact import Artifact, ArtifactVersion, ChangeSource, VersionStatus
from tests.conftest import TestSessionFactory

CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class BenchmarkLLM:
    """Deterministic stand-in LLM client that attaches a size-proxy token_usage to every call.

    Routes on the same signals as tests/integration/scenarios/scripted_llm.ScriptedLLM
    (response_format / tools), but — unlike that harness client, which always returns
    usage=None — computes usage from actual payload size so baseline/final runs have a
    non-trivial token metric to compare.
    """

    def __init__(self, tool_brain: list[dict[str, Any]]):
        self._tool_brain = list(tool_brain)
        self._tool_brain_idx = 0
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **_kwargs: Any,
    ) -> tuple[Any, dict[str, int]]:
        input_chars = len(system or "") + sum(len(str(m.get("content", ""))) for m in messages)
        input_chars += len(str(tools or ""))
        input_tokens = _estimate_tokens(str(system or "") + str(messages) + str(tools or ""))

        props = ((response_format or {}).get("properties")) or {}
        if tools is not None:
            route = "tool_select"
            result = self._next_tool_turn()
        elif "turn_type" in props:
            route = "triage"
            result = {"turn_type": "work", "locale": "vi", "reply": None}
        elif set(props.keys()) == {"score", "findings", "suggestions"}:
            route = "judge"
            result = {"score": 0.6, "findings": ["missing acceptance criteria"], "suggestions": ["add one"]}
        else:
            route = "empty"
            result = {}

        output_tokens = _estimate_tokens(str(result))
        usage = {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}
        self.calls.append({"route": route, "input_tokens": input_tokens, "output_tokens": output_tokens})
        return result, usage

    def _next_tool_turn(self) -> AIMessage:
        if self._tool_brain_idx >= len(self._tool_brain):
            return AIMessage(content="", tool_calls=[])
        turn = self._tool_brain[self._tool_brain_idx]
        self._tool_brain_idx += 1
        tool_calls = [
            {"id": f"bench:{self._tool_brain_idx}:{i}", "name": item["name"], "args": dict(item.get("args") or {})}
            for i, item in enumerate(turn.get("tools") or [])
        ]
        return AIMessage(content="", tool_calls=tool_calls)


# One tool-selection turn per fixture turn. Turn 5 selects run_critique — the analyst call whose
# same-turn critique round-trip is invoked separately in run_fixture.
TOOL_BRAIN: list[dict[str, Any]] = [
    {"tools": [{"name": "ask_user", "args": {"message": "What is the primary user goal?"}}]},
    {"tools": [{"name": "write_draft", "args": {"body": "Draft v1: users can log in."}}]},
    {"tools": [{"name": "write_draft", "args": {"body": "Draft v2: users can log in and reset password."}}]},
    {"tools": [{"name": "ask_user", "args": {"message": "Any non-functional requirements?"}}]},
    {"tools": [{"name": "run_critique", "args": {"mode": "completeness"}}]},
    {"tools": [{"name": "write_draft", "args": {"body": "Draft v3: incorporates critique feedback."}}]},
    {"tools": [{"name": "ask_user", "args": {"message": "Ready to finalize?"}}]},
    {"tools": [{"name": "finalize", "args": {"summary": "Goal artifact ready for review."}}]},
]

FIXTURE_TURN_COUNT = len(TOOL_BRAIN)
SAME_TURN_CRITIQUE_INDEX = 4  # 0-based index into TOOL_BRAIN where run_critique is selected.

_USER_MESSAGES = [
    "I want to define the main user goal for this feature.",
    "Users need to be able to log in.",
    "They should also be able to reset a forgotten password.",
    "No specific performance requirements yet.",
    "Please check the draft for completeness before we continue.",
    "Good, please fold that feedback into the draft.",
    "I think we're close to done.",
    "Yes, let's finalize it.",
]


def _session_factory():
    @asynccontextmanager
    async def factory():
        async with TestSessionFactory() as db:
            yield db

    return factory


FIXTURE_ARTIFACT_TYPE = "problem_statement"


def _state() -> WorkflowState:
    return {
        "artifact_type": FIXTURE_ARTIFACT_TYPE,
        "workflow_area": "analysis",
        "step_key": None,
        "messages": [],
        "conversation_summary": "",
        "analysis_result": None,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": 0,
        "missing_context": [],
        "user_confirmed": None,
        "critique_rounds": 0,
        "quality_report": None,
        "last_critiqued_draft_hash": None,
        "locale": None,
        "turn_type": None,
        "triage_reply": None,
        "section_coverage": None,
        "coverage_complete": None,
        "section_coverage_stall_count": None,
        "assumptions": [],
        "risks": [],
        "open_questions": [],
        "key_facts": [],
        "focused_artifact_id": None,
        "draft_body": None,
        "method_profile": dict(DEFAULT_METHOD_PROFILE),
        "artifact_chain": dict(DEFAULT_ARTIFACT_CHAIN),
        "readiness": dict(DEFAULT_READINESS),
        "candidate_readiness": None,
        "tool_errors": [],
        "feedback_summary": None,
        "verification_status": None,
        "latest_checked_revision": None,
        "mode_hint": None,
        "session_elicit_count": 0,
        "decision_nodes": {},
    }


def _config(session_id: str, project_id: str, llm_client: Any) -> dict:
    return {
        "configurable": {
            "thread_id": session_id,
            "project_id": project_id,
            "llm_client": llm_client,
            "session_factory": _session_factory(),
        }
    }


async def _make_agent_session(db_session, project_id: uuid.UUID) -> AgentSession:
    session = AgentSession(
        project_id=project_id,
        artifact_type=FIXTURE_ARTIFACT_TYPE,
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.commit()
    return session


async def _seed_draft_body(db_session, project_id: uuid.UUID, body: str) -> None:
    """Overwrite the current draft body for the (goal, project) pair the fixture reads each turn.

    Mirrors the flush ordering tests/integration/test_graph_nodes.py uses so current_version_id
    resolves to the new body immediately.
    """
    artifact = Artifact(
        project_id=project_id,
        parent_id=None,
        type=FIXTURE_ARTIFACT_TYPE,
        title="Problem Statement",
        extra_metadata={},
        status="draft",
    )
    db_session.add(artifact)
    await db_session.flush()
    version = ArtifactVersion(
        artifact_id=artifact.id,
        version_number=1,
        title="Problem Statement",
        body=body,
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.AI_GENERATION,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    artifact.current_version_id = version.id
    await db_session.commit()


async def run_fixture(db_session, project_id: uuid.UUID) -> list[dict[str, Any]]:
    """Run the fixed 8-turn scenario once and return per-turn metrics.

    Each list entry: {turn, triage_latency_ms, analyze_latency_ms, triage_tokens, analyze_tokens,
    critique_tokens (0 if none this turn), total_latency_ms, total_tokens}.
    """
    import time

    agent_session = await _make_agent_session(db_session, project_id)
    llm = BenchmarkLLM(TOOL_BRAIN)
    state = _state()
    metrics: list[dict[str, Any]] = []

    for turn_idx in range(FIXTURE_TURN_COUNT):
        state["messages"] = [*state["messages"], {"role": "user", "content": _USER_MESSAGES[turn_idx]}]

        triage_config = _config(str(agent_session.id), str(project_id), llm)
        t0 = time.monotonic()
        triage_result = await triage_node(state, triage_config)
        triage_latency_ms = int((time.monotonic() - t0) * 1000)
        triage_tokens = llm.calls[-1]["input_tokens"] + llm.calls[-1]["output_tokens"]
        state = {**state, **triage_result}

        analyze_config = _config(str(agent_session.id), str(project_id), llm)
        t0 = time.monotonic()
        analyze_result = await analyze_node(state, analyze_config)
        analyze_latency_ms = int((time.monotonic() - t0) * 1000)
        analyze_tokens = llm.calls[-1]["input_tokens"] + llm.calls[-1]["output_tokens"]
        state = {**state, **analyze_result}

        critique_tokens = 0
        critique_latency_ms = 0
        if turn_idx == SAME_TURN_CRITIQUE_INDEX:
            draft_body = state.get("draft_body") or ""
            t0 = time.monotonic()
            await _invoke_judge(draft_body, "completeness", llm)
            critique_latency_ms = int((time.monotonic() - t0) * 1000)
            critique_tokens = llm.calls[-1]["input_tokens"] + llm.calls[-1]["output_tokens"]

        # write_draft turns grow the persisted draft body so later analyze calls read a bigger
        # prompt, mirroring how a real session's draft accretes.
        tool_turn = TOOL_BRAIN[turn_idx]["tools"][0]
        if tool_turn["name"] == "write_draft":
            await _seed_draft_body(db_session, project_id, tool_turn["args"]["body"])

        metrics.append(
            {
                "turn": turn_idx + 1,
                "triage_latency_ms": triage_latency_ms,
                "analyze_latency_ms": analyze_latency_ms,
                "critique_latency_ms": critique_latency_ms,
                "total_latency_ms": triage_latency_ms + analyze_latency_ms + critique_latency_ms,
                "triage_tokens": triage_tokens,
                "analyze_tokens": analyze_tokens,
                "critique_tokens": critique_tokens,
                "total_tokens": triage_tokens + analyze_tokens + critique_tokens,
            }
        )

    return metrics
