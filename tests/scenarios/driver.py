"""HTTP-level scenario driver.

Executes a scenario as a sequence of user-facing API actions (create session,
send message, approve/reject tool calls), draining the graph between actions and
recording every API snapshot to a transcript. Everything a real client does goes
through HTTP; only loop-draining and status polling touch the DB directly.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from tests.conftest import BASE
from tests.scenarios.conftest import ScenarioEnv
from tests.scenarios.recorder import TranscriptRecorder
from tests.scenarios.scripted_llm import ScriptedLLM


@dataclass
class Scenario:
    name: str
    artifact_type: str
    llm: ScriptedLLM
    actions: list[dict[str, Any]] = field(default_factory=list)
    expect: dict[str, Any] = field(default_factory=dict)


class ScenarioDriver:
    def __init__(self, client, env: ScenarioEnv, headers: dict, project_id: uuid.UUID, scenario: Scenario) -> None:
        self.client = client
        self.env = env
        self.headers = headers
        self.project_id = project_id
        self.scenario = scenario
        self.recorder = TranscriptRecorder(scenario.name)
        self.session_id: uuid.UUID | None = None

    # ------------------------------------------------------------------
    # HTTP wrappers
    # ------------------------------------------------------------------

    def _sessions_url(self) -> str:
        return f"{BASE}/projects/{self.project_id}/agent-sessions"

    async def _create_session(self) -> dict[str, Any]:
        resp = await self.client.post(
            self._sessions_url(),
            json={"artifact_type": self.scenario.artifact_type},
            headers=self.headers,
        )
        assert resp.status_code == 201, f"create session failed: {resp.status_code} {resp.text}"
        return resp.json()["data"]

    async def _send_message(self, content: str) -> int:
        resp = await self.client.post(
            f"{self._sessions_url()}/{self.session_id}/messages",
            json={"content": content},
            headers=self.headers,
        )
        return resp.status_code

    async def _list_messages(self) -> list[dict[str, Any]]:
        resp = await self.client.get(
            f"{self._sessions_url()}/{self.session_id}/messages", headers=self.headers
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]

    async def _list_tool_calls(self) -> list[dict[str, Any]]:
        resp = await self.client.get(
            f"{self._sessions_url()}/{self.session_id}/tool-calls", headers=self.headers
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]

    async def _get_session(self) -> dict[str, Any]:
        resp = await self.client.get(
            f"{self._sessions_url()}/{self.session_id}", headers=self.headers
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]

    async def _resolve_tool_call(self, tool_call_id: str, decision: str) -> int:
        resp = await self.client.post(
            f"{BASE}/projects/{self.project_id}/agent-tool-calls/{tool_call_id}/{decision}",
            headers=self.headers,
        )
        return resp.status_code

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    async def _snapshot(self) -> dict[str, Any]:
        session = await self._get_session()
        return {
            "session": {
                "status": session.get("status"),
                "interrupt_type": session.get("interrupt_type"),
            },
            "messages": await self._list_messages(),
            "tool_calls": await self._list_tool_calls(),
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def run(self) -> TranscriptRecorder:
        self.env.set_llm(self.scenario.llm)

        data = await self._create_session()
        self.session_id = uuid.UUID(data["session_id"])
        self.recorder.record_step(
            action={"type": "create_session", "artifact_type": self.scenario.artifact_type},
            snapshot=await self._snapshot(),
        )

        for action in self.scenario.actions:
            await self._apply(action)

        final = await self._get_session()
        self.recorder.set_summary(
            session_id=str(self.session_id),
            artifact_type=self.scenario.artifact_type,
            final_status=final.get("status"),
            final_interrupt=final.get("interrupt_type"),
            brain_turns_consumed=self.scenario.llm._tool_brain_idx,
        )

        # Slot-coverage harness: only scenarios that opt in via expect["min_coverage"]
        # touch this path, so the 11 existing scenarios stay byte-for-byte unaffected.
        if "min_coverage" in self.scenario.expect:
            min_coverage = self.scenario.expect["min_coverage"]
            coverage_ratio = await self.env.get_checkpoint_field(self.session_id, "coverage_ratio")
            assert coverage_ratio is not None, "expect.min_coverage set but coverage_ratio missing from checkpoint"
            assert coverage_ratio >= min_coverage, f"coverage_ratio {coverage_ratio} < min_coverage {min_coverage}"

        return self.recorder

    async def _apply(self, action: dict[str, Any]) -> None:
        kind = action["type"]
        if kind == "send":
            code = await self._send_message(action["content"])
            await self.env.drain(self.session_id)
            self.recorder.record_step(
                action={**action, "http_status": code}, snapshot=await self._snapshot()
            )
        elif kind in ("approve_all", "reject_all"):
            decision = "approve" if kind == "approve_all" else "reject"
            pending = [tc for tc in await self._list_tool_calls() if tc["status"] == "proposed"]
            results = []
            for tc in pending:
                code = await self._resolve_tool_call(tc["id"], decision)
                results.append({"tool_call_id": tc["id"], "decision": decision, "http_status": code})
            await self.env.drain(self.session_id)
            self.recorder.record_step(
                action={**action, "resolved": results}, snapshot=await self._snapshot()
            )
        else:
            raise ValueError(f"Unknown scenario action: {kind}")

    # ------------------------------------------------------------------
    # Produced artifacts (for eval) — read from approved tool-call snapshots,
    # which are exactly what became the ArtifactVersion body.
    # ------------------------------------------------------------------

    async def executed_artifacts(self) -> list[dict[str, Any]]:
        out = []
        for tc in await self._list_tool_calls():
            if tc["status"] == "executed":
                snap = tc.get("input_snapshot") or {}
                out.append(
                    {
                        "artifact_type": snap.get("artifact_type", self.scenario.artifact_type),
                        "title": snap.get("title", ""),
                        "body": snap.get("body", ""),
                    }
                )
        return out
