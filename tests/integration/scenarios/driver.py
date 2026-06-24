"""HTTP-level scenario driver.

Executes a scenario as a sequence of user-facing API actions (create session,
send message, approve/reject tool calls), draining the graph between actions and
recording every API snapshot to a transcript. Everything a real client does goes
through HTTP; only loop-draining and status polling touch the DB directly.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from tests.conftest import BASE
from tests.integration.scenarios.conftest import ScenarioEnv
from tests.integration.scenarios.recorder import TranscriptRecorder
from tests.integration.scenarios.scripted_llm import ScriptedLLM

_SCENARIO_ITEM_TYPES = {
    "intent": "vision_objectives",
    "problem": "problem_statement",
    "stakeholder": "stakeholder_register",
    "goal": "scope_capabilities",
    "functional_requirement": "functional_requirement",
    "non_functional_requirement": "non_functional_requirement",
    "epic": "use_case",
    "story": "acceptance_criteria",
}

_ITEM_CONTAINERS = {
    "vision_objectives": "brd",
    "problem_statement": "brd",
    "stakeholder_register": "brd",
    "scope_capabilities": "brd",
    "functional_requirement": "prd",
    "non_functional_requirement": "prd",
    "use_case": "prd",
    "acceptance_criteria": "prd",
}


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
        item_type = _SCENARIO_ITEM_TYPES.get(
            self.scenario.artifact_type,
            self.scenario.artifact_type,
        )
        focused_artifact_id = await self._ensure_document_item(item_type)
        await self._seed_accepted_predecessors(item_type)
        resp = await self.client.post(
            self._sessions_url(),
            json={
                "artifact_type": item_type,
                "focused_artifact_id": focused_artifact_id,
            },
            headers=self.headers,
        )
        assert resp.status_code == 201, f"create session failed: {resp.status_code} {resp.text}"
        return resp.json()["data"]

    async def _ensure_document_item(self, item_type: str) -> str:
        document_type = _ITEM_CONTAINERS[item_type]
        container = await self.client.post(
            f"{BASE}/projects/{self.project_id}/documents/{document_type}",
            headers=self.headers,
        )
        assert container.status_code == 201, container.text

        existing = await self.client.get(
            f"{BASE}/projects/{self.project_id}/documents/{document_type}/{item_type}",
            headers=self.headers,
        )
        if existing.status_code == 200:
            return existing.json()["data"]["artifact_id"]

        created = await self.client.post(
            f"{BASE}/projects/{self.project_id}/documents/{document_type}/{item_type}",
            json={
                "title": item_type.replace("_", " ").title(),
                "body": "Chưa có nội dung.",
                "status": "draft",
            },
            headers=self.headers,
        )
        assert created.status_code == 201, created.text
        return created.json()["data"]["artifact_id"]

    async def _seed_accepted_predecessors(self, artifact_type: str) -> None:
        from app.graphs.policy import ARTIFACT_PREDECESSORS
        from app.models.artifact import Artifact, ArtifactStatus
        from tests.integration.scenarios.conftest import ScenarioSessionFactory

        pending = list(ARTIFACT_PREDECESSORS.get(artifact_type, []))
        seen: set[str] = set()
        predecessors: list[str] = []
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            predecessors.append(current)
            pending.extend(ARTIFACT_PREDECESSORS.get(current, []))
        if not predecessors:
            return

        async with ScenarioSessionFactory() as db:
            for pred in predecessors:
                existing_rows = (
                    await db.execute(
                        select(Artifact).where(
                            Artifact.project_id == self.project_id,
                            Artifact.type == pred,
                        )
                    )
                ).scalars().all()
                if existing_rows:
                    for row in existing_rows:
                        row.status = ArtifactStatus.ACCEPTED
                    continue
                db.add(
                    Artifact(
                        project_id=self.project_id,
                        type=pred,
                        title=f"Seed {pred}",
                        status=ArtifactStatus.ACCEPTED,
                        extra_metadata={"scenario_seed": True},
                    )
                )
            await db.commit()

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
