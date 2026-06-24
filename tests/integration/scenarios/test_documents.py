"""Aggregated-document scenario: build a BRD/PRD-like document from per-type sessions.

The agent has no single "BRD" artifact type — a document is the *collection* of
artifacts produced across separate sessions. This test runs the full BA→PM
pipeline (intent → problem → stakeholder → goal → functional/non-functional
requirements → epic → story) in one project, in topological predecessor order,
aggregates every produced artifact into one document, and scores the whole set.
A combined transcript captures every raw message for later validation.
"""

import uuid

import pytest

from tests.integration.scenarios.driver import ScenarioDriver
from tests.integration.scenarios.eval_support import mock_judge, score_artifacts
from tests.integration.scenarios.library import DOCUMENT_PIPELINE
from tests.integration.scenarios.recorder import TranscriptRecorder

pytestmark = pytest.mark.asyncio

# Every artifact type the aggregated document must cover (BA → PM pipeline).
_EXPECTED_TYPES = {
    "vision_objectives",
    "problem_statement",
    "stakeholder_register",
    "scope_capabilities",
    "functional_requirement",
    "non_functional_requirement",
    "use_case",
    "acceptance_criteria",
}


async def test_document_aggregates_full_pipeline(client, scenario_env, scenario_project):
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    doc = TranscriptRecorder("document-brd-prd")
    document_artifacts: list[dict] = []

    # Predecessor order matters and is honored by DOCUMENT_PIPELINE: each session
    # runs only after the artifacts it derives from already exist in the project.
    for factory in DOCUMENT_PIPELINE:
        scenario = factory()
        driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)
        recorder = await driver.run()

        assert recorder.summary["final_status"] == "completed", (
            f"{scenario.name} did not complete: {recorder.summary}"
        )
        produced = await driver.executed_artifacts()
        assert produced, f"{scenario.name} produced no artifacts"
        document_artifacts.extend(produced)

        # Fold each session's steps into the combined document transcript.
        doc.steps.append({"session": scenario.name, "steps": recorder.steps, "summary": recorder.summary})

    # The aggregated document must cover every type in the pipeline.
    types = {a["artifact_type"] for a in document_artifacts}
    assert _EXPECTED_TYPES <= types, f"Document missing types: {_EXPECTED_TYPES - types}"

    scored = await score_artifacts(document_artifacts, mock_judge())
    for s in scored:
        doc.record_eval(artifact_type=s["artifact_type"], title=s["title"], body=s["body"], score=s["score"])

    overalls = [s["score"]["overall"] for s in scored]
    doc.set_summary(
        document="BRD/PRD",
        artifact_types=sorted(types),
        artifacts_total=len(document_artifacts),
        mean_overall=(sum(overalls) / len(overalls)) if overalls else None,
    )
    path = doc.write()
    assert path.exists() and path.stat().st_size > 0
