"""Aggregated-document scenario: build a BRD/PRD smoke document from canonical journeys.

The agent has no single "BRD" artifact type; a document is the collection of
artifacts produced across separate sessions. This evidence test now uses a
short topological smoke pipeline instead of an exhaustive artifact-type matrix.
"""

import uuid

import pytest

from tests.integration.scenarios.driver import ScenarioDriver
from tests.integration.scenarios.eval_support import mock_judge, score_artifacts
from tests.integration.scenarios.library import DOCUMENT_PIPELINE
from tests.integration.scenarios.recorder import TranscriptRecorder

pytestmark = [pytest.mark.integration, pytest.mark.evidence, pytest.mark.asyncio]

# Artifact types covered by the reduced BA to PM smoke pipeline.
_EXPECTED_TYPES = {
    "vision_objectives",
    "problem_statement",
    "functional_requirement",
}


async def test_document_aggregates_smoke_pipeline(client, scenario_env, scenario_project, tmp_path):
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    doc = TranscriptRecorder("document-brd-prd")
    document_artifacts: list[dict] = []

    # Predecessor order matters and is honored by DOCUMENT_PIPELINE.
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
    path = doc.write(tmp_path / "document-transcripts")
    assert path.exists() and path.stat().st_size > 0
