"""Coverage is derived from accepted document children, not LLM self-assessment."""

import uuid

import pytest

from app.documents.registry import children_of, get_config
from app.graphs.nodes import _build_section_coverage_hint, _document_coverage
from app.models.artifact import Artifact, ArtifactStatus, ArtifactType


@pytest.mark.asyncio
async def test_document_coverage_three_of_seven_is_not_complete(db_session):
    project_id = uuid.uuid4()
    container = Artifact(
        project_id=project_id,
        type=ArtifactType.BRD,
        status=ArtifactStatus.DRAFT,
        title="BRD",
        extra_metadata={},
    )
    db_session.add(container)
    await db_session.flush()
    for item_type in children_of("brd")[:3]:
        db_session.add(
            Artifact(
                project_id=project_id,
                parent_id=container.id,
                type=ArtifactType(item_type),
                status=ArtifactStatus.ACCEPTED,
                title=get_config(item_type).label,
                extra_metadata={},
            )
        )
    await db_session.flush()

    coverage = await _document_coverage(
        db=db_session,
        project_id=project_id,
        artifact_type="vision_objectives",
        focused_artifact_id=None,
    )

    assert sum(value == "filled" for value in coverage["section_coverage"].values()) == 3
    assert coverage["coverage_complete"] is False


@pytest.mark.asyncio
async def test_document_coverage_zero_children_is_not_ready(db_session):
    project_id = uuid.uuid4()
    db_session.add(
        Artifact(
            project_id=project_id,
            type=ArtifactType.BRD,
            status=ArtifactStatus.DRAFT,
            title="BRD",
            extra_metadata={},
        )
    )
    await db_session.flush()

    coverage = await _document_coverage(
        db=db_session,
        project_id=project_id,
        artifact_type="vision_objectives",
        focused_artifact_id=None,
    )

    assert all(value == "missing" for value in coverage["section_coverage"].values())
    assert coverage["coverage_complete"] is False


def test_coverage_hint_uses_registry_descriptions():
    state = {
        "coverage_complete": False,
        "section_coverage": {
            item_type: "missing"
            for item_type in children_of("brd")
        },
        "section_coverage_stall_count": 0,
    }
    hint = _build_section_coverage_hint(state)
    assert get_config("vision_objectives").description in hint
    assert get_config("problem_statement").description in hint
