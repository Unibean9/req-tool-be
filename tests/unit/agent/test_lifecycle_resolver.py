import inspect

from app.graphs import lifecycle_resolver
from app.graphs.lifecycle_resolver import (
    ArtifactLifecycleSnapshot,
    BasedOnPredecessorSnapshot,
    LifecycleAction,
    LifecycleState,
    StructuralPredecessorSnapshot,
    resolve_lifecycle,
)
from app.models.artifact import ArtifactStatus


def _artifact(
    *,
    status: ArtifactStatus = ArtifactStatus.ACCEPTED,
    version_id: str = "version-current",
    based_on: dict[str, str] | None = None,
) -> ArtifactLifecycleSnapshot:
    return ArtifactLifecycleSnapshot(
        artifact_type="vision_objectives",
        artifact_id="artifact-vision",
        status=status,
        current_version_id=version_id,
        based_on=based_on or {},
    )


def test_missing_artifact_allows_create():
    verdict = resolve_lifecycle("vision_objectives", None)

    assert verdict.state == LifecycleState.MISSING
    assert verdict.allowed_actions == frozenset({LifecycleAction.CREATE})
    assert "not created yet" in verdict.reason


def test_blocked_when_structural_predecessor_is_not_accepted():
    verdict = resolve_lifecycle(
        "vision_objectives",
        None,
        required_predecessors=[
            StructuralPredecessorSnapshot(
                artifact_type="problem_statement",
                artifact_id="artifact-problem",
                status=ArtifactStatus.DRAFT,
                current_version_id="version-problem",
            )
        ],
    )

    assert verdict.state == LifecycleState.BLOCKED
    assert verdict.allowed_actions == frozenset({LifecycleAction.ELICIT, LifecycleAction.ASK})
    assert verdict.blockers == ("problem_statement",)


def test_in_progress_when_current_artifact_is_not_accepted():
    verdict = resolve_lifecycle(
        "vision_objectives",
        _artifact(status=ArtifactStatus.DRAFT),
        required_predecessors=[
            StructuralPredecessorSnapshot(
                artifact_type="problem_statement",
                artifact_id="artifact-problem",
                status=ArtifactStatus.ACCEPTED,
                current_version_id="version-problem",
            )
        ],
    )

    assert verdict.state == LifecycleState.IN_PROGRESS
    assert verdict.allowed_actions == frozenset({LifecycleAction.AMEND})


def test_current_when_accepted_and_based_on_matches_live_predecessor():
    verdict = resolve_lifecycle(
        "vision_objectives",
        _artifact(based_on={"artifact-problem": "version-problem"}),
        required_predecessors=[
            StructuralPredecessorSnapshot(
                artifact_type="problem_statement",
                artifact_id="artifact-problem",
                status=ArtifactStatus.ACCEPTED,
                current_version_id="version-problem",
            )
        ],
        based_on_predecessors={
            "artifact-problem": BasedOnPredecessorSnapshot(
                artifact_id="artifact-problem",
                artifact_type="problem_statement",
                status=ArtifactStatus.ACCEPTED,
                current_version_id="version-problem",
            )
        },
    )

    assert verdict.state == LifecycleState.CURRENT
    assert verdict.allowed_actions == frozenset()
    assert "match live versions" in verdict.reason


def test_stale_when_based_on_predecessor_moved():
    verdict = resolve_lifecycle(
        "vision_objectives",
        _artifact(based_on={"artifact-problem": "version-old"}),
        based_on_predecessors={
            "artifact-problem": BasedOnPredecessorSnapshot(
                artifact_id="artifact-problem",
                artifact_type="problem_statement",
                status=ArtifactStatus.ACCEPTED,
                current_version_id="version-new",
            )
        },
    )

    assert verdict.state == LifecycleState.STALE
    assert verdict.allowed_actions == frozenset({LifecycleAction.RECONCILE})
    assert "version-old to version-new" in verdict.reason


def test_orphan_when_based_on_predecessor_was_deleted():
    verdict = resolve_lifecycle(
        "vision_objectives",
        _artifact(based_on={"artifact-problem": "version-problem"}),
        based_on_predecessors={
            "artifact-problem": BasedOnPredecessorSnapshot(
                artifact_id="artifact-problem",
                artifact_type="problem_statement",
                found=False,
            )
        },
    )

    assert verdict.state == LifecycleState.ORPHAN
    assert verdict.allowed_actions == frozenset({LifecycleAction.RETIRE, LifecycleAction.RELINK})
    assert "was not found" in verdict.reason


def test_orphan_when_based_on_predecessor_was_retired():
    verdict = resolve_lifecycle(
        "vision_objectives",
        _artifact(based_on={"artifact-problem": "version-problem"}),
        based_on_predecessors={
            "artifact-problem": BasedOnPredecessorSnapshot(
                artifact_id="artifact-problem",
                artifact_type="problem_statement",
                status=ArtifactStatus.ARCHIVED,
                current_version_id="version-problem",
            )
        },
    )

    assert verdict.state == LifecycleState.ORPHAN
    assert "is retired" in verdict.reason


def test_restore_to_based_on_version_is_current_not_stale():
    verdict = resolve_lifecycle(
        "vision_objectives",
        _artifact(based_on={"artifact-problem": "version-restored"}),
        based_on_predecessors={
            "artifact-problem": BasedOnPredecessorSnapshot(
                artifact_id="artifact-problem",
                artifact_type="problem_statement",
                status=ArtifactStatus.ACCEPTED,
                current_version_id="version-restored",
            )
        },
    )

    assert verdict.state == LifecycleState.CURRENT


def test_resolver_module_does_not_import_database_or_io_primitives():
    source = inspect.getsource(lifecycle_resolver)

    assert "AsyncSession" not in source
    assert "Session" not in source
    assert "select(" not in source
    assert "sqlalchemy" not in source
