import pytest
from sqlalchemy import inspect

from app.models.artifact import (
    Artifact,
    ArtifactEvidence,
    ArtifactLink,
    ArtifactPriority,
    ArtifactReview,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    ChangeSource,
    EvidenceSourceType,
    RelationType,
    ReviewStatus,
    SourceDocument,
    SourceType,
    VersionStatus,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStep,
    WorkflowStepKey,
    WorkflowStepPhase,
    WorkflowStepStatus,
)


@pytest.mark.asyncio
async def test_artifact_tables_and_promoted_columns_exist(db_session):
    async with db_session.bind.connect() as conn:
        table_names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
        columns = await conn.run_sync(
            lambda sync_conn: {
                column["name"] for column in inspect(sync_conn).get_columns("artifacts")
            }
        )

    assert {
        "artifacts",
        "artifact_versions",
        "source_documents",
        "artifact_links",
        "artifact_evidence",
        "artifact_reviews",
        "workflow_runs",
        "workflow_steps",
    }.issubset(table_names)
    assert "nfr_category" not in columns
    assert "stakeholder_role" not in columns


@pytest.mark.asyncio
async def test_artifact_repository_required_columns_match_contract(db_session):
    async with db_session.bind.connect() as conn:
        artifact_columns = await conn.run_sync(
            lambda sync_conn: {column["name"]: column for column in inspect(sync_conn).get_columns("artifacts")}
        )
        version_columns = await conn.run_sync(
            lambda sync_conn: {column["name"]: column for column in inspect(sync_conn).get_columns("artifact_versions")}
        )
        source_columns = await conn.run_sync(
            lambda sync_conn: {column["name"]: column for column in inspect(sync_conn).get_columns("source_documents")}
        )
        workflow_run_columns = await conn.run_sync(
            lambda sync_conn: {column["name"]: column for column in inspect(sync_conn).get_columns("workflow_runs")}
        )
        workflow_step_columns = await conn.run_sync(
            lambda sync_conn: {column["name"]: column for column in inspect(sync_conn).get_columns("workflow_steps")}
        )

    assert {
        "parent_id",
        "run_id",
        "step_id",
        "code",
        "confidence",
        "created_by_id",
    }.issubset(artifact_columns)
    assert artifact_columns["priority"]["nullable"] is True
    assert {
        "change_summary",
        "parent_version_id",
        "created_by_id",
        "model_name",
    }.issubset(version_columns)
    assert version_columns["body"]["nullable"] is False
    assert version_columns["metadata"]["nullable"] is False
    assert {"title", "locator", "content_text", "size_bytes", "uploaded_by_id"}.issubset(source_columns)
    assert {"name", "current_step_key", "created_by_id"}.issubset(workflow_run_columns)
    assert {
        "run_id",
        "project_id",
        "input_snapshot",
        "output_snapshot",
        "approved_at",
        "approved_by_id",
    }.issubset(workflow_step_columns)


def test_artifact_enum_values_are_constrained():
    enum_expectations = {
        ArtifactStatus: {"draft", "needs_clarification", "accepted", "rejected", "archived"},
        ArtifactPriority: {"must", "should", "could", "wont"},
        ArtifactType: {
            "brd",
            "prd",
            "sad",
            "vision_objectives",
            "problem_statement",
            "stakeholder_register",
            "scope_capabilities",
            "business_rules",
            "constraints_assumptions",
            "risks_issues",
            "domain_entity",
            "functional_requirement",
            "non_functional_requirement",
            "use_case",
            "component",
            "interface",
            "tech_decision",
            "epic",
            "story",
            "acceptance_criteria",
        },
        ChangeSource: {"manual", "import", "ai_generation", "tool_call", "system"},
        VersionStatus: {"draft", "proposed", "accepted", "rejected", "archived"},
        SourceType: {"markdown_upload", "text_paste", "repo_file", "url", "external_doc"},
        RelationType: {
            "derives_from",
            "decomposes_to",
            "satisfies",
            "supports",
            "depends_on",
            "blocks",
            "conflicts_with",
            "clarifies",
            "constrains",
            "informs",
            "validates",
        },
        EvidenceSourceType: {"chat", "repo_file", "document", "url", "user_input", "ai_output"},
        ReviewStatus: {"approved", "rejected", "changes_requested"},
        WorkflowRunStatus: {"draft", "active", "completed", "archived"},
        WorkflowStepKey: {
            "intent_vision",
            "capability_map",
            "domain_model",
            "requirements_spec",
            "realization_backlog",
        },
        WorkflowStepPhase: {"brd", "prd", "delivery"},
        WorkflowStepStatus: {
            "pending",
            "ready",
            "in_progress",
            "waiting_review",
            "approved",
            "rejected",
            "skipped",
        },
    }

    for enum_class, expected_values in enum_expectations.items():
        assert {item.value for item in enum_class} == expected_values
        with pytest.raises(ValueError):
            enum_class("không-hợp-lệ")


def test_artifact_model_imports_all_repository_tables():
    assert SourceDocument.__tablename__ == "source_documents"
    assert Artifact.__tablename__ == "artifacts"
    assert ArtifactVersion.__tablename__ == "artifact_versions"
    assert ArtifactLink.__tablename__ == "artifact_links"
    assert ArtifactEvidence.__tablename__ == "artifact_evidence"
    assert ArtifactReview.__tablename__ == "artifact_reviews"
    assert WorkflowRun.__tablename__ == "workflow_runs"
    assert WorkflowStep.__tablename__ == "workflow_steps"
