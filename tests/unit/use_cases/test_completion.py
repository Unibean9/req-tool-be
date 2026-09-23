from app.use_cases.completion import complete_use_case_table
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    SourceEvidence,
    StoredComponentSnapshot,
    StoredDocumentSnapshot,
    UseCaseModel,
)
from app.use_cases.rules import validate_use_case_model


def _source() -> RequirementsSourceSnapshot:
    stakeholder_body = """
## Stakeholders
| role | responsibility |
| --- | --- |
| Researcher / Data Analyst | Runs research |
| Research Project Manager | Owns the project |
| System Administrator | Maintains access |
""".strip()
    capability_body = """
## Business Capabilities

### BC-01: Research Workspace and Access Control
- **goal:** Establish a governed project workspace.
- **user_segment:** Researcher / Data Analyst, Research Project Manager, System Administrator
- **business_value:** Keeps project work governed.
- **scope:** Authentication and project management.

### BC-02: Dataset Understanding and Quality
- **goal:** Upload and validate research datasets.
- **user_segment:** Researcher / Data Analyst
- **business_value:** Improves data readiness.
- **scope:** Dataset validation.
""".strip()
    functional_body = """
## Functional Requirements
| id | requirement | behavior | inputs/outputs | acceptance signal | priority | dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| FR-01 | BC-01 · FR-AUTH-01 — Login | System shall support authenticated login. | Input | Access is denied when unauthorized. | P0 | None |
| FR-02 | BC-01 · FR-PROJ-01 — Create Project | Project owner can create and edit metadata. | Input | Project is available. | P0 | None |
| FR-03 | BC-02 · FR-DATA-01 — Upload | System shall accept a supported dataset. | Input | Dataset is available. | P0 | None |
""".strip()
    brd = StoredDocumentSnapshot(
        document_type="brd",
        label="BRD",
        components=[
            StoredComponentSnapshot(
                document_type="brd",
                artifact_type="stakeholder_register",
                label="Stakeholders",
                body=stakeholder_body,
            )
        ],
    )
    prd = StoredDocumentSnapshot(
        document_type="prd",
        label="PRD",
        components=[
            StoredComponentSnapshot(
                document_type="prd",
                artifact_type="use_case",
                label="Business Capabilities",
                body=capability_body,
            ),
            StoredComponentSnapshot(
                document_type="prd",
                artifact_type="functional_requirement",
                label="Functional Requirements",
                body=functional_body,
            ),
        ],
    )
    evidence = [
        SourceEvidence(
            evidence_id="component:brd:stakeholder_register",
            document_type="brd",
            artifact_type="stakeholder_register",
            kind="component",
            locator="brd.stakeholder_register",
            excerpt=stakeholder_body,
        ),
        SourceEvidence(
            evidence_id="component:prd:use_case",
            document_type="prd",
            artifact_type="use_case",
            kind="component",
            locator="prd.use_case",
            excerpt=capability_body,
        ),
        SourceEvidence(
            evidence_id="component:prd:functional_requirement",
            document_type="prd",
            artifact_type="functional_requirement",
            kind="component",
            locator="prd.functional_requirement",
            excerpt=functional_body,
        ),
    ]
    for bc_id, line in (("BC-01", 3), ("BC-02", 9)):
        evidence.append(
            SourceEvidence(
                evidence_id=f"entity:prd:use_case:{bc_id}:{line}",
                document_type="prd",
                artifact_type="use_case",
                kind="business_capability",
                locator=f"prd.use_case:L{line}",
                excerpt=bc_id,
                entity_id=bc_id,
                entity_name=bc_id,
            )
        )
    for code, line, _bc in (("FR-AUTH-01", 5, "BC-01"), ("FR-PROJ-01", 6, "BC-01"), ("FR-DATA-01", 7, "BC-02")):
        evidence.append(
            SourceEvidence(
                evidence_id=f"entity:prd:functional_requirement:{code}:{line}",
                document_type="prd",
                artifact_type="functional_requirement",
                kind="functional_requirement",
                locator=f"prd.functional_requirement:L{line}",
                excerpt=code,
                entity_id=code,
                entity_name=code,
            )
        )
    return RequirementsSourceSnapshot(
        project_id="project",
        brd=brd,
        prd=prd,
        components=[*brd.components, *prd.components],
        evidence=evidence,
        source_hash="test-source",
    )


def test_completion_enumerates_capabilities_and_requirement_families_with_actor_associations():
    source = _source()
    model = complete_use_case_table(source)

    assert len(model.use_cases) == 5
    assert sum(item.level == "L0" for item in model.use_cases) == 2
    assert sum(item.level == "L1" for item in model.use_cases) == 3
    assert len(model.actors) == 3
    assert all(
        any(
            relation.kind == "association"
            and {relation.source_id, relation.target_id} == {actor_id, item.id}
            for relation in model.relations
        )
        for item in model.use_cases
        for actor_id in [item.primary_actor_id, *item.secondary_actor_ids]
    )
    report = validate_use_case_model(model, source)
    assert report.errors == []


def test_completion_replaces_short_llm_enumeration_without_fabricating_include_or_extend():
    source = _source()
    candidate = UseCaseModel.model_validate(
        {
            "system_name": "Demo",
            "actors": [],
            "subsystems": [],
            "use_cases": [],
            "relations": [],
            "diagrams": [],
        }
    )

    model = complete_use_case_table(source, candidate)

    assert len(model.use_cases) == 5
    assert not [relation for relation in model.relations if relation.kind in {"include", "extend"}]
