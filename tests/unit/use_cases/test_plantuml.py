from app.use_cases.models import UseCaseActor, UseCaseEntry, UseCaseModel, UseCaseModule, UseCaseRelation, UseCaseSystem
from app.use_cases.plantuml import render_plantuml


def _model() -> UseCaseModel:
    return UseCaseModel(
        system=UseCaseSystem(id="SYSTEM", name='Research "Platform"'),
        actors=[UseCaseActor(id="ACT-RESEARCHER", name="Researcher", source_refs=["e1"])],
        modules=[UseCaseModule(id="SUB-01", name="Research", source_refs=["e1"])],
        use_cases=[
            UseCaseEntry(
                id="UC-SUM-01",
                name="Run experiment",
                module_id="SUB-01",
                primary_actor_id="ACT-RESEARCHER",
                description="Run an experiment.",
                preconditions=["A project exists."],
                priority="required",
                evidence="explicit",
                source_refs=["e1"],
            ),
            UseCaseEntry(
                id="UC-PLAN-01",
                name="Plan experiment",
                module_id="SUB-01",
                primary_actor_id="ACT-RESEARCHER",
                description="Plan an experiment.",
                preconditions=["A project exists."],
                priority="required",
                evidence="explicit",
                source_refs=["e1"],
            ),
        ],
        relationships=[
            UseCaseRelation(id="REL-I", kind="include", source_id="UC-SUM-01", target_id="UC-PLAN-01"),
            UseCaseRelation(
                id="REL-E",
                kind="extend",
                source_id="UC-PLAN-01",
                target_id="UC-SUM-01",
                condition="When planning is optional.",
            ),
        ],
    )


def test_render_plantuml_preserves_table_ids_and_uml_relation_notation():
    """Actor association is derived from each use case's own actor ids, not a stored relation row
    (UseCaseRelation.kind no longer has an "association" value). There is no L0/L1 hierarchy to
    render either -- every use case is a flat row under its module."""
    source = render_plantuml(_model())

    assert source.startswith("@startuml")
    assert source.rstrip().endswith("@enduml")
    assert 'rectangle "Research \\\"Platform\\\""' in source
    assert "ACT_ACT_RESEARCHER -- UC_UC_SUM_01" in source
    assert "ACT_ACT_RESEARCHER -- UC_UC_PLAN_01" in source
    assert "UC_UC_SUM_01 ..> UC_UC_PLAN_01 : <<include>>" in source
    assert "UC_UC_PLAN_01 ..> UC_UC_SUM_01 : <<extend>>" in source
