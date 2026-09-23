from app.use_cases.models import UseCaseActor, UseCaseEntry, UseCaseModel, UseCaseRelation, UseCaseSubsystem
from app.use_cases.plantuml import render_plantuml


def _model() -> UseCaseModel:
    return UseCaseModel(
        system_name="Research \"Platform\"",
        actors=[UseCaseActor(id="ACT-RESEARCHER", name="Researcher", source_refs=["e1"])],
        subsystems=[UseCaseSubsystem(id="SUB-01", name="Research", source_refs=["e1"])],
        use_cases=[
            UseCaseEntry(
                id="UC-SUM-01",
                name="Run experiment",
                level="L0",
                abstraction="summary",
                primary_actor_id="ACT-RESEARCHER",
                subsystem_id="SUB-01",
                description="Run an experiment.",
                precondition="A project exists.",
                priority="must",
                status="confirmed",
                source_refs=["e1"],
            ),
            UseCaseEntry(
                id="UC-PLAN-01",
                name="Plan experiment",
                level="L1",
                abstraction="user_goal",
                primary_actor_id="ACT-RESEARCHER",
                subsystem_id="SUB-01",
                parent_use_case_id="UC-SUM-01",
                description="Plan an experiment.",
                precondition="A project exists.",
                priority="must",
                status="confirmed",
                source_refs=["e1"],
            ),
        ],
        relations=[
            UseCaseRelation(id="REL-A", kind="association", source_id="ACT-RESEARCHER", target_id="UC-SUM-01"),
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
    source = render_plantuml(_model())

    assert source.startswith("@startuml")
    assert source.rstrip().endswith("@enduml")
    assert 'rectangle "Research \\\"Platform\\\""' in source
    assert "ACT_ACT_RESEARCHER -- UC_UC_SUM_01" in source
    assert "UC_UC_SUM_01 ..> UC_UC_PLAN_01 : <<include>>" in source
    assert "UC_UC_PLAN_01 ..> UC_UC_SUM_01 : <<extend>>" in source
    assert "extend condition (REL-E): When planning is optional." in source
    assert "hierarchy: UC-SUM-01 -> UC-PLAN-01" in source
