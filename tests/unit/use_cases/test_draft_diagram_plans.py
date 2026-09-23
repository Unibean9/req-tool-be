from app.services.use_case_service import UseCaseService


def test_response_rebuilds_a_renderable_draft_plan_when_generation_stored_none():
    payload = {
        "projectId": "project-1",
        "projectName": "Research platform",
        "actors": [{"id": "ACT-RESEARCHER", "name": "Researcher", "kind": "Primary actor"}],
        "useCases": [
            {
                "id": "UC-RESEARCH",
                "level": "L0",
                "title": "Run experiment",
                "primaryActorId": "ACT-RESEARCHER",
                "supportingActorIds": [],
                "subsystem": "Research",
                "status": "Confirmed",
                "priority": "Must",
                "parentUseCaseId": None,
                "description": "Run a research experiment.",
                "precondition": "The research project exists.",
                "sourceTrace": [],
            }
        ],
        "relationships": [
            {
                "id": "REL-RESEARCH",
                "sourceId": "ACT-RESEARCHER",
                "targetId": "UC-RESEARCH",
                "type": "association",
                "condition": None,
            }
        ],
        "diagrams": [
            {
                "id": "DGM-L0",
                "level": "L0",
                "systemBoundary": "Research platform",
                "subsystem": None,
                "actorIds": ["ACT-RESEARCHER"],
                "useCaseIds": ["UC-RESEARCH"],
                "relationIds": ["REL-RESEARCH"],
            },
            {
                "id": "DGM-L1-RESEARCH",
                "level": "L1",
                "systemBoundary": "Research platform",
                "subsystem": "Research",
                "actorIds": ["ACT-RESEARCHER"],
                "useCaseIds": ["UC-RESEARCH"],
                "relationIds": ["REL-RESEARCH"],
            },
        ],
        "diagramPlans": [],
        "sourceHash": None,
        "validation": None,
        "generation": None,
    }

    response = UseCaseService(None)._response(payload)

    assert [plan.diagram_id for plan in response.diagram_plans] == ["DGM-L0", "DGM-L1-RESEARCH"]
    plan = response.diagram_plans[0]
    assert plan.diagram_id == "DGM-L0"
    assert plan.subsystem is None
    assert {node.kind for node in plan.nodes} == {"system_boundary", "actor", "use_case"}
    assert plan.edges[0].kind == "association"
    assert plan.edges[0].marker == "none"

    partial_payload = {
        **payload,
        "diagramPlans": [response.diagram_plans[0].model_dump(by_alias=True)],
    }
    partial_response = UseCaseService(None)._response(partial_payload)
    assert {plan.diagram_id for plan in partial_response.diagram_plans} == {"DGM-L0", "DGM-L1-RESEARCH"}
