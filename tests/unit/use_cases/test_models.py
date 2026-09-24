"""Regression tests for the `_accept_aliases` before-validators in app/use_cases/models.py.

These validators exist so a draft/LLM payload using the FE-facing camelCase shape
(participantType, flowType, ...) still validates against the canonical snake_case model. Two of
them (UseCaseFlowStep, UseCaseFlow) copied the camelCase value onto the snake_case field with
setdefault() but never removed the original camelCase key -- harmless under a lenient model, but
every one of these models is `extra="forbid"`, so the leftover camelCase key was rejected as an
unrecognized field. This broke real generations: any use case whose main/alternative/exception
flow the LLM wrote with participantType/participantId (its only documented shape) failed with a
"Could not parse" 502 immediately after a valid LLM response, one `_draft_row` call away from
being saved.
"""

from app.use_cases.models import UseCaseEntry, UseCaseFlow, UseCaseFlowStep


def test_use_case_flow_step_accepts_camel_case_participant_fields():
    step = UseCaseFlowStep.model_validate(
        {"step": 1, "participantType": "actor", "participantId": "ACT-MANAGER", "action": "Opens the form."}
    )

    assert step.participant_type == "actor"
    assert step.participant_id == "ACT-MANAGER"


def test_use_case_flow_accepts_camel_case_flow_type_and_branch_at_step():
    flow = UseCaseFlow.model_validate(
        {
            "flowType": "alternative",
            "label": "Skip approval",
            "branchAtStep": 2,
            "steps": [{"step": 1, "participantType": "system", "action": "Auto-approves the request."}],
        }
    )

    assert flow.flow_type == "alternative"
    assert flow.branch_at_step == 2
    assert flow.steps[0].participant_type == "system"


def test_use_case_entry_validates_with_camel_case_flow_steps():
    """End-to-end shape of what UseCaseService._draft_row hands to UseCaseEntry: a main_flow
    list of dicts using the same camelCase keys the harness prompts the LLM to return."""
    entry = UseCaseEntry.model_validate(
        {
            "id": "UC-TASK-001",
            "name": "Create recurring task template",
            "module_id": "SUB-TEMPLATE-RECURRING-PROCESS",
            "primary_actor_id": "ACT-MANAGER",
            "description": "A manager defines a template that spawns tasks on a schedule.",
            "preconditions": ["A project exists."],
            "main_flow": [
                {"step": 1, "participantType": "actor", "participantId": "ACT-MANAGER", "action": "Opens the form."},
                {"step": 2, "participantType": "system", "action": "Saves the template."},
            ],
            "alternative_flows": [],
            "exception_flows": [],
            "priority": "required",
            "evidence": "explicit",
            "source_refs": ["entity:prd:use_case:C1:5"],
        }
    )

    assert [step.participant_type for step in entry.main_flow] == ["actor", "system"]
    assert entry.main_flow[0].participant_id == "ACT-MANAGER"
