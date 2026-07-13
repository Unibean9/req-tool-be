"""Registry and instruction round-trip for the event_storming container and its item types.

Locks the container/children shape, the per-item output contracts (id_prefix/render_style/
elicit/review profile), and role resolution for the five new artifact types, without disturbing
any pre-existing container or item lookup.
"""

from app.documents.registry import (
    all_container_types,
    all_item_types,
    children_of,
    container_for,
    item_configs,
    output_contract,
)
from app.instructions import ARTIFACT_ROLE_MAP, get_instruction, load_instructions

_ITEM_TYPES = ("domain_event", "actor_command", "policy", "aggregate")


def test_event_storming_registered_as_container_with_four_children():
    assert "event_storming" in all_container_types()
    assert children_of("event_storming") == _ITEM_TYPES
    for item_type in _ITEM_TYPES:
        assert container_for(item_type) == "event_storming"
        assert item_type in all_item_types()


def test_event_storming_item_configs_round_trip():
    configs = item_configs("event_storming")
    assert [config.artifact_type for config in configs] == list(_ITEM_TYPES)


def test_domain_event_contract_has_hotspots_and_no_data_schema():
    contract = output_contract("domain_event")
    assert "## Domain Events" in contract.required_headings
    assert "## Hotspots" in contract.required_headings
    assert contract.id_prefix == "EVT"
    assert contract.render_style == "entries"
    assert "schema" not in " ".join(contract.table_columns).lower()
    blob = " ".join((contract.guidance, *contract.elicit_checklist, *contract.review_criteria))
    assert "past-tense" in blob or "past tense" in blob


def test_actor_command_contract_shape():
    contract = output_contract("actor_command")
    assert contract.id_prefix == "CMD"
    assert contract.render_style == "table"
    assert contract.table_columns == ("id", "actor", "command", "precondition", "resulting event", "flow")


def test_policy_contract_shape():
    contract = output_contract("policy")
    assert contract.id_prefix == "POL"
    assert contract.render_style == "entries"


def test_aggregate_contract_shape():
    contract = output_contract("aggregate")
    assert contract.id_prefix == "AGG"
    assert contract.render_style == "entries"


def test_each_item_has_its_own_scoped_elicit_checklist():
    checklists = [output_contract(item_type).elicit_checklist for item_type in _ITEM_TYPES]
    for checklist in checklists:
        assert checklist
    # No two items share the same checklist.
    assert len({checklist for checklist in checklists}) == len(checklists)


def test_existing_registry_lookups_unaffected():
    """Pre-existing containers and items are untouched by the additive registration."""
    assert "brd" in all_container_types()
    assert "prd" in all_container_types()
    assert "add" in all_container_types()
    assert children_of("add") == ("tech_stack", "domain_entity", "component", "interface", "tech_decision")
    assert output_contract("use_case").id_prefix == "BC"


def test_event_storming_role_map_entries():
    for artifact_type in ("event_storming", *_ITEM_TYPES):
        assert ARTIFACT_ROLE_MAP[artifact_type] == "product_manager"


def test_get_instruction_resolves_product_manager_for_event_storming():
    load_instructions()
    instruction = get_instruction(artifact_type="event_storming", workflow_area="analysis", agent_role=None)
    assert instruction is not None
    assert "Product Manager" in instruction
