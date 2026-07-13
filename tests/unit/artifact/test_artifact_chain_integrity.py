"""Chain-integrity invariants for ARTIFACT_PREDECESSORS.

Locks the artifact chain against drift: every registry child has a predecessor
entry, no accidental dead-end leaves, the map is acyclic, and it references
only real registry types.
"""

from app.documents.registry import all_container_types, all_item_types, children_of
from app.graphs.policy import ARTIFACT_PREDECESSORS, ancestor_types

# Types that are intentionally terminal — nothing downstream consumes them.
_INTENTIONAL_LEAVES = {
    "add",
    "interface",
    "tech_decision",
    "tech_stack",
    "stakeholder_register",
    "policy",
    "aggregate",
}


def _all_registry_types() -> set[str]:
    return set(all_container_types()) | set(all_item_types())


def test_every_registry_child_is_a_map_key():
    """No orphans: every child type in the registry has a predecessor entry."""
    missing = [
        child
        for container in all_container_types()
        for child in children_of(container)
        if child not in ARTIFACT_PREDECESSORS
    ]
    assert not missing, f"registry children missing from chain map: {sorted(set(missing))}"


def test_map_keys_and_values_are_real_registry_types():
    valid = _all_registry_types()
    bad_keys = sorted(k for k in ARTIFACT_PREDECESSORS if k not in valid)
    bad_values = sorted(
        {v for preds in ARTIFACT_PREDECESSORS.values() for v in preds if v not in valid}
    )
    assert not bad_keys, f"map keys not in registry: {bad_keys}"
    assert not bad_values, f"predecessor values not in registry: {bad_values}"


def test_every_type_is_consumed_or_an_intentional_leaf():
    consumed = {v for preds in ARTIFACT_PREDECESSORS.values() for v in preds}
    leaves = {k for k in ARTIFACT_PREDECESSORS if k not in consumed}
    unexpected = sorted(leaves - _INTENTIONAL_LEAVES)
    assert not unexpected, f"dead-end types not on the leaf allowlist: {unexpected}"
    stale = sorted(_INTENTIONAL_LEAVES & consumed)
    assert not stale, f"allowlisted leaves that are actually consumed: {stale}"


def test_chain_is_acyclic():
    for artifact_type in ARTIFACT_PREDECESSORS:
        assert artifact_type not in ancestor_types(artifact_type), (
            f"{artifact_type} is its own ancestor — cycle in ARTIFACT_PREDECESSORS"
        )


def test_predecessors_match_intended_pre_es_shape():
    """Representative edges of the intended chain shape (brainstorm §4.2)."""
    assert ARTIFACT_PREDECESSORS["problem_statement"] == []
    assert ARTIFACT_PREDECESSORS["vision_objectives"] == ["problem_statement"]
    assert ARTIFACT_PREDECESSORS["use_case"] == ["scope_capabilities"]
    assert ARTIFACT_PREDECESSORS["functional_requirement"] == ["use_case", "business_rules"]
    assert ARTIFACT_PREDECESSORS["non_functional_requirement"] == ["constraints_assumptions"]
    assert ARTIFACT_PREDECESSORS["domain_entity"] == ["functional_requirement"]
    assert ARTIFACT_PREDECESSORS["add"] == ["event_storming"]


def test_ancestor_chain_routes_add_through_event_storming():
    assert "event_storming" in ancestor_types("add")
    assert "prd" in ancestor_types("event_storming")
    assert "brd" in ancestor_types("event_storming")
