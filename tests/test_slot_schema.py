from app.models.artifact import ArtifactType


def test_all_brd_keys_present():
    from app.graphs.slot_schema import BRD_SLOTS

    assert set(BRD_SLOTS) == {
        "intent",
        "problem",
        "goal",
        "stakeholder",
        "capability",
        "constraint",
        "assumption",
        "risk",
        "open_question",
    }


def test_brd_keys_are_valid_artifact_types():
    from app.graphs.slot_schema import BRD_SLOTS

    artifact_type_values = {item.value for item in ArtifactType}

    assert set(BRD_SLOTS).issubset(artifact_type_values)


def test_each_key_has_required_and_optional():
    from app.graphs.slot_schema import BRD_SLOTS

    for slot_spec in BRD_SLOTS.values():
        assert isinstance(slot_spec["required"], list)
        assert slot_spec["required"]
        assert isinstance(slot_spec["optional"], list)


def test_problem_required_slots_match_spec():
    from app.graphs.slot_schema import BRD_SLOTS

    assert BRD_SLOTS["problem"]["required"] == ["who", "obstacle", "root_cause", "frequency", "impact"]


def test_intent_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # JTBD (message.txt §1): why_now, sponsor, expected_outcome, success_state.
    assert BRD_SLOTS["intent"]["required"] == ["why_now", "sponsor", "expected_outcome", "success_state"]


def test_goal_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # SMART/OKR (message.txt §3): business + user goal, metric, target, timeframe.
    assert BRD_SLOTS["goal"]["required"] == ["business_goal", "user_goal", "metric", "target", "timeframe"]


def test_stakeholder_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # RACI (message.txt §4): primary user, secondary stakeholders, decision maker, operator.
    assert BRD_SLOTS["stakeholder"]["required"] == [
        "primary_user",
        "secondary_stakeholders",
        "decision_maker",
        "operator",
    ]


def test_capability_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # MoSCoW (message.txt §9): capability, description, priority, availability.
    assert BRD_SLOTS["capability"]["required"] == ["capability", "description", "priority", "availability"]


def test_constraint_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # NFR mapping (message.txt §5): time, budget, technical, people, compliance.
    assert BRD_SLOTS["constraint"]["required"] == ["time", "budget", "technical", "people", "compliance"]


def test_assumption_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # Assumption Mapping (message.txt §6): user, market, technical feasibility, riskiest, validation.
    assert BRD_SLOTS["assumption"]["required"] == [
        "user_behavior",
        "market",
        "technical_feasibility",
        "riskiest",
        "validation",
    ]


def test_risk_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # Risk Matrix + Pre-mortem (message.txt §7): risk, likelihood, impact, mitigation.
    assert BRD_SLOTS["risk"]["required"] == ["risk", "likelihood", "impact", "mitigation"]


def test_open_question_required_slots():
    from app.graphs.slot_schema import BRD_SLOTS

    # Starbursting (message.txt §8): question, domain, decision needed, research needed, blocker.
    assert BRD_SLOTS["open_question"]["required"] == [
        "question",
        "domain",
        "decision_needed",
        "research_needed",
        "blocker",
    ]


def test_all_required_lists_within_optimal_size():
    from app.graphs.slot_schema import BRD_SLOTS

    # Keep elicitation bounded: 3-5 required dimensions per key (plan Phase 5 risk).
    for key, slot_spec in BRD_SLOTS.items():
        assert 3 <= len(slot_spec["required"]) <= 5, key


def test_coverage_threshold_in_range():
    from app.graphs.slot_schema import COVERAGE_THRESHOLD

    for threshold in COVERAGE_THRESHOLD.values():
        assert 0.0 < threshold <= 1.0


def test_slot_descriptions_cover_all_slots():
    from app.graphs.slot_schema import BRD_SLOTS, SLOT_DESCRIPTIONS

    all_slots = set()
    for slot_spec in BRD_SLOTS.values():
        all_slots.update(slot_spec["required"])
        all_slots.update(slot_spec["optional"])

    assert all_slots.issubset(SLOT_DESCRIPTIONS)
