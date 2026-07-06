"""Jsonschema adoption parity + per-turn diagnosis judge budget.

Parity: the production `_validate_json_schema` (now backed by the `jsonschema` library) must accept
every payload the old hand-rolled walker accepted. Divergences are only allowed where the new
validator is STRICTER because the old one had a genuine hole (documented below and asserted).

Budget: the diagnosis judge budget resets on every human resume, so successive turns each get their
own escalation allowance.
"""

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.services.llm_clients import _validate_json_schema  # new, jsonschema-backed


# --- embedded copy of the DELETED hand-rolled walker (the parity baseline) ---


def _old_matches_json_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_old_matches_json_type(value, item) for item in expected_type)
    if expected_type == "null":
        return value is None
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _old_validate(value: Any, schema: dict, path: str = "$") -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _old_matches_json_type(value, expected_type):
        raise ValueError(f"wrong type at {path}")
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise ValueError(f"wrong enum at {path}")
    if value is None:
        return
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"missing {missing[0]} at {path}")
        if schema.get("additionalProperties") is False:
            extra = [key for key in value if key not in properties]
            if extra:
                raise ValueError(f"extra {extra[0]} at {path}")
        for key, child_schema in properties.items():
            if key in value:
                _old_validate(value[key], child_schema, f"{path}.{key}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _old_validate(item, schema["items"], f"{path}[{index}]")


def _accepts(fn, value, schema) -> bool:
    try:
        fn(value, schema)
        return True
    except ValueError:
        return False


# Normalized-shape schemas (additionalProperties=False, nullable via type lists) mirroring what
# _normalize_json_schema emits before validation, plus representative valid payloads.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "findings": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "findings", "suggestions"],
    "additionalProperties": False,
}
_NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "n": {"type": ["integer", "null"]}},
                "required": ["name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

_PARITY_CORPUS = [
    (_JUDGE_SCHEMA, {"score": 0.0, "findings": [], "suggestions": []}),
    (_JUDGE_SCHEMA, {"score": 1.0, "findings": ["a", "b"], "suggestions": ["c"]}),
    (_JUDGE_SCHEMA, {"score": 0.42, "findings": ["thiếu metric"], "suggestions": ["thêm KPI"]}),
    (_NESTED_SCHEMA, {"items": []}),
    (_NESTED_SCHEMA, {"items": [{"name": "x", "n": 3}, {"name": "y", "n": None}]}),
    (_NESTED_SCHEMA, {"items": [{"name": "only-required"}]}),
]


@pytest.mark.parametrize("schema,payload", _PARITY_CORPUS)
def test_new_validator_accepts_everything_old_accepted(schema, payload):
    assert _accepts(_old_validate, payload, schema), "corpus payload should be old-valid by construction"
    assert _accepts(_validate_json_schema, payload, schema), "new validator must accept old-valid payloads"


def test_new_validator_closes_numeric_bound_hole():
    # score out of [0,1]: the old walker ignored minimum/maximum (silent pass -> degraded output),
    # the jsonschema-backed validator rejects it with a precise error.
    bad = {"score": 1.5, "findings": [], "suggestions": []}
    assert _accepts(_old_validate, bad, _JUDGE_SCHEMA) is True
    assert _accepts(_validate_json_schema, bad, _JUDGE_SCHEMA) is False


def test_new_validator_still_rejects_shared_violations():
    # Missing required, wrong type, and extra key are rejected by BOTH validators.
    for bad in (
        {"findings": [], "suggestions": []},  # missing score
        {"score": "high", "findings": [], "suggestions": []},  # wrong type
        {"score": 0.5, "findings": [], "suggestions": [], "extra": 1},  # additionalProperties
    ):
        assert _accepts(_old_validate, bad, _JUDGE_SCHEMA) is False
        assert _accepts(_validate_json_schema, bad, _JUDGE_SCHEMA) is False


def test_error_is_valueerror_with_location():
    with pytest.raises(ValueError) as exc:
        _validate_json_schema({"score": 5, "findings": [], "suggestions": []}, _JUDGE_SCHEMA)
    assert "JSON Schema" in str(exc.value)


# --- per-turn diagnosis judge budget -----------------------------------------


def test_resume_resets_diagnosis_judge_budget(db_session):
    from contextlib import asynccontextmanager

    from app.models.agent import AgentSession
    from app.services.agent_service import AgentService

    @asynccontextmanager
    async def _sf():
        yield db_session

    svc = AgentService(db=db_session, graph=MagicMock(), session_factory=_sf)
    session = AgentSession(
        project_id=uuid.uuid4(), artifact_type="goal", workflow_area="analysis", graph_checkpoint={}
    )

    command = svc._resume_command(session, {"content": "tiếp"})

    # Per-turn budget: each human resume clears the counter so the next turn can escalate again.
    assert command.update["diagnosis_judge_calls_used"] == 0
    assert command.update["turn_count"] == 0
