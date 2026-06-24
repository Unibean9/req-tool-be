"""Extract structured analytical objects from note-tool content (spec §7.1).

Pure Python, no LLM. Note tools (critique_note / explore_note) emit free text that may carry one
or more tagged lines:

    ASSUMPTION: <statement> | source: x | confidence: y | impact: z | owner: w | status: s
    RISK: <statement> | likelihood: x | impact: y | mitigation: z | owner: w | status: s
    OPEN_QUESTION: <question> | domain: x | decision_needed: y | status: z

The leading segment after the prefix is the statement/question; the remaining ` | `-separated
segments are ``key: value`` pairs. Unrecognized lines parse to nothing — graceful degradation, so
free-form notes never crash or fabricate empty objects.
"""

from typing import Any

# prefix -> (head_field, [tail_fields]). head_field takes the first segment; tail_fields are
# filled from key:value pairs, defaulting to "" when absent.
_OBJECT_SPECS: dict[str, tuple[str, list[str]]] = {
    "ASSUMPTION": ("statement", ["source", "confidence", "impact", "owner", "status"]),
    "RISK": ("statement", ["likelihood", "impact", "mitigation", "owner", "status"]),
    "OPEN_QUESTION": ("question", ["domain", "decision_needed", "status"]),
    "KEY_FACT": ("statement", ["source", "turn"]),
}

_PREFIX_TO_BUCKET = {
    "ASSUMPTION": "assumptions",
    "RISK": "risks",
    "OPEN_QUESTION": "open_questions",
    "KEY_FACT": "key_facts",
}


def extract_structured_objects(content: str) -> dict[str, list[dict[str, Any]]]:
    """Parse tagged lines in `content` into assumption / risk / open_question / key_fact objects."""
    result: dict[str, list[dict[str, Any]]] = {"assumptions": [], "risks": [], "open_questions": [], "key_facts": []}
    for line in (content or "").splitlines():
        line = line.strip()
        prefix, _, remainder = line.partition(":")
        spec = _OBJECT_SPECS.get(prefix)
        if spec is None or not remainder.strip():
            continue
        head_field, tail_fields = spec
        segments = [seg.strip() for seg in remainder.split("|")]
        obj = {field: "" for field in tail_fields}
        obj[head_field] = segments[0]
        for segment in segments[1:]:
            key, sep, value = segment.partition(":")
            if sep and key.strip() in obj:
                obj[key.strip()] = value.strip()
        result[_PREFIX_TO_BUCKET[prefix]].append(obj)
    return result
