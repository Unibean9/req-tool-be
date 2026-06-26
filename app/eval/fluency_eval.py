"""Eval: Harness Conversation Fluency.

Proves each defect-fix from the harness-conversation-fluency plan.
Mirrors the runtime_harness.py gate pattern: each scenario returns a
RuntimeGateResult; the runner aggregates and renders markdown.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Gate result — same shape as runtime_harness.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    score: float
    threshold: float
    critical: bool
    reason: str


# ---------------------------------------------------------------------------
# Key Facts State — note_parser TAG parsing + prompt injection
# ---------------------------------------------------------------------------

def _key_facts_note_parser_parses_tag() -> GateResult:
    from app.graphs.note_parser import extract_structured_objects

    content = "KEY_FACT: App sẽ hỗ trợ 10k DAU | source: user | turn: 2"
    result = extract_structured_objects(content)
    facts = result.get("key_facts", [])
    passed = (
        len(facts) == 1
        and facts[0].get("statement") == "App sẽ hỗ trợ 10k DAU"
        and facts[0].get("source") == "user"
    )
    return GateResult(
        gate="note_parser parses KEY_FACT tag",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="KEY_FACT tag extracted correctly" if passed else f"unexpected result: {facts}",
    )


def _key_facts_optional_fields_default_empty() -> GateResult:
    from app.graphs.note_parser import extract_structured_objects

    content = "KEY_FACT: DAU target là 10k"
    result = extract_structured_objects(content)
    facts = result.get("key_facts", [])
    passed = (
        len(facts) == 1
        and facts[0].get("source") == ""
        and facts[0].get("turn") == ""
    )
    return GateResult(
        gate="optional KEY_FACT fields default to empty string",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=False,
        reason="optional fields default correctly" if passed else f"fields: {facts[0] if facts else {}}",
    )


def _key_facts_injected_into_prompt() -> GateResult:
    from app.graphs.nodes import _build_tool_selection_prompt
    from tests.integration.test_graph_nodes import _state

    state = _state()
    state["key_facts"] = [{"statement": "Budget tối đa 500tr", "source": "cfo", "turn": "2"}]
    prompt = _build_tool_selection_prompt(state, [])
    passed = "Budget tối đa 500tr" in prompt
    return GateResult(
        gate="key_facts block injected into tool-selection prompt",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="key_facts visible in prompt" if passed else "key_facts absent from prompt",
    )


def _key_facts_empty_no_section_in_prompt() -> GateResult:
    from app.graphs.nodes import _build_tool_selection_prompt
    from tests.integration.test_graph_nodes import _state

    prompt = _build_tool_selection_prompt(_state(), [])
    passed = "KEY FACTS" not in prompt
    return GateResult(
        gate="empty key_facts → no KEY FACTS section in prompt",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=False,
        reason="section absent when empty" if passed else "KEY FACTS section present when it should not be",
    )


# ---------------------------------------------------------------------------
# Interrupt Semantics — STREAM_RESPONSE enum value; ask_user keeps session ACTIVE
# ---------------------------------------------------------------------------

def _interrupt_stream_response_in_enum() -> GateResult:
    from app.models.agent import AgentSessionInterruptType

    passed = "stream_response" in [e.value for e in AgentSessionInterruptType]
    return GateResult(
        gate="STREAM_RESPONSE value exists in AgentSessionInterruptType enum",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="STREAM_RESPONSE registered in DB enum" if passed else "STREAM_RESPONSE missing from enum",
    )


def _interrupt_ask_user_uses_stream_response() -> GateResult:
    """ask_user tool must call _save_and_interrupt_ask with interrupt_kind='stream_response'."""
    from app.graphs.agent_tools import _ask_user_impl

    called_with: dict[str, Any] = {}

    async def _fake_save(state, config, content, *, run_id, kind="question", mode=None, interrupt_kind="ask_human"):
        called_with["interrupt_kind"] = interrupt_kind
        called_with["kind"] = kind
        return "ok"

    async def run():
        state = {"messages": [], "user_confirmed": None}
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        with patch("app.graphs.agent_tools.nodes._save_and_interrupt_ask", new=AsyncMock(side_effect=_fake_save)):
            await _ask_user_impl("Câu hỏi test?", state, config, "tc-001")

    asyncio.run(run())
    passed = called_with.get("interrupt_kind") == "stream_response"
    return GateResult(
        gate="ask_user dispatches STREAM_RESPONSE interrupt (not ASK_HUMAN)",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"interrupt_kind={called_with.get('interrupt_kind')!r}",
    )


# ---------------------------------------------------------------------------
# Contextual Instruction Layers — has_draft=False skips critique/governance/output layers
# ---------------------------------------------------------------------------

def _layers_no_draft_skips_critique_governance() -> GateResult:
    from app import instructions

    instructions.load_instructions()
    full = instructions.get_instruction("vision_objectives", "analysis", None, context=None)
    no_draft = instructions.get_instruction("vision_objectives", "analysis", None, context={"has_draft": False})

    if full is None or no_draft is None:
        return GateResult(
            gate="has_draft=False skips layers 08/09/10",
            passed=False, score=0.0, threshold=1.0, critical=True,
            reason="get_instruction returned None (instructions not loaded?)",
        )
    passed = len(no_draft) < len(full)
    diff = len(full) - len(no_draft)
    return GateResult(
        gate="has_draft=False skips layers 08/09/10 (shorter instruction)",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"no_draft={len(no_draft)} chars vs full={len(full)} chars (diff={diff})" if passed
               else "no_draft instruction is not shorter than full — layers not filtered",
    )


def _layers_cache_key_has_draft_distinct() -> GateResult:
    from app import instructions

    instructions.load_instructions()
    full = instructions.get_instruction("vision_objectives", "analysis", None, context=None)
    with_draft = instructions.get_instruction("vision_objectives", "analysis", None, context={"has_draft": True})
    no_draft = instructions.get_instruction("vision_objectives", "analysis", None, context={"has_draft": False})

    if full is None or with_draft is None or no_draft is None:
        return GateResult(
            gate="cache returns distinct entries for each has_draft value",
            passed=False, score=0.0, threshold=1.0, critical=False,
            reason="instruction None",
        )
    cache = instructions._assembled_cache
    passed = (
        ("business_analyst", None) in cache
        and ("business_analyst", True) in cache
        and ("business_analyst", False) in cache
        and no_draft != full
    )
    return GateResult(
        gate="(role, has_draft) cache keys are distinct",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=False,
        reason="three distinct cache entries found" if passed else f"cache keys: {list(cache.keys())[:6]}",
    )


# ---------------------------------------------------------------------------
# Composite Dispatch — tools array schema, gate logic, multi-tool emission per turn
# ---------------------------------------------------------------------------

def _dispatch_schema_uses_tools_array() -> GateResult:
    # Native tool calling: analyze_node binds the available tools as a provider-agnostic schema list,
    # each {name, description, parameters} — the model returns native tool_calls, no JSON shim.
    from app.graphs.agent_tools import get_available_tools
    from app.graphs.nodes import _build_tool_schemas
    from tests.integration.test_graph_nodes import _state

    schemas = _build_tool_schemas(get_available_tools(_state()))
    passed = bool(schemas) and all(
        {"name", "parameters"} <= set(s) and isinstance(s["parameters"], dict) for s in schemas
    )
    return GateResult(
        gate="analyze_node binds native tool schemas (not legacy JSON tool-selection)",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"bound tools: {[s['name'] for s in schemas]}" if passed
               else f"malformed schemas: {schemas}",
    )


def _dispatch_gate_passes_two_tools() -> GateResult:
    from app.graphs.nodes import _gate_selected_tools
    from tests.integration.test_graph_nodes import _state

    state = _state()
    state["user_confirmed"] = True
    requested = [
        {"name": "critique_note", "args": {"content": "note A"}},
        {"name": "explore_note", "args": {"content": "note B"}},
    ]
    result = _gate_selected_tools(state, requested)
    passed = [r["name"] for r in result] == ["critique_note", "explore_note"]
    return GateResult(
        gate="gate passes two non-interrupt tools through unchanged",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"dispatched: {[r['name'] for r in result]}",
    )


def _dispatch_interrupt_drops_companion() -> GateResult:
    from app.graphs.nodes import _gate_selected_tools
    from tests.integration.test_graph_nodes import _state

    state = _state()
    state["user_confirmed"] = True
    requested = [
        {"name": "ask_user", "args": {"message": "Câu hỏi?"}},
        {"name": "explore_note", "args": {"content": "note"}},
    ]
    result = _gate_selected_tools(state, requested)
    passed = len(result) == 1 and result[0]["name"] == "ask_user"
    return GateResult(
        gate="interrupt-bearing ask_user drops companion explore_note",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"remaining tools: {[r['name'] for r in result]}",
    )


def _dispatch_gate_coerces_unavailable() -> GateResult:
    from app.graphs.nodes import _gate_selected_tools
    from tests.integration.test_graph_nodes import _state

    state = _state()
    # user_confirmed=None → write_draft not available
    state["user_confirmed"] = None
    requested = [{"name": "write_draft", "args": {"title": "t", "body": "b"}}]
    result = _gate_selected_tools(state, requested)
    passed = len(result) == 1 and result[0]["name"] == "ask_user"
    return GateResult(
        gate="gate coerces unavailable write_draft (intent gate) → ask_user",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"coerced to: {result[0]['name'] if result else 'none'}",
    )


def _dispatch_analyze_node_emits_two_calls() -> GateResult:
    """analyze_node with two non-interrupt tools emits AIMessage(tool_calls=[tc0, tc1])."""
    from contextlib import asynccontextmanager

    async def run() -> tuple[bool, str]:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        from app.models.base import Base
        from app.models.agent import AgentSession, AgentSessionStatus
        from app.graphs.nodes import analyze_node
        from tests.integration.test_graph_nodes import _state

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        raw_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with raw_factory() as db:
            sess = AgentSession(
                id=uuid.uuid4(),
                project_id=uuid.uuid4(),
                artifact_type="vision_objectives",
                workflow_area="analysis",
                status=AgentSessionStatus.ACTIVE,
            )
            db.add(sess)
            await db.commit()
            session_id = sess.id
            project_id = sess.project_id

        @asynccontextmanager
        async def session_factory():
            async with raw_factory() as db:
                yield db

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=({
            "tools": [
                {"name": "critique_note", "args": {"content": "Note A"}},
                {"name": "explore_note", "args": {"content": "Note B"}},
            ],
            "active_mode": "critique",
        }, None))

        state = _state()
        state["user_confirmed"] = True
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": str(session_id),
                "project_id": str(project_id),
                "llm_client": mock_llm,
                "session_factory": session_factory,
            }
        }
        out = await analyze_node(state, config)
        msgs = out.get("messages", [])
        if not msgs:
            return False, "no messages returned"
        last = msgs[-1]
        if not isinstance(last, AIMessage):
            return False, f"last message is {type(last).__name__}, not AIMessage"
        tcs = last.tool_calls
        if len(tcs) != 2:
            return False, f"expected 2 tool_calls, got {len(tcs)}: {[t['name'] for t in tcs]}"
        names = [t["name"] for t in tcs]
        if names != ["critique_note", "explore_note"]:
            return False, f"unexpected tool order: {names}"
        if tcs[0]["id"] == tcs[1]["id"]:
            return False, "tool_call IDs are not unique"
        return True, f"dispatched {names} with distinct IDs"

    passed, reason = asyncio.run(run())
    return GateResult(
        gate="analyze_node emits AIMessage with 2 distinct tool_calls",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Aggregation & markdown
# ---------------------------------------------------------------------------

def run_fluency_eval() -> dict[str, Any]:
    gates = [
        # key facts
        _key_facts_note_parser_parses_tag(),
        _key_facts_optional_fields_default_empty(),
        _key_facts_injected_into_prompt(),
        _key_facts_empty_no_section_in_prompt(),
        # interrupt semantics
        _interrupt_stream_response_in_enum(),
        _interrupt_ask_user_uses_stream_response(),
        # contextual layers
        _layers_no_draft_skips_critique_governance(),
        _layers_cache_key_has_draft_distinct(),
        # composite dispatch
        _dispatch_schema_uses_tools_array(),
        _dispatch_gate_passes_two_tools(),
        _dispatch_interrupt_drops_companion(),
        _dispatch_gate_coerces_unavailable(),
        _dispatch_analyze_node_emits_two_calls(),
    ]
    rows = [asdict(g) for g in gates]
    overall = all(g.passed for g in gates)
    return {"passed": overall, "gates": rows}


def _markdown_report(report: dict[str, Any]) -> str:
    icon = "✅" if report["passed"] else "❌"
    lines = [
        "# Eval: Harness Conversation Fluency",
        "",
        f"**Status:** {icon} {'PASSED' if report['passed'] else 'FAILED'}",
        "",
        "## Scenarios",
        "",
        "| # | Scenario | Score | Passed | Reason |",
        "| --- | --- | ---: | :---: | --- |",
    ]
    for i, row in enumerate(report["gates"], 1):
        icon = "✅" if row["passed"] else "❌"
        lines.append(
            f"| {i} | {row['gate']} | {row['score']:.2f} | {icon} | {row['reason']} |"
        )
    total = len(report["gates"])
    passed_count = sum(1 for r in report["gates"] if r["passed"])
    lines += [
        "",
        f"**{passed_count}/{total} scenarios passed.**",
        "",
        "## Plan coverage",
        "",
        "| Feature | Scenarios |",
        "| --- | --- |",
        "| Key Facts State | note_parser tag parsing, optional fields, prompt injection, empty guard |",
        "| Interrupt Semantics | STREAM_RESPONSE enum, ask_user interrupt kind |",
        "| Contextual Layers | has_draft=False skips 08/09/10, distinct cache keys |",
        "| Composite Dispatch | tools array schema, gate pass-through, interrupt drop, coerce, analyze_node 2-tool emit |",
    ]
    return "\n".join(lines) + "\n"


def main(output_path: Path | None = None) -> int:
    report = run_fluency_eval()
    md = _markdown_report(report)
    if output_path:
        output_path.write_text(md, encoding="utf-8")
        print(f"Report written to {output_path}")
    else:
        print(md)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(out))
