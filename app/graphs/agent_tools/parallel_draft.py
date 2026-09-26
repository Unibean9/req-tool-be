"""draft_in_parallel — write a large artifact as several small LLM calls running at the same time.

An agent writing a big table itself does it in sequential batches (read, then 5-8 write calls, each
re-sending the whole history), so a turn adds up to 80-170s and each batch is slower than the last.
Here the code splits the work, runs the parts concurrently and assembles the result, so the whole
table takes about as long as its slowest part:

- Items plans (Business Capabilities, Business Rules, Functional Requirements): the upstream items
  (Scope capabilities / Business Capabilities, "Won't" exempt) are split into groups; each group's
  call writes the rows for its items only. Every item is assigned to exactly one group, so none can
  be forgotten; the code numbers the rows (FR-01, ...) so parts never clash.
- Outline plan (Non-Functional Requirements): one short call lists the NFRs (so there are no
  duplicates across parts and every constraint is mapped), then the rows are written in parallel.
- Sections plan (Constraints, Assumptions, and Risks): independent sections run in parallel;
  a section that builds on another (Validation Plan on Assumptions) runs right after it.

Each part receives only the context it needs (its own items' rows, and the upstream rows that cite
them) plus the conversation brief; anything not supported by that context must be marked as needing
confirmation, never invented. Completed parts are cached in state, so a retry re-runs only the parts
that failed. The result lands in draft_sections exactly like write_draft_section; write_draft then
assembles, gates (coverage included) and proposes it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.config import settings
from app.documents.registry import output_contract
from app.graphs.agent_tools._shared import RecoverableToolError, _recoverable_tool_update
from app.graphs.analysis.context_excerpts import SectionPick, excerpt_body
from app.graphs.analysis.context_loader import load_accepted_artifacts
from app.graphs.analysis.coverage import (
    COVERAGE_RULES,
    CoverageItem,
    cited_ids,
    coverage_items,
    item_rows,
    item_summaries,
    missing_coverage,
    rows_citing,
)
from app.graphs.state import WorkflowState

logger = logging.getLogger(__name__)


def _pick(heading: str, *columns: str) -> SectionPick:
    return SectionPick(heading, tuple(columns))


@dataclass(frozen=True)
class Column:
    key: str
    en: str
    vi: str
    guide: str


@dataclass(frozen=True)
class ItemsPlan:
    """One table under `heading`, rows written per group of upstream items."""

    heading: str
    id_prefix: str
    source_label: tuple[str, str]  # (en, vi) header of the column holding the upstream item IDs
    columns: tuple[Column, ...]
    guidance: str
    context: dict[str, tuple[SectionPick, ...]]
    # Columns that identify a row's content: a part repeating a row verbatim is written once.
    dedupe_on: tuple[str, ...] = ()
    # False for a table keyed by the items it answers (a validation plan row per assumption A-01):
    # no new ID column, the source column comes first.
    own_ids: bool = True
    # After assembly, one short call finds rows that two parts both wrote (a shared behaviour such as
    # history or moderation, owned by two overlapping items) and the code merges them.
    review_duplicates: bool = False
    # One broad capability measured 3.5-5.3k output tokens of Vietnamese rows (Bedrock Sonnet). A
    # truncated answer is unparseable and costs a whole second call, so the cap errs high -- it only
    # bounds the length, the model stops when the rows are done.
    tokens_per_item: int = 7000


@dataclass(frozen=True)
class OutlinePlan:
    """One table under `heading`: a single outline call, then the rows in parallel."""

    heading: str
    id_prefix: str
    source_label: tuple[str, str]
    columns: tuple[Column, ...]
    outline_guidance: str
    guidance: str
    context: dict[str, tuple[SectionPick, ...]]
    dedupe_on: tuple[str, ...] = ()
    tokens_per_item: int = 700


@dataclass(frozen=True)
class SectionJob:
    heading: str
    guidance: str


@dataclass(frozen=True)
class SectionsPlan:
    """Several `##` sections; each chain runs in parallel with the others, its steps in order."""

    # A chain step is a free-form section, or an ItemsPlan answering the previous section row by row.
    chains: tuple[tuple[SectionJob | ItemsPlan, ...], ...]
    context: dict[tuple[str, ...], dict[str, tuple[SectionPick, ...]]] = field(default_factory=dict)
    tokens_per_section: int = 5000


_PRIORITY = Column(
    "priority",
    "Priority",
    "Ưu tiên",
    "MoSCoW (Must/Should/Could), inherited from the source item unless a rule says otherwise",
)

PLANS: dict[str, ItemsPlan | OutlinePlan | SectionsPlan] = {
    "use_case": ItemsPlan(
        heading="## Business Capabilities",
        id_prefix="BC",
        source_label=("Scope capability", "Năng lực (Scope)"),
        columns=(
            Column("capability", "Capability", "Năng lực", "short name of the business capability"),
            Column("goal", "Goal", "Mục tiêu", "what the business achieves with it"),
            Column("user_segment", "User Segment", "Nhóm người dùng", "who uses / benefits from it"),
            Column("business_value", "Business Value", "Giá trị nghiệp vụ", "why it matters, linked to objectives"),
            Column("scope", "Scope", "Phạm vi", "what is included and what is explicitly excluded"),
            _PRIORITY,
        ),
        guidance=(
            "Write one row per scope capability (two only if it clearly holds two distinct business "
            "capabilities). Take goal, user segment and value from the objectives and stakeholders shown."
        ),
        dedupe_on=("capability",),
        context={
            "vision_objectives": (_pick("Objectives", "objective", "value"),),
            "stakeholder_register": (_pick("Stakeholders", "role", "needs"),),
            "scope_capabilities": (_pick("Out of Scope"),),
        },
    ),
    "business_rules": ItemsPlan(
        heading="## Business Rules",
        id_prefix="BR",
        source_label=("Capability", "Năng lực"),
        columns=(
            Column("condition", "Condition", "Điều kiện", "when the rule applies (the state that must hold)"),
            Column("trigger", "Trigger", "Sự kiện kích hoạt", "the event or action that fires it"),
            Column("outcome", "Outcome", "Kết quả", "what must happen / is enforced"),
            Column("scope", "Scope", "Phạm vi", "who or what it applies to"),
            Column("exception", "Exception", "Ngoại lệ", "cases where it does not apply, or 'None'"),
        ),
        guidance=(
            "Write the business rules each of your capabilities needs: eligibility, limits, validation, "
            "state changes, permissions, pricing, compliance. Usually 2-5 per capability; one rule per row."
        ),
        dedupe_on=("condition", "trigger", "outcome"),
        review_duplicates=True,
        context={
            "stakeholder_register": (_pick("Stakeholders", "role", "responsibility", "decision"),),
            "scope_capabilities": (_pick("Out of Scope"),),
        },
    ),
    "functional_requirement": ItemsPlan(
        heading="## Functional Requirements",
        id_prefix="FR",
        source_label=("Capability", "Năng lực"),
        columns=(
            Column("requirement", "Requirement", "Yêu cầu", "one testable statement of what the system does"),
            Column("behavior", "Behavior", "Hành vi", "main flow and the business rules it applies (cite BR IDs)"),
            Column("inputs_outputs", "Inputs/Outputs", "Đầu vào / Đầu ra", "data in and data/result out"),
            Column("acceptance_signal", "Acceptance Signal", "Tiêu chí chấp nhận", "observable, verifiable result"),
            _PRIORITY,
            Column(
                "dependencies", "Dependencies", "Phụ thuộc", "other capability / FR / rule IDs it relies on, or 'None'"
            ),
        ),
        guidance=(
            "Write the functional requirements that fully realise each of your capabilities: usually "
            "3-8 per capability depending on its scope. Apply every business rule shown that concerns it."
        ),
        dedupe_on=("requirement",),
        review_duplicates=True,
        context={
            "business_rules": (_pick("Business Rules"),),
        },
    ),
    "non_functional_requirement": OutlinePlan(
        heading="## Non-Functional Requirements",
        id_prefix="NFR",
        source_label=("Source", "Nguồn"),
        columns=(
            Column("quality_attribute", "Quality Attribute", "Thuộc tính chất lượng", "e.g. Performance, Security"),
            Column("requirement", "Requirement", "Yêu cầu", "the specific, bounded quality requirement"),
            Column("measurement", "Measurement", "Đo lường", "metric, threshold and how it is verified"),
            Column(
                "scope_tradeoff", "Scope / Tradeoff", "Phạm vi / Đánh đổi", "where it applies and what it trades off"
            ),
        ),
        outline_guidance=(
            "List every non-functional requirement the product needs: one entry per distinct, measurable "
            "requirement. Use product quality attributes only (performance, security, privacy, "
            "reliability/availability, scalability, usability, accessibility, compatibility/portability, "
            "maintainability, compliance); business goals, budgets and schedules are not NFRs. Map EVERY "
            "constraint ID either to the entries it drives, or -- only if it drives none -- to "
            "not_applicable with a one-sentence reason (e.g. a budget or a deadline). Also derive entries "
            "from the success metrics and the capabilities. Keep each title under 15 words: this is an "
            "outline, the full rows are written afterwards."
        ),
        guidance="Write exactly one row per outline entry you are given, keeping its sources.",
        dedupe_on=("requirement",),
        context={
            "constraints_assumptions": (_pick("Constraints"), _pick("Risks", "risk", "impact", "severity")),
            "vision_objectives": (_pick("Success Metrics"),),
            "use_case": (_pick("Business Capabilities", "capability", "priority"),),
            "functional_requirement": (_pick("Functional Requirements", "requirement"),),
        },
    ),
    "constraints_assumptions": SectionsPlan(
        chains=(
            (
                SectionJob(
                    "## Constraints",
                    "A table of hard constraints (legal, technical, business, timeline, resources) with ID "
                    "(C-01, ...), constraint, type, impact, owner/source and how it is confirmed.",
                ),
            ),
            (
                SectionJob(
                    "## Assumptions",
                    "A table of assumptions with ID (A-01, ...), assumption, impact if wrong, owner/source, "
                    "confidence.",
                ),
                ItemsPlan(
                    heading="## Validation Plan",
                    id_prefix="",
                    own_ids=False,
                    source_label=("Assumption", "Giả định"),
                    columns=(
                        Column("assumption", "Assumption", "Nội dung giả định", "the assumption restated in full"),
                        Column("method", "Validation Method", "Cách kiểm chứng", "how it will be validated"),
                        Column("owner", "Owner", "Người phụ trách", "who validates it"),
                        Column("deadline", "Deadline", "Thời hạn", "by when (before which milestone)"),
                        Column("if_false", "If False", "Nếu sai", "the decision or change if it proves false"),
                    ),
                    guidance="Write exactly one validation row per assumption given.",
                    context={},
                    dedupe_on=("assumption",),
                    tokens_per_item=1500,
                ),
            ),
            (
                SectionJob(
                    "## Risks",
                    "A table of risks with ID (R-01, ...), risk, cause, likelihood, impact, severity.",
                ),
                ItemsPlan(
                    heading="## Mitigation Plan",
                    id_prefix="",
                    own_ids=False,
                    source_label=("Risk", "Rủi ro"),
                    columns=(
                        Column("risk", "Risk", "Nội dung rủi ro", "the risk restated in full"),
                        Column("mitigation", "Mitigation", "Biện pháp giảm thiểu", "the preventive actions"),
                        Column("owner", "Owner", "Người phụ trách", "who owns the risk"),
                        Column("early_warning", "Early Warning", "Dấu hiệu cảnh báo", "the trigger to act on"),
                        Column("contingency", "Contingency", "Phương án dự phòng", "what to do if it happens"),
                    ),
                    guidance="Write exactly one mitigation row per risk given.",
                    context={},
                    dedupe_on=("risk",),
                    tokens_per_item=1500,
                ),
            ),
        ),
        context={
            ("## Constraints",): {
                "vision_objectives": (_pick("Objectives", "objective", "target", "timeframe"),),
                "stakeholder_register": (_pick("Stakeholders", "role", "decision"),),
                "scope_capabilities": (
                    _pick("Capabilities", "capability", "priority", "dependency"),
                    _pick("Out of Scope"),
                ),
                "business_rules": (_pick("Business Rules"),),
            },
            ("## Assumptions", "## Validation Plan"): {
                "problem_statement": (_pick("Assumptions"),),
                "vision_objectives": (_pick("Assumptions"), _pick("Objectives", "objective", "target")),
                "stakeholder_register": (_pick("Assumptions"),),
                "scope_capabilities": (_pick("Assumptions"), _pick("Capabilities", "capability", "priority")),
            },
            ("## Risks", "## Mitigation Plan"): {
                "vision_objectives": (_pick("Objectives", "objective", "target", "timeframe"),),
                "stakeholder_register": (_pick("Stakeholders", "role", "decision"),),
                "scope_capabilities": (
                    _pick("Capabilities", "capability", "priority", "dependency"),
                    _pick("Out of Scope"),
                ),
                "business_rules": (_pick("Business Rules"),),
            },
        },
    ),
}

MAX_GROUP_SIZE = 3
OUTLINE_ITEMS_PER_GROUP = 4
# The outline is the one step that cannot run in parallel; a cut-off outline costs a full second call.
OUTLINE_MAX_TOKENS = 8000
ROWS_MAX_TOKENS = 9000
_LOCALE_NAMES = {"vi": "Vietnamese", "en": "English"}
# Waits before re-trying a rate-limited call (seconds, jittered +-25%).
THROTTLE_WAITS = (3.0, 6.0, 12.0, 20.0)


class PartFailed(Exception):
    def __init__(self, label: str, reason: str):
        super().__init__(f"{label}: {reason}")
        self.label = label
        self.reason = reason


def has_parallel_plan(artifact_type: str | None) -> bool:
    return artifact_type in PLANS


def split_groups(items: list[CoverageItem], concurrency: int) -> list[list[CoverageItem]]:
    """Consecutive groups. Output speed per call is fixed, so with the parts running `concurrency` at
    a time the table takes about ceil(groups / concurrency) x (one group's length): the groups are
    as small as one wave allows (one item each when there are few), and never more than
    MAX_GROUP_SIZE items, which keeps each call well inside the per-call limit."""
    if not items:
        return []
    size = min(MAX_GROUP_SIZE, max(1, math.ceil(len(items) / max(1, concurrency))))
    return [items[index : index + size] for index in range(0, len(items), size)]


# --- prompt pieces -----------------------------------------------------------------------------


def _base_rules(locale: str, confirmation_note: str) -> str:
    language = _LOCALE_NAMES.get(locale, "the language of the sources")
    return (
        f"- Write every value in {language}.\n"
        "- Use only the sources and the confirmed facts given here. A detail they do not support must be "
        f"marked inline with {confirmation_note}; never present an invented fact as confirmed.\n"
        "- Write each value out in full so it reads on its own. Do not replace content with a reference "
        "such as 'see C2' or 'as above'; cite source IDs in addition to the content, not instead of it.\n"
        "- No filler: no introductions, no summaries, no repeating the same point twice, no narrated test "
        "scenarios. A value is usually one or two sentences -- complete, but only what the reader needs.\n"
        "- Never leave a value empty; if unknown, write what is missing and mark it as needing confirmation."
    )


USER_MESSAGES_MAX = 12
USER_MESSAGE_MAX_CHARS = 800


def _user_messages(messages: list[Any]) -> list[str]:
    """The human's own messages in this session, newest last. Passed to every part verbatim, so a
    decision the user stated reaches the parts even if the agent's brief leaves it out."""
    texts = []
    for message in messages:
        if isinstance(message, dict):
            is_human = message.get("role") in {"user", "human"} and not message.get("tool_call_id")
            content = message.get("content")
        else:
            is_human = getattr(message, "type", "") == "human"
            content = getattr(message, "content", "")
        if not is_human:
            continue
        if isinstance(content, list):
            content = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        text = str(content or "").strip()
        if text:
            texts.append(text[:USER_MESSAGE_MAX_CHARS])
    return texts[-USER_MESSAGES_MAX:]


def _brief_block(brief: str, key_facts: list[dict[str, Any]], user_messages: list[str] | None = None) -> str:
    facts = "\n".join(f"- {fact.get('statement')}" for fact in key_facts if fact.get("statement"))
    parts = []
    if brief.strip():
        parts.append(f"Decisions and facts agreed in this conversation (they override the sources):\n{brief.strip()}")
    if facts:
        parts.append(f"Confirmed key facts:\n{facts}")
    if user_messages:
        quoted = "\n".join(f"- {text}" for text in user_messages)
        parts.append(f"The user's own messages in this conversation (their answers override the sources):\n{quoted}")
    return "\n\n".join(parts)


def _context_block(bodies: dict[str, str], picks: dict[str, tuple[SectionPick, ...]], related: set[str] | None) -> str:
    blocks = []
    for artifact_type, sections in picks.items():
        body = bodies.get(artifact_type)
        if not body:
            continue
        text = excerpt_body(body, sections).text
        if related:
            text = rows_citing(text, related)
        if text.strip():
            blocks.append(f"### [{artifact_type}]\n{text.strip()}")
    return "\n\n".join(blocks)


def _rows_schema(columns: tuple[Column, ...], *, with_sources: bool) -> dict[str, Any]:
    row_properties: dict[str, Any] = {column.key: {"type": "string"} for column in columns}
    required = [column.key for column in columns]
    if with_sources:
        row_properties = {"source_ids": {"type": "array", "items": {"type": "string"}}, **row_properties}
        required = ["source_ids", *required]
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": row_properties, "required": required},
            },
            "not_applicable": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["id", "reason"],
                },
            },
        },
        "required": ["rows", "not_applicable"],
    }


_OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quality_attribute": {"type": "string"},
                    "title": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["quality_attribute", "title", "source_ids"],
            },
        },
        "not_applicable": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["id", "reason"],
            },
        },
    },
    "required": ["entries", "not_applicable"],
}

_SECTION_SCHEMA = {
    "type": "object",
    "properties": {"content": {"type": "string"}},
    "required": ["content"],
}


def _columns_spec(columns: tuple[Column, ...]) -> str:
    return "\n".join(f"- {column.key}: {column.guide}" for column in columns)


# --- rendering ---------------------------------------------------------------------------------


def _cell(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return re.sub(r"\s*\n\s*", "<br>", text).replace("|", "/") or "—"


def _header(label: tuple[str, str] | Column, locale: str) -> str:
    if isinstance(label, Column):
        return label.vi if locale == "vi" else label.en
    return label[1] if locale == "vi" else label[0]


def _dedupe_key(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str]:
    source = ",".join(sorted(row.get("source_ids") or []))
    content = " ".join(str(row.get(key) or "") for key in keys)
    return source, re.sub(r"\W+", " ", content).strip().casefold()


def _unique_rows(plan: ItemsPlan | OutlinePlan, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop rows repeated verbatim (same sources, same content columns)."""
    keys = plan.dedupe_on or tuple(column.key for column in plan.columns)
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = _dedupe_key(row, keys)
        if key[1] and key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def render_table(
    plan: ItemsPlan | OutlinePlan, rows: list[dict[str, Any]], not_applicable: list[dict[str, str]], locale: str
) -> str:
    unique = _unique_rows(plan, rows)
    own_ids = getattr(plan, "own_ids", True)
    headers = [_header(plan.source_label, locale), *(_header(column, locale) for column in plan.columns)]
    if own_ids:
        headers.insert(0, "ID")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for number, row in enumerate(unique, start=1):
        cells = [", ".join(row.get("source_ids") or []), *(row.get(column.key) for column in plan.columns)]
        if own_ids:
            cells.insert(0, f"{plan.id_prefix}-{number:02d}")
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    body = "\n".join(lines)
    if not_applicable:
        title = "Không áp dụng" if locale == "vi" else "Not applicable"
        reason = "Lý do" if locale == "vi" else "Reason"
        na_lines = [f"### {title}", "", f"| ID | {reason} |", "|---|---|"]
        na_lines += [f"| {_cell(item.get('id'))} | {_cell(item.get('reason'))} |" for item in not_applicable]
        body += "\n\n" + "\n".join(na_lines)
    return body


# --- LLM parts ---------------------------------------------------------------------------------


@dataclass
class Runner:
    client: Any
    semaphore: asyncio.Semaphore
    locale: str = ""
    timings: list[dict[str, Any]] = field(default_factory=list)

    async def call(self, label: str, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int) -> dict:
        """One structured call, retried once on any failure (timeout, provider error, no answer).

        The answer comes back as the arguments of a forced `submit` tool call -- the same native
        tool calling the agent loop uses on every provider -- rather than as JSON text: text JSON
        broke intermittently on unescaped quotes in Vietnamese content ("nhấn "Nộp bài""). A missing
        answer is almost always one cut off at max_tokens, so the retry raises the budget. A
        rate-limited call (several parts at once can exceed a provider's quota) waits and tries
        again without using up an attempt, and gives up its concurrency slot while it waits."""
        tool_schema = {"name": "submit", "description": "Submit your part of the document.", "parameters": schema}
        last_error = ""
        attempt = 1
        throttled = 0
        while attempt <= 2:
            async with self.semaphore:
                started = time.monotonic()
                try:
                    message, usage = await self.client.generate(
                        messages=[{"role": "user", "content": prompt}],
                        system=system,
                        max_tokens=max_tokens,
                        tools=[tool_schema],
                        tool_choice="required",
                    )
                    error = None
                except Exception as exc:  # noqa: BLE001 - any provider failure is retried, then reported
                    error = exc
                elapsed = time.monotonic() - started
            if error is not None:
                last_error = f"{type(error).__name__}: {error}"[:300]
                if _is_rate_limited(error) and throttled < len(THROTTLE_WAITS):
                    wait = THROTTLE_WAITS[throttled] * random.uniform(0.75, 1.25)
                    throttled += 1
                    logger.warning("parallel_draft part=%s rate limited, waiting %.1fs", label, wait)
                    await asyncio.sleep(wait)
                    continue
                logger.warning("parallel_draft part=%s attempt=%d failed: %s", label, attempt, last_error)
                attempt += 1
                continue
            calls = getattr(message, "tool_calls", None) or []
            result = next((call.get("args") for call in calls if call.get("name") == "submit"), None)
            if not isinstance(result, dict) or not result:
                last_error = "no answer (cut off at the token limit?)"
                logger.warning("parallel_draft part=%s attempt=%d: %s", label, attempt, last_error)
                # Bounded so the retry still fits the per-call deadline (~90 tokens/s measured).
                max_tokens = max(max_tokens, min(ROWS_MAX_TOKENS, max_tokens * 2))
                attempt += 1
                continue
            self.timings.append({"part": label, "seconds": round(elapsed, 1), "usage": usage, "attempt": attempt})
            logger.info("parallel_draft part=%s seconds=%.1f usage=%s", label, elapsed, usage)
            return result
        raise PartFailed(label, last_error or "no result")


_RATE_LIMIT_MARKERS = ("throttl", "rate limit", "ratelimit", "too many requests", "429", "resource_exhausted", "quota")


def _is_rate_limited(error: BaseException) -> bool:
    """A short-window limit worth waiting out. A daily quota ("Too many tokens per day") is not:
    waiting seconds cannot help, so it fails at once and is reported."""
    text = f"{type(error).__name__} {error}".casefold()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS) and "per day" not in text


def _clean_ids(raw: Any, allowed: set[str], fallback: list[str]) -> list[str]:
    values = [str(item).strip() for item in (raw or []) if str(item).strip()]
    valid = [item for item in values if item in allowed]
    return valid or fallback


async def _items_group(
    runner: Runner,
    plan: ItemsPlan,
    group: list[CoverageItem],
    *,
    all_items: list[CoverageItem],
    summaries: dict[str, str],
    source_rows: str,
    context: str,
    system: str,
    brief: str,
) -> dict[str, Any]:
    ids = [item.id for item in group]
    label = ",".join(ids)
    others = "\n".join(
        f"- {item.id}: {summaries.get(item.id) or item.label}".rstrip(": ") for item in all_items if item.id not in ids
    )
    prompt = (
        f"Write the rows of the '{plan.heading.removeprefix('## ')}' table for YOUR ITEMS only: {', '.join(ids)}.\n"
        f"{plan.guidance}\n"
        "Every row lists in source_ids the item ID(s) it realises. Every one of your items must get at "
        "least one row, unless it genuinely does not apply -- then put it in not_applicable with the reason.\n\n"
        f"Columns:\n{_columns_spec(plan.columns)}\n\n"
        f"YOUR ITEMS (full source rows):\n{source_rows}\n\n"
        + (
            "Other items (written separately at the same time -- do not write rows for them):\n"
            f"{others}\n"
            "Something several items share (e.g. login, history, content moderation) is written ONCE, under "
            "the item whose scope covers it most directly. If that is one of the other items, do not write "
            "it here: put that item's ID in your row's dependencies instead.\n\n"
            if others
            else ""
        )
        + (f"{brief}\n\n" if brief else "")
        + (f"Related context:\n{context}\n" if context else "")
    )
    schema = _rows_schema(plan.columns, with_sources=True)
    max_tokens = min(ROWS_MAX_TOKENS, plan.tokens_per_item * len(group))
    feedback = ""
    for _round in (1, 2):
        result = await runner.call(label, system=system, prompt=prompt + feedback, schema=schema, max_tokens=max_tokens)
        rows = [row for row in result.get("rows") or [] if isinstance(row, dict)]
        for row in rows:
            # A row citing none of this part's items is still about them (it was asked for them only).
            row["source_ids"] = _clean_ids(row.get("source_ids"), set(ids), ids)
        not_applicable = [
            {"id": str(item.get("id")).strip(), "reason": str(item.get("reason") or "").strip()}
            for item in result.get("not_applicable") or []
            if isinstance(item, dict) and str(item.get("id") or "").strip() in ids
        ]
        covered = {source for row in rows for source in row["source_ids"]} | {item["id"] for item in not_applicable}
        missing = [item_id for item_id in ids if item_id not in covered]
        if not missing:
            return {"rows": rows, "not_applicable": not_applicable}
        feedback = (
            f"\n\nYour previous answer left out {', '.join(missing)}. Answer again with rows for every one "
            "of your items (or a not_applicable entry with the reason)."
        )
    raise PartFailed(label, f"left out {', '.join(missing)}")


# --- plan runners ------------------------------------------------------------------------------


@dataclass
class DraftResult:
    sections: dict[str, str]
    failures: list[PartFailed]
    cache: dict[str, Any]
    timings: list[dict[str, Any]]
    summary: str


def _cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _system(artifact_type: str, locale: str) -> str:
    contract = output_contract(artifact_type)
    return (
        "You are a senior business analyst writing one part of a requirements document. Several parts are "
        "written at the same time by other writers; write only your part, completely, and submit it with "
        "the submit tool.\n" + _base_rules(locale, contract.confirmation_note)
    )


_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "duplicates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"keep": {"type": "string"}, "drop": {"type": "array", "items": {"type": "string"}}},
                "required": ["keep", "drop"],
            },
        }
    },
    "required": ["duplicates"],
}
REVIEW_MAX_TOKENS = 1500
REVIEW_ROW_MAX_CHARS = 300


async def _merge_duplicates(
    runner: Runner, plan: ItemsPlan, rows: list[dict[str, Any]], system: str
) -> list[dict[str, Any]]:
    """Merge rows two parallel parts both wrote. The model only names the duplicates; the code keeps
    one row per set and gives it every dropped row's source IDs, so coverage cannot be lost. If the
    review fails, the rows are returned unchanged -- it is a refinement, never a reason to fail."""
    numbered = {f"{plan.id_prefix}-{number:02d}": row for number, row in enumerate(rows, start=1)}
    keys = plan.dedupe_on or tuple(column.key for column in plan.columns)
    listing = "\n".join(
        f"{row_id} [{', '.join(row['source_ids'])}]: "
        + " / ".join(str(row.get(key) or "") for key in keys)[:REVIEW_ROW_MAX_CHARS]
        for row_id, row in numbered.items()
    )
    prompt = (
        f"These '{plan.heading.removeprefix('## ')}' rows were written in parallel parts, one per item in "
        "brackets, so the same requirement can appear twice under different items. Find rows that state "
        "the same requirement (the same behaviour or rule, even if worded differently) -- not rows that "
        "are merely related, share a condition, or cover different steps of one flow. For each set, keep "
        "the most complete row and list the others to drop. Return an empty list if there are none.\n\n"
        f"{listing}\n"
    )
    try:
        result = await runner.call(
            "duplicate review", system=system, prompt=prompt, schema=_REVIEW_SCHEMA, max_tokens=REVIEW_MAX_TOKENS
        )
    except PartFailed as exc:
        logger.warning("parallel_draft duplicate review skipped: %s", exc)
        return rows
    dropped: set[str] = set()
    for entry in result.get("duplicates") or []:
        if not isinstance(entry, dict):
            continue
        keep = str(entry.get("keep") or "").strip()
        if keep not in numbered or keep in dropped:
            continue
        for row_id in entry.get("drop") or []:
            row_id = str(row_id).strip()
            # A row that already absorbed others passes their sources on in turn, so none is lost.
            if row_id == keep or row_id not in numbered or row_id in dropped:
                continue
            target = numbered[keep]
            target["source_ids"] = list(dict.fromkeys([*target["source_ids"], *numbered[row_id]["source_ids"]]))
            dropped.add(row_id)
    if dropped:
        logger.info("parallel_draft merged duplicate rows: %s", sorted(dropped))
    return [row for row_id, row in numbered.items() if row_id not in dropped]


async def _write_item_rows(
    runner: Runner,
    plan: ItemsPlan,
    items: list[CoverageItem],
    *,
    source_body: str,
    section: str,
    context_for: Any,
    system: str,
    brief: str,
    cache_scope: tuple[str, ...],
    cache: dict[str, Any],
    new_cache: dict[str, Any],
) -> tuple[str, int, int]:
    """Rows for every item, written in parallel groups; returns (table, rows, parts). Raises the first
    PartFailed once every part has finished (the finished ones are in new_cache for the retry)."""
    groups = split_groups(items, settings.parallel_draft_concurrency)
    results: dict[str, dict[str, Any]] = {}
    summaries = item_summaries(source_body, section)

    def part(group: list[CoverageItem]) -> tuple[str, str, str]:
        ids = [item.id for item in group]
        rows_text = item_rows(source_body, section, ids)
        context = context_for(set(ids) | cited_ids(rows_text))
        return rows_text, context, _cache_key(*cache_scope, ids, rows_text, context, brief)

    async def split(group: list[CoverageItem]) -> None:
        singles = await asyncio.gather(*(run([item]) for item in group), return_exceptions=True)
        for outcome in singles:
            if isinstance(outcome, BaseException) and not isinstance(outcome, PartFailed):
                raise outcome
        failed = [outcome for outcome in singles if isinstance(outcome, PartFailed)]
        if failed:
            raise PartFailed(", ".join(failure.label for failure in failed), failed[0].reason)

    async def run(group: list[CoverageItem]) -> None:
        rows_text, context, key = part(group)
        if key in cache:
            results[key] = cache[key]
            return
        if len(group) > 1 and any(part([item])[2] in cache for item in group):
            await split(group)  # an earlier attempt already fell back to single items
            return
        try:
            results[key] = await _items_group(
                runner,
                plan,
                group,
                all_items=items,
                summaries=summaries,
                source_rows=rows_text,
                context=context,
                system=system,
                brief=brief,
            )
        except PartFailed:
            if len(group) == 1:
                raise
            # A part that keeps failing is usually too long for one answer: write its items one by one.
            await split(group)
            return
        new_cache[key] = results[key]

    outcomes = await asyncio.gather(*(run(group) for group in groups), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException) and not isinstance(outcome, PartFailed):
            raise outcome
    failures = [outcome for outcome in outcomes if isinstance(outcome, PartFailed)]
    if failures:
        raise (
            failures[0]
            if len(failures) == 1
            else PartFailed(", ".join(failure.label for failure in failures), failures[0].reason)
        )
    # Rows in the upstream items' order, whatever order the parts finished in.
    order = {item.id: index for index, item in enumerate(items)}
    rows = [row for result in results.values() for row in result["rows"]]
    rows.sort(key=lambda row: min(order.get(source, len(order)) for source in row["source_ids"]))
    not_applicable = sorted(
        (entry for result in results.values() for entry in result["not_applicable"]),
        key=lambda entry: order.get(entry["id"], len(order)),
    )
    rows = _unique_rows(plan, rows)
    if plan.review_duplicates and len(groups) > 1:
        rows = await _merge_duplicates(runner, plan, rows, system)
    table = render_table(plan, rows, not_applicable, runner.locale)
    missing = missing_coverage(table, items)
    if missing:  # defensive: every group was checked, so this means rendering lost a row
        raise PartFailed("assembly", f"lost {', '.join(item.id for item in missing)}")
    return table, len(rows), len(groups)


async def _run_items_plan(
    runner: Runner, artifact_type: str, plan: ItemsPlan, bodies: dict[str, str], brief: str, cache: dict
) -> DraftResult:
    source_type, section = COVERAGE_RULES[artifact_type]
    source_body = bodies.get(source_type) or ""
    items = coverage_items(source_body, section)
    if not items:
        raise PartFailed(source_type, f"no accepted {source_type} with ID'd items in '{section}' to split on")
    new_cache: dict[str, Any] = {}
    try:
        table, row_count, part_count = await _write_item_rows(
            runner,
            plan,
            items,
            source_body=source_body,
            section=section,
            context_for=lambda related: _context_block(bodies, plan.context, related),
            system=_system(artifact_type, runner.locale),
            brief=brief,
            cache_scope=(artifact_type, runner.locale),
            cache=cache,
            new_cache=new_cache,
        )
    except PartFailed as exc:
        return DraftResult({}, [exc], new_cache, runner.timings, "")
    summary = f"{row_count} rows from {part_count} parallel parts covering {', '.join(item.id for item in items)}"
    return DraftResult({plan.heading: table}, [], new_cache, runner.timings, summary)


async def _run_outline_plan(
    runner: Runner, artifact_type: str, plan: OutlinePlan, bodies: dict[str, str], brief: str, cache: dict
) -> DraftResult:
    locale = runner.locale
    source_type, section = COVERAGE_RULES[artifact_type]
    required = coverage_items(bodies.get(source_type) or "", section)
    required_ids = [item.id for item in required]
    system = _system(artifact_type, locale)
    context = _context_block(bodies, plan.context, None)
    new_cache: dict[str, Any] = {}

    outline_key = _cache_key(artifact_type, "outline", context, brief, locale)
    outline = cache.get(outline_key)
    if outline is None:
        prompt = (
            f"{plan.outline_guidance}\n"
            "Constraint IDs that must each be mapped or listed as not applicable: "
            f"{', '.join(required_ids) or '(none)'}.\n"
            "Each entry: quality_attribute, a one-line title, and source_ids (the constraint / metric / "
            "capability IDs it comes from).\n\n" + (f"{brief}\n\n" if brief else "") + f"Sources:\n{context}\n"
        )
        feedback = ""
        for _round in (1, 2):
            outline = await runner.call(
                "outline",
                system=system,
                prompt=prompt + feedback,
                schema=_OUTLINE_SCHEMA,
                max_tokens=OUTLINE_MAX_TOKENS,
            )
            mapped = {
                str(source).strip()
                for entry in outline.get("entries") or []
                for source in entry.get("source_ids") or []
            }
            mapped |= {str(item.get("id")).strip() for item in outline.get("not_applicable") or []}
            missing = [item_id for item_id in required_ids if item_id not in mapped]
            if not missing and outline.get("entries"):
                break
            feedback = f"\n\nYour previous outline did not map: {', '.join(missing) or '(no entries)'}. Map every ID."
        else:
            return DraftResult(
                {},
                [PartFailed("outline", f"did not map {', '.join(missing) or 'any entry'}")],
                new_cache,
                runner.timings,
                "",
            )
        new_cache[outline_key] = outline

    entries = [entry for entry in outline.get("entries") or [] if isinstance(entry, dict)]
    not_applicable = [
        {"id": str(item.get("id")).strip(), "reason": str(item.get("reason") or "").strip()}
        for item in outline.get("not_applicable") or []
        if isinstance(item, dict)
    ]
    # As few waves as possible: all groups in one wave unless that would make a group too long.
    per_group = max(OUTLINE_ITEMS_PER_GROUP, math.ceil(len(entries) / max(1, settings.parallel_draft_concurrency)))
    groups = [entries[index : index + per_group] for index in range(0, len(entries), per_group)]
    schema = _rows_schema(plan.columns, with_sources=True)

    async def run(index: int, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = sorted({str(source) for entry in group for source in entry.get("source_ids") or []})
        group_context = _context_block(bodies, plan.context, set(sources) or None)
        listed = "\n".join(
            f"{number}. [{entry.get('quality_attribute')}] {entry.get('title')} "
            f"(sources: {', '.join(entry.get('source_ids') or [])})"
            for number, entry in enumerate(group, start=1)
        )
        key = _cache_key(artifact_type, "rows", listed, group_context, brief, locale)
        if key in cache:
            return cache[key]
        prompt = (
            f"{plan.guidance} Return the rows in the same order as the entries.\n\n"
            f"Columns:\n{_columns_spec(plan.columns)}\n\nYOUR ENTRIES:\n{listed}\n\n"
            + (f"{brief}\n\n" if brief else "")
            + f"Sources:\n{group_context}\n"
        )
        result = await runner.call(
            f"rows-{index + 1}",
            system=system,
            prompt=prompt,
            schema=schema,
            max_tokens=min(ROWS_MAX_TOKENS, plan.tokens_per_item * len(group) + 500),
        )
        rows = [row for row in result.get("rows") or [] if isinstance(row, dict)]
        if len(rows) < len(group):
            raise PartFailed(f"rows-{index + 1}", f"wrote {len(rows)} of {len(group)} entries")
        for row, entry in zip(rows, group, strict=False):
            row["source_ids"] = [str(source) for source in entry.get("source_ids") or []] or list(
                row.get("source_ids") or []
            )
        new_cache[key] = rows
        return rows

    outcomes = await asyncio.gather(*(run(index, group) for index, group in enumerate(groups)), return_exceptions=True)
    failures = [outcome for outcome in outcomes if isinstance(outcome, PartFailed)]
    for outcome in outcomes:
        if isinstance(outcome, BaseException) and not isinstance(outcome, PartFailed):
            raise outcome
    if failures:
        return DraftResult({}, failures, new_cache, runner.timings, "")
    rows = [row for outcome in outcomes for row in outcome]
    content = render_table(plan, rows, not_applicable, locale)
    summary = f"{len(rows)} requirements from 1 outline + {len(groups)} parallel parts"
    return DraftResult({plan.heading: content}, [], new_cache, runner.timings, summary)


async def _run_sections_plan(
    runner: Runner, artifact_type: str, plan: SectionsPlan, bodies: dict[str, str], brief: str, cache: dict
) -> DraftResult:
    system = _system(artifact_type, runner.locale)
    sections: dict[str, str] = {}
    new_cache: dict[str, Any] = {}

    async def write_section(job: SectionJob, context: str, earlier: str) -> str:
        key = _cache_key(artifact_type, job.heading, context, earlier, brief, runner.locale)
        if key in cache:
            return cache[key]
        prompt = (
            f"Write the content of the section '{job.heading}' (Markdown, without the heading line).\n"
            f"{job.guidance}\n\n"
            + (f"This builds on the section already written:\n{earlier}\n\n" if earlier else "")
            + (f"{brief}\n\n" if brief else "")
            + f"Sources:\n{context}\n"
        )
        result = await runner.call(
            job.heading, system=system, prompt=prompt, schema=_SECTION_SCHEMA, max_tokens=plan.tokens_per_section
        )
        content = str(result.get("content") or "").strip()
        if not content:
            raise PartFailed(job.heading, "empty section")
        new_cache[key] = content
        return content

    async def run_chain(chain: tuple[SectionJob | ItemsPlan, ...]) -> None:
        picks = plan.context.get(tuple(job.heading for job in chain), {})
        context = _context_block(bodies, picks, None)
        previous: tuple[str, str] | None = None
        for job in chain:
            items = []
            if isinstance(job, ItemsPlan) and previous is not None:
                source_body = f"{previous[0]}\n{previous[1]}"
                items = coverage_items(source_body, previous[0].removeprefix("## "))
            if items:
                # One row per item of the section it builds on (A-01, R-01, ...), written in parallel.
                content, _rows, _parts = await _write_item_rows(
                    runner,
                    job,
                    items,
                    source_body=source_body,
                    section=previous[0].removeprefix("## "),
                    context_for=lambda _related: context,
                    system=system,
                    brief=brief,
                    cache_scope=(artifact_type, job.heading, runner.locale),
                    cache=cache,
                    new_cache=new_cache,
                )
            else:
                # First section of a chain, or the section it builds on has no ID column to split on.
                earlier = f"{previous[0]}\n{previous[1]}" if previous else ""
                section_job = job if isinstance(job, SectionJob) else SectionJob(job.heading, job.guidance)
                content = await write_section(section_job, context, earlier)
            sections[job.heading] = content
            previous = (job.heading, content)

    outcomes = await asyncio.gather(*(run_chain(chain) for chain in plan.chains), return_exceptions=True)
    failures = [outcome for outcome in outcomes if isinstance(outcome, PartFailed)]
    for outcome in outcomes:
        if isinstance(outcome, BaseException) and not isinstance(outcome, PartFailed):
            raise outcome
    ordered = {job.heading: sections[job.heading] for chain in plan.chains for job in chain if job.heading in sections}
    summary = f"{len(ordered)} sections from {len(plan.chains)} parallel chains"
    return DraftResult(ordered if not failures else {}, failures, new_cache, runner.timings, summary)


async def generate_parallel_draft(
    artifact_type: str,
    bodies: dict[str, str],
    *,
    client: Any,
    brief: str,
    locale: str,
    cache: dict[str, Any] | None = None,
) -> DraftResult:
    plan = PLANS[artifact_type]
    runner = Runner(client=client, semaphore=asyncio.Semaphore(settings.parallel_draft_concurrency), locale=locale)
    cache = cache or {}
    if isinstance(plan, ItemsPlan):
        return await _run_items_plan(runner, artifact_type, plan, bodies, brief, cache)
    if isinstance(plan, OutlinePlan):
        return await _run_outline_plan(runner, artifact_type, plan, bodies, brief, cache)
    return await _run_sections_plan(runner, artifact_type, plan, bodies, brief, cache)


def plan_source_types(artifact_type: str) -> list[str]:
    plan = PLANS[artifact_type]
    types: list[str] = []
    if artifact_type in COVERAGE_RULES:
        types.append(COVERAGE_RULES[artifact_type][0])
    contexts = plan.context.values() if isinstance(plan, SectionsPlan) else [plan.context]
    for context in contexts:
        types.extend(context)
    return list(dict.fromkeys(item for item in types if item != artifact_type))


# --- tool ----------------------------------------------------------------------------------------


async def _draft_in_parallel_impl(
    brief: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str
) -> Command:
    artifact_type = state.get("artifact_type") or ""
    if not has_parallel_plan(artifact_type):
        return _recoverable_tool_update(
            RecoverableToolError(
                code="parallel_draft_unavailable",
                message=f"draft_in_parallel does not support {artifact_type}.",
                recovery="Write the draft with write_draft_section / write_draft instead.",
            ),
            tool_call_id,
        )
    cfg = config["configurable"]
    client = cfg.get("strong_llm_client") or cfg.get("llm_client")
    project_id = uuid.UUID(str(cfg["project_id"])) if cfg.get("project_id") else None
    if client is None or project_id is None:
        raise ValueError("draft_in_parallel needs an LLM client and a project")

    async with cfg["session_factory"]() as db:
        artifacts = await load_accepted_artifacts(
            db, project_id=project_id, artifact_types=plan_source_types(artifact_type)
        )
    bodies = {artifact.type.value: artifact.current_version.body or "" for artifact in artifacts}

    brief_text = _brief_block(brief or "", state.get("key_facts") or [], _user_messages(state.get("messages") or []))
    started = time.monotonic()
    try:
        result = await generate_parallel_draft(
            artifact_type,
            bodies,
            client=client,
            brief=brief_text,
            locale=state.get("locale") or "",
            cache=state.get("parallel_draft_cache") or {},
        )
    except PartFailed as exc:
        return _recoverable_tool_update(
            RecoverableToolError(
                code="parallel_draft_unavailable",
                message=str(exc),
                recovery=("Draft it yourself with write_draft_section / write_draft instead."),
            ),
            tool_call_id,
        )
    elapsed = time.monotonic() - started
    logger.info("parallel_draft artifact=%s seconds=%.1f parts=%s", artifact_type, elapsed, result.timings)
    cache_update = {"parallel_draft_cache": result.cache} if result.cache else {}
    if result.failures:
        failed = "; ".join(f"{failure.label} ({failure.reason})" for failure in result.failures)
        return _recoverable_tool_update(
            RecoverableToolError(
                code="parallel_draft_incomplete",
                message=f"Some parts could not be written: {failed}. Nothing was saved as the draft.",
                recovery=(
                    "Call draft_in_parallel again: finished parts are reused and only these are re-run. "
                    "If it fails again, tell the user which parts failed with ask_user."
                ),
            ),
            tool_call_id,
            extra_update=cache_update,
        )
    draft_sections = {
        heading: {"content": f"{heading}\n{content.strip()}", "done": True}
        for heading, content in result.sections.items()
    }
    message = (
        f"Draft written in {elapsed:.0f}s: {result.summary}. It is saved as the draft sections -- call "
        "write_draft now to assemble and propose it (body can be a short placeholder). Do not rewrite it."
    )
    return Command(
        update={
            "draft_sections": draft_sections,
            **cache_update,
            "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
        }
    )


@tool
async def draft_in_parallel(
    brief: Annotated[
        str,
        "Everything agreed in this conversation that the draft must follow: the user's answers, "
        "confirmed numbers/dates/scope decisions, and anything they asked to include or exclude. "
        "Plain bullet points; write '(none)' if nothing was agreed beyond the accepted artifacts.",
    ],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write this artifact's whole draft in one step: the code splits it into parts written at the
    same time (much faster than writing it yourself batch by batch) and assembles them, covering
    every upstream item. Use it for the first draft of this artifact once the intent is confirmed,
    instead of write_draft_section. Call it alone, then call write_draft to propose the result."""
    return await _draft_in_parallel_impl(brief, state, config, tool_call_id)
