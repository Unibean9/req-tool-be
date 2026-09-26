"""draft_in_parallel (app/graphs/agent_tools/parallel_draft.py), with a scripted LLM."""

import asyncio
import re
import time
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from app.graphs.agent_tools import _draft_in_parallel_impl, _write_draft_impl
from app.graphs.agent_tools.parallel_draft import (
    PLANS,
    CoverageItem,
    generate_parallel_draft,
    split_groups,
)
from app.graphs.analysis.coverage import coverage_items, missing_coverage
from app.models.artifact import ArtifactType
from tests.factories import (
    _accept_predecessor,
    _config,
    _focused_items,
    _make_agent_run,
    _make_agent_session,
    _project,
    _session_factory,
    _state,
)

CAPABILITIES = """## Business Capabilities

| ID | Capability | Goal | Priority |
|---|---|---|---|
| BC1 | Tạo task (C1) | Tạo nhanh | Must |
| BC2 | Auto-assign (C2) | Gợi ý người | Must |
| BC3 | Báo cáo (C3) | Theo dõi | Should |
| BC4 | Chat (C4) | Trao đổi | Won't |
| BC5 | Nhắc hạn (C5) | Đúng hạn | Could |
"""

RULES = """## Business Rules

| Rule ID | Condition | Outcome |
|---|---|---|
| BR-R1 | Task thiếu tiêu đề (C1) | Từ chối lưu |
| BR-R2 | Thành viên quá tải (C2) | Không gợi ý |
| BR-R3 | Hết hạn (C5) | Gửi nhắc |
"""


def _submit(result: dict) -> AIMessage:
    """The shape a provider returns for the forced `submit` tool call."""
    return AIMessage(content="", tool_calls=[{"id": "call-1", "name": "submit", "args": result}])


def _ids_asked(prompt: str) -> list[str]:
    match = re.search(r"YOUR ITEMS only: ([^.\n]+)\.", prompt)
    return [item.strip() for item in match.group(1).split(",")] if match else []


class ScriptedLLM:
    """Answers each part from its prompt. `drop` leaves an item out (per attempt), `delay` sleeps."""

    def __init__(
        self, *, drop: dict[str, int] | None = None, delay: float = 0.0, fail_labels=(), slow=(), duplicates=()
    ):
        self.duplicates = list(duplicates)  # the duplicate review's answer
        self.reviews: list[str] = []
        self.drop = dict(drop or {})
        self.delay = delay
        self.slow = set(slow)
        self.fail_labels = set(fail_labels)
        self.prompts: list[str] = []

    async def generate(self, *, messages, system, max_tokens, tools, tool_choice):
        assert tool_choice == "required" and tools[0]["name"] == "submit"
        prompt = messages[0]["content"]
        if "duplicates" in tools[0]["parameters"]["properties"]:
            self.reviews.append(prompt)
            if self.duplicates == "fail":
                raise TimeoutError("review hung")
            return _submit({"duplicates": self.duplicates}), None
        self.prompts.append(prompt)
        ids = _ids_asked(prompt)
        await asyncio.sleep(self.delay + (0.05 if self.slow & set(ids) else 0))
        if any(item in self.fail_labels for item in ids):
            raise TimeoutError("provider hung")
        rows = []
        for item in ids:
            if self.drop.get(item, 0) > 0:
                self.drop[item] -= 1
                continue
            rows.append(
                {
                    "source_ids": [item],
                    "requirement": f"Hệ thống hỗ trợ {item}",
                    "behavior": "Luồng chính",
                    "inputs_outputs": "Vào / ra",
                    "acceptance_signal": "Kiểm thử được",
                    "priority": "Must",
                    "dependencies": "None",
                }
            )
        return _submit({"rows": rows, "not_applicable": []}), {"input": 10, "output": 10, "total": 20}


@pytest.fixture(autouse=True)
def _two_parts_at_a_time(monkeypatch):
    # Groups of two for these 4-item fixtures, so grouping and parallelism are both exercised.
    from app.config import settings

    monkeypatch.setattr(settings, "parallel_draft_concurrency", 2)


def test_groups_fill_one_wave_but_stay_small():
    items = [CoverageItem(f"C{index}", "") for index in range(1, 14)]

    assert [len(group) for group in split_groups(items, 8)] == [2] * 6 + [1]
    assert [len(group) for group in split_groups(items[:8], 8)] == [1] * 8
    assert [len(group) for group in split_groups(items * 3, 8)] == [3] * 13


@pytest.mark.asyncio
async def test_every_item_gets_rows_numbered_in_item_order_whatever_finishes_first():
    llm = ScriptedLLM(slow={"BC1"})  # the first part finishes last

    result = await generate_parallel_draft(
        "functional_requirement",
        {"use_case": CAPABILITIES, "business_rules": RULES},
        client=llm,
        brief="",
        locale="vi",
    )

    assert not result.failures
    table = result.sections["## Functional Requirements"]
    items = coverage_items(CAPABILITIES, "Business Capabilities")
    assert [item.id for item in items] == ["BC1", "BC2", "BC3", "BC5"]  # BC4 is Won't
    assert missing_coverage(table, items) == []
    rows = [line for line in table.splitlines() if line.startswith("| FR-")]
    assert [row.split("|")[1].strip() for row in rows] == ["FR-01", "FR-02", "FR-03", "FR-04"]
    assert [row.split("|")[2].strip() for row in rows] == ["BC1", "BC2", "BC3", "BC5"]
    assert "| ID | Năng lực | Yêu cầu |" in table  # Vietnamese headers for a vi session


@pytest.mark.asyncio
async def test_parts_run_at_the_same_time():
    llm = ScriptedLLM(delay=0.2)
    started = time.monotonic()

    await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=llm, brief="", locale="en"
    )

    assert len(llm.prompts) == 2  # 4 items -> 2 groups of 2
    assert time.monotonic() - started < 0.35


@pytest.mark.asyncio
async def test_each_part_gets_its_own_rows_and_only_the_rules_that_cite_them():
    llm = ScriptedLLM()

    await generate_parallel_draft(
        "functional_requirement",
        {"use_case": CAPABILITIES, "business_rules": RULES},
        client=llm,
        brief="- Nhóm 5 người",
        locale="vi",
    )

    first = next(prompt for prompt in llm.prompts if _ids_asked(prompt) == ["BC1", "BC2"])
    # Its own capability rows in full, reached through the C ids those rows cite...
    assert "| BC1 | Tạo task (C1) | Tạo nhanh | Must |" in first
    assert "BR-R1" in first and "BR-R2" in first
    # ...but not another part's rows or rules.
    assert "| BC3 |" not in first and "BR-R3" not in first
    assert "Nhóm 5 người" in first


@pytest.mark.asyncio
async def test_a_part_that_leaves_an_item_out_is_asked_again():
    llm = ScriptedLLM(drop={"BC2": 1})

    result = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=llm, brief="", locale="en"
    )

    assert not result.failures
    assert any("left out BC2" in prompt for prompt in llm.prompts)
    assert "BC2" in result.sections["## Functional Requirements"]


@pytest.mark.asyncio
async def test_a_failed_part_saves_nothing_and_a_retry_reruns_only_that_part():
    failing = ScriptedLLM(fail_labels={"BC3"})

    first = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=failing, brief="", locale="en"
    )

    assert first.sections == {}
    # The failing BC3,BC5 part fell back to one item each: BC5 then succeeded on its own.
    assert [failure.label for failure in first.failures] == ["BC3"]
    assert len(first.cache) == 2  # the BC1,BC2 part and BC5 finished and are kept

    healthy = ScriptedLLM()
    second = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=healthy, brief="", locale="en", cache=first.cache
    )

    assert not second.failures
    assert [_ids_asked(prompt) for prompt in healthy.prompts] == [["BC3"]]
    assert (
        missing_coverage(
            second.sections["## Functional Requirements"], coverage_items(CAPABILITIES, "Business Capabilities")
        )
        == []
    )


CONSTRAINTS = """## Constraints

| # | Ràng buộc | Tác động |
|---|---|---|
| CON-1 | Dữ liệu cá nhân theo NĐ 13 | Cao |
| CON-2 | Ngân sách 200 triệu | Trung bình |
"""


class OutlineLLM:
    def __init__(self):
        self.prompts: list[str] = []
        self.outline_calls = 0

    async def generate(self, *, messages, system, max_tokens, tools, tool_choice):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if "entries" in tools[0]["parameters"]["properties"]:
            self.outline_calls += 1
            if self.outline_calls == 1:  # forgets CON-2 the first time
                return _submit(
                    {
                        "entries": [
                            {"quality_attribute": "Security", "title": "Mã hoá dữ liệu", "source_ids": ["CON-1"]}
                        ],
                        "not_applicable": [],
                    }
                ), None
            return _submit(
                {
                    "entries": [
                        {"quality_attribute": "Security", "title": "Mã hoá dữ liệu", "source_ids": ["CON-1"]},
                        {"quality_attribute": "Privacy", "title": "Xoá dữ liệu", "source_ids": ["CON-1"]},
                    ],
                    "not_applicable": [{"id": "CON-2", "reason": "Ràng buộc ngân sách, không phải chất lượng"}],
                }
            ), None
        count = len(re.findall(r"^\d+\. ", prompt, flags=re.MULTILINE))
        rows = [
            {
                "source_ids": [],
                "quality_attribute": "Security",
                "requirement": f"Yêu cầu {index}",
                "measurement": "Đo",
                "scope_tradeoff": "Toàn hệ thống",
            }
            for index in range(count)
        ]
        return _submit({"rows": rows, "not_applicable": []}), None


@pytest.mark.asyncio
async def test_nfr_outline_maps_every_constraint_before_rows_are_written():
    llm = OutlineLLM()

    result = await generate_parallel_draft(
        "non_functional_requirement", {"constraints_assumptions": CONSTRAINTS}, client=llm, brief="", locale="vi"
    )

    assert not result.failures
    assert llm.outline_calls == 2  # asked again for the unmapped CON-2
    table = result.sections["## Non-Functional Requirements"]
    assert "| NFR-01 | CON-1 |" in table and "| NFR-02 | CON-1 |" in table
    assert "| CON-2 | Ràng buộc ngân sách" in table  # listed as not applicable, with its reason
    assert missing_coverage(table, coverage_items(CONSTRAINTS, "Constraints")) == []


class SectionsLLM:
    """Free-form sections get a two-row table (X-01, X-02); a plan answering a section row by row
    (Validation / Mitigation Plan) gets one row per ID it is asked for."""

    def __init__(self):
        self.prompts: dict[str, list[str]] = {}

    async def generate(self, *, messages, system, max_tokens, tools, tool_choice):
        prompt = messages[0]["content"]
        free_form = re.search(r"section '(## [^']+)'", prompt)
        heading = free_form.group(1) if free_form else re.search(r"rows of the '([^']+)' table", prompt).group(1)
        self.prompts.setdefault(heading.removeprefix("## "), []).append(prompt)
        if free_form:
            return _submit(
                {"content": f"| ID | Nội dung |\n|---|---|\n| X-01 | {heading} một |\n| X-02 | {heading} hai |"}
            ), None
        rows = [
            {
                "source_ids": [item],
                "assumption": f"{item} đầy đủ",
                "risk": f"{item} đầy đủ",
                "method": "Khảo sát",
                "mitigation": "Giảm",
                "owner": "PO",
                "deadline": "Q1",
                "if_false": "Đổi",
                "early_warning": "Chỉ số",
                "contingency": "Dự phòng",
            }
            for item in _ids_asked(prompt)
        ]
        return _submit({"rows": rows, "not_applicable": []}), None


@pytest.mark.asyncio
async def test_sections_plan_writes_every_heading_and_chains_dependent_sections():
    llm = SectionsLLM()

    result = await generate_parallel_draft("constraints_assumptions", {}, client=llm, brief="", locale="vi")

    assert list(result.sections) == [
        "## Constraints",
        "## Assumptions",
        "## Validation Plan",
        "## Risks",
        "## Mitigation Plan",
    ]
    # A dependent section is written one part per row of the section it builds on (two parts at a
    # time here), each seeing its own source row in full, and gets one row per ID.
    validation = llm.prompts["Validation Plan"]
    assert [_ids_asked(prompt) for prompt in validation] == [["X-01"], ["X-02"]]
    # Its own source row in full; another part's item only by name (so it does not write it too).
    assert "| X-01 | ## Assumptions một |" in validation[0] and "| X-02 |" not in validation[0]
    assert all("## Risks" not in prompt for prompt in validation)
    assert "| X-01 | X-01 đầy đủ | Khảo sát |" in result.sections["## Validation Plan"]
    assert "| X-02 | X-02 đầy đủ | Giảm |" in result.sections["## Mitigation Plan"]


def test_every_plan_matches_its_output_contract():
    from app.documents.registry import output_contract

    for artifact_type, plan in PLANS.items():
        headings = output_contract(artifact_type).required_headings
        if hasattr(plan, "chains"):
            assert [job.heading for chain in plan.chains for job in chain] == list(headings) or set(
                job.heading for chain in plan.chains for job in chain
            ) == set(headings)
        else:
            assert (plan.heading,) == headings


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_tool_saves_the_draft_and_write_draft_passes_the_coverage_gate(mock_interrupt, client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.FUNCTIONAL_REQUIREMENT)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)
    await _accept_predecessor(db_session, project_id, "use_case", body=CAPABILITIES)
    await _accept_predecessor(db_session, project_id, "business_rules", body=RULES)

    state = _state(artifact_type="functional_requirement")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()
    config["configurable"]["llm_client"] = ScriptedLLM()

    command = await _draft_in_parallel_impl("- (none)", state, config, "call_1")

    sections = command.update["draft_sections"]
    assert sections["## Functional Requirements"]["done"] is True
    assert "write_draft" in command.update["messages"][0].content

    state["draft_sections"] = sections
    proposed = await _write_draft_impl("FR", "placeholder", state, config, "call_2")

    assert not (proposed.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    assert "| FR-04 | BC5 |" in proposed.update["draft_body"]


@pytest.mark.asyncio
async def test_tool_reports_failed_parts_without_saving_a_partial_draft(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    await _accept_predecessor(db_session, project_id, "use_case", body=CAPABILITIES)
    state = _state(artifact_type="functional_requirement")
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()
    config["configurable"]["llm_client"] = ScriptedLLM(fail_labels={"BC1"})

    command = await _draft_in_parallel_impl("- (none)", state, config, "call_1")

    assert "draft_sections" not in command.update
    [error] = command.update["tool_errors"]
    assert error["code"] == "parallel_draft_incomplete"
    assert "BC1 (TimeoutError" in command.update["messages"][0].content  # narrowed to the failing item
    assert command.update["parallel_draft_cache"]  # the finished part is kept for the retry


class RepeatingLLM(ScriptedLLM):
    async def generate(self, **kwargs):
        message, usage = await super().generate(**kwargs)
        rows = message.tool_calls[0]["args"]["rows"]
        different = {**rows[0], "requirement": rows[0]["requirement"] + " (xuất báo cáo)"}
        return _submit({"rows": [*rows, dict(rows[0]), different], "not_applicable": []}), usage


@pytest.mark.asyncio
async def test_a_row_repeated_verbatim_is_written_once_but_distinct_rows_stay():
    result = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=RepeatingLLM(), brief="", locale="en"
    )

    table = result.sections["## Functional Requirements"]
    assert table.count("Hệ thống hỗ trợ BC1 |") == 1
    assert table.count("Hệ thống hỗ trợ BC1 (xuất báo cáo)") == 1


class FailingSectionLLM(SectionsLLM):
    async def generate(self, **kwargs):
        if "section '## Risks'" in kwargs["messages"][0]["content"]:
            raise TimeoutError("provider hung")
        return await super().generate(**kwargs)


@pytest.mark.asyncio
async def test_a_failed_section_saves_no_sections_but_keeps_the_finished_ones_for_the_retry():
    result = await generate_parallel_draft(
        "constraints_assumptions", {}, client=FailingSectionLLM(), brief="", locale="vi"
    )

    assert result.sections == {}
    assert [failure.label for failure in result.failures] == ["## Risks"]
    # Constraints, Assumptions and the Validation Plan's two parts are kept for the retry.
    assert len(result.cache) == 4


class CutOffLLM(ScriptedLLM):
    """First answer per part has no submit call (cut off at the token limit); records budgets."""

    def __init__(self):
        super().__init__()
        self.budgets: list[int] = []

    async def generate(self, **kwargs):
        self.budgets.append(kwargs["max_tokens"])
        if len(self.budgets) == 1:
            return AIMessage(content="", tool_calls=[]), None
        return await super().generate(**kwargs)


@pytest.mark.asyncio
async def test_a_cut_off_answer_is_retried_with_a_larger_budget(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "parallel_draft_concurrency", 8)  # one item per part
    llm = CutOffLLM()

    result = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=llm, brief="", locale="en"
    )

    assert not result.failures
    first, retry = llm.budgets[0], llm.budgets[1]
    assert retry > first


class ThrottledLLM(ScriptedLLM):
    """Rate-limits the first `times` calls for BC1's part, like Bedrock's ThrottlingException."""

    def __init__(self, times: int):
        super().__init__()
        self.times = times

    async def generate(self, **kwargs):
        if self.times > 0 and "BC1" in _ids_asked(kwargs["messages"][0]["content"]):
            self.times -= 1
            raise RuntimeError("ThrottlingException: Too many requests, please wait before trying again.")
        return await super().generate(**kwargs)


@pytest.mark.asyncio
async def test_a_rate_limited_part_waits_and_retries_without_using_up_its_attempts(monkeypatch):
    from app.config import settings
    from app.graphs.agent_tools import parallel_draft

    monkeypatch.setattr(parallel_draft, "THROTTLE_WAITS", (0.01, 0.01, 0.01, 0.01))
    monkeypatch.setattr(settings, "parallel_draft_concurrency", 8)  # one item per part: no split fallback
    # Three throttles in a row would exhaust the two ordinary attempts; they must not count.
    result = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=ThrottledLLM(times=3), brief="", locale="en"
    )

    assert not result.failures
    assert (
        missing_coverage(
            result.sections["## Functional Requirements"], coverage_items(CAPABILITIES, "Business Capabilities")
        )
        == []
    )


@pytest.mark.asyncio
async def test_every_failed_part_is_reported():
    result = await generate_parallel_draft(
        "functional_requirement",
        {"use_case": CAPABILITIES},
        client=ScriptedLLM(fail_labels={"BC1", "BC5"}),
        brief="",
        locale="en",
    )

    [failure] = result.failures
    assert set(failure.label.split(", ")) == {"BC1", "BC5"}


def test_a_daily_quota_is_not_waited_out():
    from app.graphs.agent_tools.parallel_draft import _is_rate_limited

    assert _is_rate_limited(RuntimeError("ThrottlingException: Too many requests, please wait"))
    assert _is_rate_limited(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not _is_rate_limited(RuntimeError("ThrottlingException: Too many tokens per day, please wait"))
    assert not _is_rate_limited(TimeoutError("provider hung"))


@pytest.mark.asyncio
async def test_rows_two_parts_both_wrote_are_merged_keeping_both_sources():
    # FR-03 merges into FR-01, then FR-01 into FR-04: every source travels along. FR-99 is unknown,
    # and FR-03 cannot absorb anything once it is gone.
    llm = ScriptedLLM(
        duplicates=[
            {"keep": "FR-01", "drop": ["FR-03", "FR-99"]},
            {"keep": "FR-03", "drop": ["FR-02"]},
            {"keep": "FR-04", "drop": ["FR-01"]},
        ]
    )

    result = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=llm, brief="", locale="en"
    )

    table = result.sections["## Functional Requirements"]
    rows = [line for line in table.splitlines() if line.startswith("| FR-")]
    assert [row.split("|")[2].strip() for row in rows] == ["BC2", "BC5, BC1, BC3"]
    assert [row.split("|")[1].strip() for row in rows] == ["FR-01", "FR-02"]  # renumbered
    assert "FR-01 [BC1]: Hệ thống hỗ trợ BC1" in llm.reviews[0]
    assert missing_coverage(table, coverage_items(CAPABILITIES, "Business Capabilities")) == []


@pytest.mark.asyncio
async def test_a_failed_duplicate_review_keeps_the_table_unchanged():
    llm = ScriptedLLM(duplicates="fail")

    result = await generate_parallel_draft(
        "functional_requirement", {"use_case": CAPABILITIES}, client=llm, brief="", locale="en"
    )

    assert not result.failures
    assert (
        len([line for line in result.sections["## Functional Requirements"].splitlines() if line.startswith("| FR-")])
        == 4
    )


@pytest.mark.asyncio
async def test_each_part_sees_the_other_items_scope_and_who_owns_shared_things():
    llm = ScriptedLLM()
    capabilities = CAPABILITIES.replace(
        "| ID | Capability | Goal | Priority |", "| ID | Capability | Scope | Priority |"
    )

    await generate_parallel_draft(
        "functional_requirement", {"use_case": capabilities}, client=llm, brief="", locale="en"
    )

    first = next(prompt for prompt in llm.prompts if _ids_asked(prompt) == ["BC1", "BC2"])
    assert "- BC3: Báo cáo (C3) -- scope: Theo dõi" in first
    assert "Something several items share (e.g. login, history, content moderation) is written ONCE" in first


def test_the_users_own_messages_reach_every_part():
    from langchain_core.messages import HumanMessage, ToolMessage

    from app.graphs.agent_tools.parallel_draft import _brief_block, _user_messages

    messages = [
        HumanMessage(content="viết FR"),
        AIMessage(content="", tool_calls=[{"id": "t1", "name": "ask_user", "args": {}}]),
        ToolMessage(content="tool output", tool_call_id="t1"),
        HumanMessage(content="team 5 người, ra mắt trước 3/2027"),
    ]

    block = _brief_block("", [], _user_messages(messages))

    assert "- viết FR\n- team 5 người, ra mắt trước 3/2027" in block
    assert "tool output" not in block


def test_the_approved_intent_summary_reaches_every_part():
    from app.graphs.agent_tools.parallel_draft import _approved_intent, _brief_block

    messages = [
        AIMessage(
            content="", tool_calls=[{"id": "c1", "name": "confirm_intent", "args": {"summary": "FR cho MVP, bỏ B2B"}}]
        ),
        {"role": "user", "content": "ok"},
    ]

    block = _brief_block("", [], ["ok"], _approved_intent(messages))

    assert "Intent summary the user approved:\nFR cho MVP, bỏ B2B" in block
