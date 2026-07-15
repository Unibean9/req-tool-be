"""Negative-proof regression test for the required CI lane (Phase 8 Step 1).

Guards two invariants that GitHub Actions itself does not check for us:

1. Every `tests/integration/*_postgres.py` file is actually executed by some non-`deploy` job's
   `run:` step — a Postgres integration test that exists on disk but is never wired into any job
   is a silent pass in CI (it only skips when `AGENT_TURN_POSTGRES_URL` is unset, per each file's
   own fixture), not a red build.
2. `deploy.needs` is a superset of every other job id in the workflow — a job added without being
   added to `deploy.needs` would let `deploy` run even if that job is red or missing.

This test is pure YAML parsing (no Postgres, no network) so it belongs in the deterministic PR
suite, and it must fail loudly if either invariant regresses — see the two negative-proof tests
below, which assert the checking logic itself catches a deliberately broken workflow.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
POSTGRES_TEST_DIR = REPO_ROOT / "tests" / "integration"


def _postgres_test_basenames() -> list[str]:
    return sorted(path.name for path in POSTGRES_TEST_DIR.glob("*_postgres.py"))


def _job_run_steps_text(job: dict) -> str:
    steps = job.get("steps") or []
    return "\n".join(str(step.get("run") or "") for step in steps)


def _assert_postgres_lane_complete(workflow: dict, postgres_test_basenames: list[str]) -> None:
    """Raise AssertionError if any Postgres integration test file is not wired into a non-deploy
    job, or if `deploy.needs` is missing any other job id."""
    jobs = workflow.get("jobs") or {}
    non_deploy_jobs = {job_id: job for job_id, job in jobs.items() if job_id != "deploy"}

    referenced_text = "\n".join(_job_run_steps_text(job) for job in non_deploy_jobs.values())
    missing = [name for name in postgres_test_basenames if name not in referenced_text]
    assert not missing, f"Postgres integration test file(s) not wired into any required CI job: {missing}"

    deploy_job = jobs.get("deploy") or {}
    deploy_needs = deploy_job.get("needs") or []
    if isinstance(deploy_needs, str):
        deploy_needs = [deploy_needs]
    missing_needs = [job_id for job_id in non_deploy_jobs if job_id not in deploy_needs]
    assert not missing_needs, f"deploy.needs is missing required job id(s): {missing_needs}"


@pytest.fixture
def ci_workflow() -> dict:
    with CI_WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def postgres_test_basenames() -> list[str]:
    basenames = _postgres_test_basenames()
    assert basenames, "expected at least one tests/integration/*_postgres.py file to exist"
    return basenames


def test_every_postgres_integration_test_file_is_wired_into_a_required_ci_job(ci_workflow, postgres_test_basenames):
    _assert_postgres_lane_complete(ci_workflow, postgres_test_basenames)


def test_deploy_needs_is_a_superset_of_every_other_job_id(ci_workflow, postgres_test_basenames):
    jobs = ci_workflow.get("jobs") or {}
    deploy_needs = jobs["deploy"]["needs"]
    other_job_ids = [job_id for job_id in jobs if job_id != "deploy"]
    assert set(other_job_ids).issubset(set(deploy_needs))


def test_gate_check_fails_when_a_postgres_test_file_is_not_wired_into_any_job(ci_workflow, postgres_test_basenames):
    """Negative proof: the check itself must go red when a Postgres test file is missing from
    every non-deploy job's `run:` steps — proves this is a real gate, not an inspection-only test."""
    with pytest.raises(AssertionError):
        _assert_postgres_lane_complete(ci_workflow, [*postgres_test_basenames, "test_not_actually_wired_postgres.py"])


def test_gate_check_fails_when_deploy_needs_drops_a_required_job(ci_workflow, postgres_test_basenames):
    """Negative proof: the check itself must go red when `deploy.needs` no longer covers every
    other job id — proves this is a real gate, not an inspection-only test."""
    broken = {"jobs": dict(ci_workflow["jobs"])}
    broken["jobs"]["deploy"] = {**broken["jobs"]["deploy"], "needs": []}
    with pytest.raises(AssertionError):
        _assert_postgres_lane_complete(broken, postgres_test_basenames)
