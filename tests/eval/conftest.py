"""Re-export the scenario-suite fixtures so behavior scenarios can run from tests/eval.

The behavior eval drives real HTTP sessions through the scenario framework in
tests/integration/scenarios; pytest resolves fixtures per directory, so the fixtures are imported
here by name. The autouse overrides (`db_session=None`, file-based scenario DB) are harmless to
the other tests in this directory — they never touch the DB.
"""

from tests.integration.scenarios.conftest import (  # noqa: F401
    _scenario_tables,
    db_session,
    scenario_env,
    scenario_project,
)
