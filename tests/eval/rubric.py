"""Re-export the rubric from app.graphs.rubric for backward compatibility.

The real content moved to `app/graphs/rubric.py` (because `app/` must not
import from `tests/`). This module only re-exports so existing imports in
`judge.py` and `test_eval_baseline.py` keep working.
"""

from app.graphs.rubric import *  # noqa: F401,F403
from app.graphs.rubric import RUBRIC_CRITERIA, render_criteria_block  # noqa: F401
