"""Core use-case modeling primitives.

This package deliberately has no router or database writes.  It turns the current BRD/PRD
document components into a source snapshot, gives the agent a strict generation contract, and
validates the returned model before a future API or persistence layer can consume it.
"""

from app.use_cases.harness import UseCaseGenerationHarness
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    UseCaseModel,
    UseCaseValidationReport,
)
from app.use_cases.rules import validate_use_case_model
from app.use_cases.source_loader import (
    load_project_requirements_source,
    source_snapshot_from_documents,
)

__all__ = [
    "RequirementsSourceSnapshot",
    "UseCaseGenerationHarness",
    "UseCaseModel",
    "UseCaseValidationReport",
    "validate_use_case_model",
    "load_project_requirements_source",
    "source_snapshot_from_documents",
]
