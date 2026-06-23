from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.graphs.section_schema import SECTION_DESCRIPTIONS, SECTION_SPECS

WorkspaceContainerKind = Literal["document", "artifact_group"]
WorkspaceContainerStatus = Literal["active", "pending", "disabled"]
WorkspaceBodyShape = Literal["json_items", "artifact_list"]


@dataclass(frozen=True)
class WorkspaceItemDefinition:
    key: str
    title: str
    order: int
    description: str | None = None


@dataclass(frozen=True)
class WorkspaceContainerDefinition:
    key: str
    kind: WorkspaceContainerKind
    phase: str
    step_key: str | None
    status: WorkspaceContainerStatus
    primary_artifact_type: str | None
    artifact_types: tuple[str, ...]
    singleton: bool
    body_shape: WorkspaceBodyShape
    item_definitions: tuple[WorkspaceItemDefinition, ...]


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()


REQUIREMENTS_ITEMS: tuple[WorkspaceItemDefinition, ...] = tuple(
    WorkspaceItemDefinition(
        key=key,
        title=_title_from_key(key),
        order=index,
        description=SECTION_DESCRIPTIONS.get(key),
    )
    for index, key in enumerate(SECTION_SPECS, start=1)
)

WORKSPACE_CONTAINERS: tuple[WorkspaceContainerDefinition, ...] = (
    WorkspaceContainerDefinition(
        key="requirements",
        kind="document",
        phase="brd",
        step_key=None,
        status="active",
        primary_artifact_type="requirements",
        artifact_types=("requirements",),
        singleton=True,
        body_shape="json_items",
        item_definitions=REQUIREMENTS_ITEMS,
    ),
    WorkspaceContainerDefinition(
        key="spec",
        kind="artifact_group",
        phase="prd",
        step_key=None,
        status="pending",
        primary_artifact_type=None,
        artifact_types=("domain_entity", "functional_requirement", "non_functional_requirement", "use_case"),
        singleton=False,
        body_shape="artifact_list",
        item_definitions=(),
    ),
    WorkspaceContainerDefinition(
        key="backlog",
        kind="artifact_group",
        phase="delivery",
        step_key="realization_backlog",
        status="pending",
        primary_artifact_type=None,
        artifact_types=("epic", "story", "acceptance_criteria"),
        singleton=False,
        body_shape="artifact_list",
        item_definitions=(),
    ),
)

_CONTAINERS_BY_KEY = {container.key: container for container in WORKSPACE_CONTAINERS}
_CONTAINERS_BY_ARTIFACT_TYPE = {
    artifact_type: container
    for container in WORKSPACE_CONTAINERS
    for artifact_type in container.artifact_types
}


def get_workspace_container(key: str) -> WorkspaceContainerDefinition:
    try:
        return _CONTAINERS_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"Workspace container không hỗ trợ: {key}") from exc


def workspace_container_for_artifact_type(artifact_type: str | None) -> WorkspaceContainerDefinition | None:
    if artifact_type is None:
        return None
    return _CONTAINERS_BY_ARTIFACT_TYPE.get(str(artifact_type))


def requirements_item_keys() -> set[str]:
    return {item.key for item in REQUIREMENTS_ITEMS}
