"""read_artifact / read_source_documents — side-effect-free reads (no interrupt, no DB write).

read_artifact pulls a sibling/ancestor artifact's body by id so the model can reuse content
already recorded instead of re-asking the user; read_source_documents returns bounded excerpts of
stored project source documents. Both loop back through analyze_node like the note tools — they
append a ToolMessage and never interrupt. Project-scoped via the query layer's project_id filter,
so a session can only read its own project's artifacts/documents. Self-contained: no import back
into the coordinator.
"""

import json
import uuid
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from app.graphs.tools import read_current_body
from app.graphs.tools import read_source_documents as read_source_documents_query

# Cap a single read so a large body cannot dominate the analyze prompt; the head is enough to orient,
# and a focused draft is reached through write_draft/current_draft_body, not this tool.
READ_ARTIFACT_MAX_CHARS = 8000
READ_SOURCE_DOCUMENT_MAX_CHARS = 8000
READ_SOURCE_DOCUMENT_MAX_ITEMS = 3


def _read_artifact_source_context(result: dict[str, Any], excerpt: str) -> list[dict[str, Any]]:
    version_id = str(result.get("current_version_id") or "").strip()
    artifact_id = str(result.get("artifact_id") or "").strip()
    if not version_id or not artifact_id or not excerpt.strip():
        return []
    return [
        {
            "source_kind": "predecessor_version",
            "artifact_id": artifact_id,
            "predecessor_version_id": version_id,
            "title": result.get("title"),
            "locator": f"artifact_version:{version_id}",
            "excerpt": excerpt,
            "truncated": bool(result.get("truncated")),
        }
    ]


def _source_document_source_context(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for document in documents:
        excerpt = str(document.get("excerpt") or "").strip()
        source_document_id = str(document.get("id") or "").strip()
        if not source_document_id or not excerpt:
            continue
        entries.append(
            {
                "source_kind": "source_document",
                "source_document_id": source_document_id,
                "title": document.get("title"),
                "locator": document.get("locator"),
                "excerpt": excerpt,
                "truncated": bool(document.get("truncated")),
            }
        )
    return entries


async def _read_artifact_impl(artifact_id: str, config: RunnableConfig, tool_call_id: str):
    cfg = config["configurable"]
    project_id_raw = cfg.get("project_id")
    session_factory = cfg.get("session_factory")
    try:
        target_id = uuid.UUID(str(artifact_id))
    except (ValueError, TypeError):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"read_artifact: invalid id ({artifact_id!r}).",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    result = None
    if session_factory is not None and project_id_raw is not None:
        async with session_factory() as db:
            result = await read_current_body(
                db=db,
                project_id=uuid.UUID(str(project_id_raw)),
                artifact_id=target_id,
            )

    if result is None:
        content = f"read_artifact: artifact not found {artifact_id} (or has no content yet) in project."
        source_context = []
    else:
        body = result["body"] or ""
        excerpt = body[:READ_ARTIFACT_MAX_CHARS]
        if len(body) > READ_ARTIFACT_MAX_CHARS:
            body = excerpt + "\n\n…(remaining content truncated)"
            result = {**result, "truncated": True}
        else:
            body = excerpt
        content = f"# {result['title']}\n\n{body}"
        source_context = _read_artifact_source_context(result, excerpt)
    update: dict[str, Any] = {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    if source_context:
        update["source_context"] = source_context
    return Command(update=update)


@tool
async def read_artifact(
    id: Annotated[str, "The artifact id (UUID) to read — a sibling or ancestor in this project."],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the current body of another artifact in this project by its id.

    Use to pull context from a sibling or ancestor artifact (e.g. the parent BRD) instead of asking
    the user for content that already exists. Read-only and non-interrupting; the body is returned to
    you, not shown to the user.
    """
    return await _read_artifact_impl(id, config, tool_call_id)


def _normalize_source_document_ids(ids: Any) -> tuple[bool, list[uuid.UUID], list[str]]:
    if ids is None:
        return False, [], []
    if isinstance(ids, str):
        raw_items = [ids]
    else:
        try:
            raw_items = list(ids or [])
        except TypeError:
            raw_items = [ids]
    parsed: list[uuid.UUID] = []
    invalid: list[str] = []
    for raw in raw_items[:READ_SOURCE_DOCUMENT_MAX_ITEMS]:
        try:
            parsed.append(uuid.UUID(str(raw)))
        except (TypeError, ValueError):
            invalid.append(str(raw))
    return True, parsed, invalid


async def _read_source_documents_impl(ids: Any, config: RunnableConfig, tool_call_id: str) -> Command:
    cfg = config["configurable"]
    project_id_raw = cfg.get("project_id")
    session_factory = cfg.get("session_factory")
    ids_supplied, parsed_ids, invalid_ids = _normalize_source_document_ids(ids)
    documents: list[dict[str, Any]] = []
    if session_factory is not None and project_id_raw is not None and (parsed_ids or not ids_supplied):
        async with session_factory() as db:
            documents = await read_source_documents_query(
                db=db,
                project_id=uuid.UUID(str(project_id_raw)),
                source_document_ids=parsed_ids or None,
                limit=READ_SOURCE_DOCUMENT_MAX_ITEMS,
                max_chars=READ_SOURCE_DOCUMENT_MAX_CHARS,
            )
    payload = {
        "documents": documents,
        "invalid_ids": invalid_ids,
        "strategy": {
            "max_items": READ_SOURCE_DOCUMENT_MAX_ITEMS,
            "max_chars_per_document": READ_SOURCE_DOCUMENT_MAX_CHARS,
            "mode": "bounded_excerpt",
        },
    }
    source_context = _source_document_source_context(documents)
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id)]
    }
    if source_context:
        update["source_context"] = source_context
    return Command(update=update)


@tool("read_source_documents")
async def read_source_documents_tool(
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    ids: Annotated[
        list[str] | None,
        "Optional source document UUIDs. Omit to read the latest project source documents as bounded excerpts.",
    ] = None,
) -> Command:
    """Read bounded stored source-document text for this project.

    Use before drafting when the user references uploaded/pasted source documents. The returned
    excerpts are available to generation and are snapshotted as evidence only if a draft is proposed.
    """
    return await _read_source_documents_impl(ids, config, tool_call_id)
