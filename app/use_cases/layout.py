"""Deterministic React Flow layout generation using the ELK JavaScript engine."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import settings


class DiagramLayoutError(RuntimeError):
    """Raised when the ELK layout worker cannot produce a valid layout."""


_RUNNER = Path(__file__).resolve().parents[2] / "diagram_layout" / "layout.mjs"


async def generate_diagram_layout(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the ELK worker and return a renderer-neutral React Flow layout payload."""

    node = shutil.which("node")
    if not node:
        raise DiagramLayoutError("Node.js is required to generate the ELK diagram layout")
    if not _RUNNER.is_file():
        raise DiagramLayoutError(f"ELK layout worker is missing: {_RUNNER}")

    # A blocking subprocess.run in a worker thread rather than asyncio.create_subprocess_exec: the
    # latter needs a Proactor event loop on Windows, and uvicorn --reload runs a Selector loop
    # there, where it raises a bare NotImplementedError.
    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            [node, str(_RUNNER)],
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            timeout=settings.use_case_layout_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise DiagramLayoutError("ELK layout generation timed out") from exc

    error_text = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode:
        raise DiagramLayoutError(error_text[-500:] or "ELK layout worker failed")
    try:
        result = json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DiagramLayoutError("ELK layout worker returned invalid JSON") from exc
    if not isinstance(result, dict) or result.get("engine") != "elk":
        raise DiagramLayoutError("ELK layout worker returned an invalid layout")
    return result
