"""Transcript recorder — saves each scenario's raw messages for validation.

One JSON file per scenario under `tests/scenarios/transcripts/`. The transcript
captures every step taken (action + resulting API snapshot), so a human (or a
later eval pass) can replay the exact conversation and draw conclusions.
"""

import json
from pathlib import Path
from typing import Any

TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"


class TranscriptRecorder:
    """Accumulates scenario events and writes them to a JSON transcript."""

    def __init__(self, scenario_name: str) -> None:
        self.scenario_name = scenario_name
        self.steps: list[dict[str, Any]] = []
        self.eval: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def record_step(self, *, action: dict[str, Any], snapshot: dict[str, Any]) -> None:
        """Record one driver action and the API snapshot observed right after it."""
        self.steps.append({"step": len(self.steps) + 1, "action": action, "snapshot": snapshot})

    def record_eval(self, *, artifact_type: str, title: str, body: str, score: dict[str, Any]) -> None:
        """Record a judge score for one produced artifact."""
        self.eval.append({"artifact_type": artifact_type, "title": title, "body": body, "score": score})

    def set_summary(self, **fields: Any) -> None:
        self.summary.update(fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "summary": self.summary,
            "steps": self.steps,
            "eval": self.eval,
        }

    def write(self) -> Path:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        path = TRANSCRIPTS_DIR / f"{self.scenario_name}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
