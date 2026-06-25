"""Diff scenario transcripts after removing runtime noise.

Raw transcripts contain UUIDs, timestamps, `run_id`, and judge scores that can
vary by environment. This utility preserves meaningful behavior: step order,
action, status, message content, payload kind, tool input, artifact title/body,
and approve/reject decisions.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
ISO_TS_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b")

VOLATILE_KEYS = {
    "id",
    "session_id",
    "run_id",
    "tool_call_id",
    "focused_artifact_id",
    "base_version_id",
    "created_artifact_id",
    "created_version_id",
    "created_at",
    "updated_at",
    "resolved_at",
}

IGNORED_SUMMARY_KEYS = {"mean_overall"}


def _normalize(value: Any, *, include_eval_score: bool, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            next_path = (*path, key)
            if key in VOLATILE_KEYS:
                normalized[key] = "<runtime>"
                continue
            if path == ("summary",) and key in IGNORED_SUMMARY_KEYS and not include_eval_score:
                normalized[key] = "<eval-score>"
                continue
            if path and path[0] == "eval" and key == "score" and not include_eval_score:
                normalized[key] = "<eval-score>"
                continue
            normalized[key] = _normalize(item, include_eval_score=include_eval_score, path=next_path)
        return normalized
    if isinstance(value, list):
        return [
            _normalize(item, include_eval_score=include_eval_score, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        return ISO_TS_RE.sub("<timestamp>", UUID_RE.sub("<uuid>", value))
    return value


def normalize_file(path: Path, *, include_eval_score: bool = False) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized = _normalize(data, include_eval_score=include_eval_score)
    return json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)


def diff_pair(baseline: Path, current: Path, *, include_eval_score: bool = False) -> list[str]:
    left = normalize_file(baseline, include_eval_score=include_eval_score).splitlines()
    right = normalize_file(current, include_eval_score=include_eval_score).splitlines()
    return list(
        difflib.unified_diff(
            left,
            right,
            fromfile=str(baseline),
            tofile=str(current),
            lineterm="",
        )
    )


def _json_files(path: Path) -> dict[str, Path]:
    return {item.name: item for item in sorted(path.glob("*.json"))}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="So transcript JSON với baseline sau khi normalize nhiễu runtime.")
    parser.add_argument("--baseline", type=Path, default=Path("plans/harness-refactor/baseline"))
    parser.add_argument("--transcripts", type=Path, default=Path("tests/integration/scenarios/transcripts"))
    parser.add_argument("--include-eval-score", action="store_true", help="So cả điểm/rationale của judge.")
    args = parser.parse_args(argv)

    baseline = _json_files(args.baseline)
    current = _json_files(args.transcripts)

    missing = sorted(set(baseline) - set(current))
    extra = sorted(set(current) - set(baseline))
    changed: list[str] = []

    if missing:
        print("Thiếu transcript so với baseline:")
        for name in missing:
            print(f"- {name}")
    if extra:
        print("Transcript mới ngoài baseline:")
        for name in extra:
            print(f"- {name}")

    for name in sorted(set(baseline) & set(current)):
        diff = diff_pair(baseline[name], current[name], include_eval_score=args.include_eval_score)
        if not diff:
            continue
        changed.append(name)
        print(f"\n## {name}")
        print("\n".join(diff))

    if missing or extra or changed:
        print(
            f"\nDIFF: {len(changed)} changed, {len(missing)} missing, {len(extra)} extra "
            f"(eval_score={'included' if args.include_eval_score else 'ignored'})."
        )
        return 1

    print(
        f"OK: {len(baseline)} transcript khớp baseline sau normalize "
        f"(eval_score={'included' if args.include_eval_score else 'ignored'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
