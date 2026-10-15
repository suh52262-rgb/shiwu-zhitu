"""Conservative suppression of consecutive identical OCR teaching inputs.

This is text-call deduplication, never evidence that a student is stuck.
"""
from __future__ import annotations

import re


def _comparison_text(text: str) -> str:
    # Preserve operators, signs, numbers, units, case and line boundaries.
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip()
                     for line in text.strip().replace("\r\n", "\n").split("\n"))


def coalesce_snapshots(snapshots: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep changes, including A→B→A; record duplicates without rewriting source time.

    Explicit requests for help must bypass this helper at the caller. The original
    perception export remains the evidence for all observed frames.
    """
    selected, skipped = [], []
    previous = None
    for snapshot in snapshots:
        key = _comparison_text(snapshot.get("text") or "")
        if selected and key == previous:
            skipped.append({"ts": snapshot["ts"],
                            "duplicate_of_ts": selected[-1]["ts"],
                            "reason": "consecutive_identical_ocr",
                            "cognitive_stuck": None})
        else:
            selected.append(dict(snapshot))
            previous = key
    return selected, skipped
