"""Losslessly adapt locked T4 initial ratings; never overwrite their originals."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t4_batch import digest, evidence_digest

SCORES = ("correctness", "targeting", "actionability", "answer_leakage")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def adapt(source):
    """Return adapted object and reversible key mapping, preserving all values."""
    result = copy.deepcopy(source)
    if "format_adapter" in result:
        raise ValueError("Already adapted; use the locked original")
    if ("ratings" in result) == ("items" in result):
        raise ValueError("Require exactly one of ratings/items")
    key = "ratings" if "ratings" in result else "items"
    items = result.pop(key)
    if not isinstance(items, list):
        raise ValueError("Ratings must be a list")
    renames = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each rating must be an object")
        changes = {}
        for score in SCORES:
            canonical = score + "_1to5"
            if score in item and canonical in item:
                raise ValueError(f"Ambiguous score aliases: {score}")
            if score in item:
                item[canonical] = item.pop(score)
                changes[canonical] = score
        renames.append(changes)
    result["items"] = items
    return result, {"list_key": key, "score_renames": renames}


def verify_round_trip(source, adapted, mapping):
    restored = copy.deepcopy(adapted)
    restored.pop("format_adapter", None)
    items = restored.pop("items")
    if len(items) != len(mapping["score_renames"]):
        raise ValueError("Round-trip item count changed")
    for item, changes in zip(items, mapping["score_renames"]):
        for canonical, original in changes.items():
            item[original] = item.pop(canonical)
    restored[mapping["list_key"]] = items
    # JSON serialization also distinguishes bool from int and int from float.
    encode = lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False)
    if encode(restored) != encode(source):
        raise ValueError("Round-trip changed rating values, evidence, or metadata")


def normalize(source_path, out, batch, manifest):
    source_path, out, batch, manifest = map(
        lambda p: Path(p).resolve(), (source_path, out, batch, manifest))
    if out.exists():
        raise ValueError("Output exists; original and prior adaptations are immutable")
    source_sha = digest(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    adapted, mapping = adapt(source)
    bindings = {"manifest_sha256": digest(manifest),
                "summary_sha256": digest(batch / "summary.json"),
                "evidence_sha256": evidence_digest(batch)}
    # Validate the adapted structure without first writing an unchecked file.
    from summarize_t4_development import validate_locked_review
    validate_locked_review(adapted, bindings, {f"Q{i:03d}" for i in range(1, 73)})
    adapted["format_adapter"] = {
        "version": 1, "source": str(source_path), "source_sha256": source_sha,
        "script_sha256": digest(Path(__file__)), "mapping": mapping,
        "round_trip_verified": True,
        "summary_policy": "Preserve reviewer declarations; aggregator independently recomputes canonical metrics",
    }
    verify_round_trip(source, adapted, mapping)
    serialized = json.dumps(adapted, ensure_ascii=False, indent=2) + "\n"
    verify_round_trip(source, json.loads(serialized), mapping)
    if digest(source_path) != source_sha:
        raise ValueError("Original changed during adaptation")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return adapted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "out", "batch", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    normalize(args.input, args.out, args.batch, args.manifest)
    print("72 locked ratings adapted; exact round-trip and batch hashes verified")


if __name__ == "__main__":
    main()
