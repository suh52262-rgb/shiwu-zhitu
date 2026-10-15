"""Freeze reviewed RAG development data before retrieval; never overwrite a freeze.

This freezes provenance, not a quality verdict. Previous batch manifests remain
unchanged. No database or model calls are made here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CELLS = {f"{topic}|{kind}|{size}" for topic in ("em", "optics")
         for kind in ("normal", "miscon") for size in ("single", "comb")}


def validate(data: dict) -> dict:
    if data.get("status") != "AI_REVIEWED_DEVELOPMENT":
        raise ValueError("Only AI_REVIEWED_DEVELOPMENT data may be frozen")
    cases = data.get("drafts")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Nonempty drafts required")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    counts = Counter()
    for case in cases:
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"Missing query: {case['id']}")
        cell = case.get("cell_key")
        if cell not in CELLS:
            raise ValueError(f"Invalid cell: {case['id']}")
        if not case.get("required_answer_points") or not case.get("review_provenance"):
            raise ValueError(f"Missing points or review: {case['id']}")
        sets = case.get("acceptable_evidence_sets")
        if not isinstance(sets, list) or not sets:
            raise ValueError(f"Missing evidence sets: {case['id']}")
        for group in sets:
            refs, ids = group.get("all_of", []), group.get("all_of_block_ids", [])
            if not refs or len(refs) != len(ids) or not all(ids):
                raise ValueError(f"Missing/mismatched identity: {case['id']}")
            if len(set(ids)) != len(ids) or len(set(refs)) != len(refs):
                raise ValueError(f"Duplicate evidence identity: {case['id']}")
            # OR alternatives must each cover the entire question. Supplemental
            # material belongs elsewhere, never in acceptable_evidence_sets.
            expected = {f"要点{i}" for i in range(1, len(case["required_answer_points"]) + 1)}
            if set(group.get("supports", [])) != expected:
                raise ValueError(f"Partial alternative or missing point mapping: {case['id']}")
            if cell.endswith("|single") and len(ids) != 1:
                raise ValueError(f"Single cell contains a combination: {case['id']}")
        if cell.endswith("|comb") and any(len(s["all_of"]) == 1 for s in sets):
            raise ValueError(f"Combination cell has a sufficient single: {case['id']}")
        counts[cell] += 1
    return {"n_items": len(cases), "cell_counts": {k: counts[k] for k in sorted(CELLS)},
            "empty_cells": sorted(CELLS - counts.keys()),
            "target_24_remaining": {k: max(0, 3 - counts[k]) for k in sorted(CELLS)}}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite an existing freeze")
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    ledger = validate(data)
    paths = [args.annotation.resolve(), Path(__file__).resolve()]
    paths += [p.resolve() for p in args.include]
    paths += [p for base in ("backend/app", "prompts") for p in (ROOT / base).rglob("*")
              if p.suffix in (".py", ".md")]
    paths += [ROOT / "requirements.txt"]
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(set(paths))}
    manifest = {"schema_version": 1, "kind": "rag_closeout_development_freeze",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_version": data["annotation_version"], "annotation_file": str(args.annotation),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                      text=True).strip(),
                "files": files, **ledger,
                "scope": "AI-reviewed development probes; not a holdout or quality gate",
                "configuration": {"RAG_RERANK": "0", "RAG_QUERY_EXPANSION": "0",
                                  "top_k": 5, "tutor_hits": 3, "budget_chars_per_hit": 400},
                "dependencies": {"python": sys.version, **{
                    name: importlib.metadata.version(name) for name in
                    ("qdrant-client", "SQLAlchemy", "sentence-transformers", "torch", "transformers")}},
                "previous_frozen_batches_unchanged": True,
                "metrics": "Computed only after this freeze; no historical scores imported"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(ledger, ensure_ascii=False))


if __name__ == "__main__":
    main()
