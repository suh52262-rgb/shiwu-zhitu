"""Replay archived hits through production expansion and audit PG provenance.

No retrieval, embedding, paid model, or database writes. The input archive is
immutable; --out must name a new directory. Counts are not a semantic-quality gate.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PromptCapture:
    """Exercise the actual Tutor renderer, stopping at its model boundary."""
    def __init__(self):
        self.messages = []

    def chat_structured_resilient(self, system, user, schema):
        self.messages.append({"system": system, "user": user})
        return SimpleNamespace(value=schema(hint="请核对这一步的适用条件。"), degraded=False, reasons=[])


def detect_tutor_budget(render):
    sentinel = "AUDIT_WINDOW_" + "".join(f"{i:04d}|" for i in range(500))
    user = render([{"text": sentinel, "context": sentinel, "source": "audit_budget", "page": 1}])["user"]
    start = user.find(sentinel[:32])
    if start < 0:
        raise ValueError("Tutor did not inject the probe; cannot determine its actual window")
    n = 0
    for actual, expected in zip(user[start:], sentinel):
        if actual != expected:
            break
        n += 1
    if not 32 <= n < len(sentinel):
        raise ValueError("Probe too short or Tutor context format changed; audit requires review")
    return n


def inspect_context(hit, facts, tutor_budget, include_generated=False):
    """Check returned spans against independent PG facts; never rebuild selection."""
    context = hit.get("context", "")
    blocks = hit.get("context_blocks")
    issues, fragments, context_facts = [], [], []
    if not isinstance(blocks, list) or not blocks:
        issues.append("missing_context_blocks")
        blocks = []
    cursor = 0
    hit_start = None
    for block in blocks:
        fact = facts.get(block.get("id"))
        if fact is None:
            issues.append("unknown_block_id")
            continue
        a, b = block.get("context_start"), block.get("context_end")
        u, v = block.get("text_start"), block.get("text_end")
        if not all(type(x) is int for x in (a, b, u, v)):
            issues.append("missing_or_invalid_offsets")
            continue
        if not (cursor <= a < b <= len(context) and 0 <= u < v <= len(fact["text"])):
            issues.append("invalid_or_unordered_offsets")
            continue
        if context[cursor:a].strip():
            issues.append("unattributed_gap")
        if context[a:b] != fact["text"][u:v]:
            issues.append("source_text_mismatch")
        for key in ("page", "type", "source_kind", "block_index"):
            if block.get(key) != fact.get(key):
                issues.append("source_metadata_mismatch:" + key)
        if fact["doc_id"] != hit.get("source"):
            issues.append("cross_document_reference")
        if block.get("source") != fact["doc_id"]:
            issues.append("source_metadata_mismatch:source")
        context_facts.append(fact)
        if fact["text"] == hit.get("text") and u == 0:
            hit_start = a if hit_start is None else min(hit_start, a)
        cursor = b
        if a < tutor_budget:
            fragments.append({"id": block["id"], "page": fact["page"], "type": fact["type"],
                "source_kind": fact["source_kind"], "indexable": fact["indexable"] is not False,
                "is_generated": bool(fact["is_generated"]), "note": fact.get("note", ""),
                "context_start": a, "context_end": min(b, tutor_budget),
                "text_start": u, "text_end": u + min(b, tutor_budget) - a,
                "sent_text": context[a:min(b, tutor_budget)]})
    if context[cursor:].strip():
        issues.append("unattributed_tail")
    if hit_start is None:
        issues.append("hit_not_attributed")
    anchor = hit.get("text", "")[:80]
    flags = {
        "unindexable_material_in_context": any(b["indexable"] is False for b in context_facts),
        "generated_material_in_context": any(b["is_generated"] for b in context_facts),
        "cross_page_material_in_tutor": any(b["page"] != hit["page"] for b in fragments),
        "generated_material_in_tutor": any(b["is_generated"] for b in fragments),
        "unindexable_material_in_tutor": any(not b["indexable"] for b in fragments),
        "hit_start_after_tutor_cut": hit_start is not None and hit_start >= tutor_budget,
        "hit_start_after_context_cut": hit_start is not None and hit_start >= len(context),
        "hit_anchor_missing_from_context": not anchor or anchor not in context,
        "hit_anchor_missing_from_tutor": not anchor or anchor not in context[:tutor_budget],
    }
    if flags["unindexable_material_in_tutor"]:
        issues.append("unindexable_material_sent")
    if flags["generated_material_in_tutor"] and not include_generated:
        issues.append("generated_material_sent_without_opt_in")
    if flags["unindexable_material_in_context"]:
        issues.append("unindexable_material_in_context")
    if flags["generated_material_in_context"] and not include_generated:
        issues.append("generated_material_in_context_without_opt_in")
    if flags["hit_start_after_tutor_cut"]:
        issues.append("hit_start_outside_tutor_window")
    if flags["hit_anchor_missing_from_tutor"]:
        issues.append("hit_anchor_lost")
    return {"flags": flags, "hit_context_start": hit_start,
            "issues": sorted(set(issues)), "sent_fragments": fragments}


def replay_case(row, facts, render, tutor_budget, include_generated=False):
    from app.rag import retriever
    # Drop archived expansions: setdefault/fallback must not reuse old material.
    hits = [{k: deepcopy(v) for k, v in hit.items() if k not in ("context", "context_blocks")}
            for hit in row.get("hits", [])[:3]]
    if row.get("status") != "succeeded" or not hits:
        return {"id": row["id"], "status": "unverified", "issues": ["no_successful_archived_hits"], "checks": []}
    originals = [h["text"] for h in hits]
    retriever._expand_context(hits, include_generated=include_generated)
    prompt = render(hits)
    checks = []
    for rank, hit in enumerate(hits, 1):
        check = inspect_context(hit, facts, tutor_budget, include_generated)
        if hit["text"] != originals[rank-1]:
            check["issues"].append("hit_text_mutated")
        checks.append({"rank": rank, "hit": hit, **check})
    # Check captured model input independently of the production formatter. Source
    # labels now interrupt the context string; a contiguous-prefix check is invalid.
    # Compare the complete ordered fragment inventory, including source and page,
    # so dropped, added, reordered or mislabelled material cannot silently pass.
    marker = "【教材参考（可引用其思路，不要照抄）】\n"
    material = prompt["user"].partition(marker)[2].split("\n\n【", 1)[0]
    actual = re.findall(r"「(.*?)」（(.*?) 第(.*?)页(?:，\*\*该页此处被截断\*\*)?）",
                        material, flags=re.DOTALL)
    expected = [(f["sent_text"], str(c["hit"]["source"]), str(f["page"]))
                for c in checks for f in c["sent_fragments"] if f["sent_text"].strip()]
    if not expected or actual != expected:
        for check in checks:
            check["issues"].append("actual_tutor_payload_mismatch")
    return {"id": row["id"], "status": "verified" if all(not c["issues"] for c in checks) else "failed",
            "checks": checks, "captured_prompt": prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path, help="Archive containing baseline.jsonl; never modified")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing paths refused")
    parser.add_argument("--include-generated", action="store_true")
    args = parser.parse_args()
    # Explicit historical-policy audit; do not imply this verifies the new assembler.
    import os
    os.environ["RAG_MATERIAL_POLICY"] = "legacy"
    archive = args.audit_dir.resolve()
    if args.out.resolve() != archive and args.out.resolve().is_relative_to(archive):
        raise ValueError("Output must be outside the immutable input archive")
    args.out.mkdir(parents=True, exist_ok=False)
    source = archive / "baseline.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    paths = [Path(__file__).resolve(), ROOT / "backend/app/rag/retriever.py",
             ROOT / "backend/app/rag/block_store.py", ROOT / "backend/app/agent/graph.py",
             ROOT / "prompts/tutor/v1.md"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for path in paths:
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "archive": str(archive), "baseline_sha256": digest(source), "source_hashes": hashes,
        "include_generated": args.include_generated, "material_policy": "legacy", "postgres_read_only": True,
        "scope": "Archived candidate replay through current production; fake model captures actual Tutor prompt; no semantic evaluation"}
    write(args.out / "manifest.json", manifest)
    from sqlalchemy import event, text
    from app.db.postgres import get_engine
    from app.agent import graph
    from app.models.student import StudentState
    engine = get_engine()
    @event.listens_for(engine, "connect")
    def read_only(connection, _):
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        connection.commit()
    requested = {}
    for row in rows:
        for hit in row.get("hits", [])[:3]:
            requested.setdefault(hit["source"], set()).update((hit["page"], hit["page"] + 1))
    facts = {}
    with engine.connect() as conn:
        for doc, pages in requested.items():
            blocks = conn.execute(text("select id,doc_id,page,block_index,type,text,indexable,is_generated,note,source_kind "
                "from document_blocks where doc_id=:doc and page=ANY(:pages) order by page,block_index"),
                {"doc": doc, "pages": sorted(pages)}).mappings()
            facts.update({b["id"]: dict(b) for b in blocks})
    write(args.out / "pg_facts.json", facts)
    manifest["pg_facts_sha256"] = digest(args.out / "pg_facts.json")
    capture = PromptCapture()
    def render(hits):
        graph.generate_hint(StudentState(hint_level=1, current_answer="审计替身输入", retrieved_context=hits))
        return capture.messages.pop()
    results = []
    with patch.object(graph, "get_llm", lambda role: capture):
        budget = detect_tutor_budget(render)
        with (args.out / "replays.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                result = replay_case(row, facts, render, budget, args.include_generated)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                results.append(result)
    counts = Counter(checked_contexts=sum(len(r["checks"]) for r in results), verified_contexts=0)
    for name in ("unindexable_material_in_tutor", "generated_material_in_tutor", "cross_page_material_in_tutor",
                 "unindexable_material_in_context", "generated_material_in_context",
                 "hit_start_after_tutor_cut", "hit_start_after_context_cut", "hit_anchor_missing_from_context",
                 "hit_anchor_missing_from_tutor"):
        counts[name] = 0
    defects = Counter()
    for result in results:
        defects.update(result.get("issues", []))
        for check in result["checks"]:
            counts.update(k for k, v in check["flags"].items() if v)
            counts["verified_contexts"] += not check["issues"]
            defects.update(check["issues"])
    drift = [str(p.relative_to(ROOT)) for p in paths if digest(p) != hashes[str(p.relative_to(ROOT))]]
    baseline_unchanged = digest(source) == manifest["baseline_sha256"]
    summary = {"audit_schema_version": 3, "scope": manifest["scope"], "counts": dict(counts),
        "case_status": dict(Counter(r["status"] for r in results)), "issues": dict(defects),
        "tutor_window_chars_observed": budget, "source_drift": drift,
        "baseline_unchanged": baseline_unchanged,
        "passed": bool(results) and all(r["status"] == "verified" for r in results) and not drift and baseline_unchanged,
        "semantic_quality_verified": False, "paid_model_requests": 0,
        "finished_at": datetime.now(timezone.utc).isoformat()}
    manifest["tutor_window_chars_observed"] = budget
    write(args.out / "manifest.json", manifest)
    write(args.out / "context_provenance_audit.json", summary)
    write(args.out / "artifact_hashes.json", {str(p.relative_to(args.out)): digest(p)
        for p in sorted(args.out.rglob("*")) if p.is_file()})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
