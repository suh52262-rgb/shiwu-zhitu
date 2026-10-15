"""New free replay, never a reconstruction of an archived run.

Uses production retrieval and delivery. Exact excerpt coverage is lexical evidence,
not a semantic sufficiency verdict. Existing outputs are never overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def verify_freeze(path, annotation):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = [name for name, expected in manifest["files"].items()
                if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
    relative = str(annotation.resolve().relative_to(ROOT)).replace("\\", "/")
    if manifest["files"].get(relative) != digest(annotation):
        failures.append("annotation not bound to freeze")
    if failures:
        raise ValueError(f"Frozen inputs changed: {failures}")
    return {"manifest_sha256": digest(path), "files_verified": len(manifest["files"]), "failures": []}

def validate_cases(data):
    cases = data.get("drafts", data.get("cases", []))
    if not cases or len({c.get("id") for c in cases}) != len(cases):
        raise ValueError("Nonempty unique case IDs required")
    for c in cases:
        if not c.get("id") or not str(c.get("query", "")).strip():
            raise ValueError("Missing case ID or query")
        for s in c.get("acceptable_evidence_sets", []):
            if len(s.get("all_of", [])) != len(s.get("all_of_block_ids", [])):
                raise ValueError("Reference and stable ID counts differ")
    return cases

def classify(bid, hits, delivered, tutor_hits=3):
    """Attribute finite returned results only, not the unrecorded vector pool."""
    if not bid or any(not h.get("block_id") for h in hits):
        return "unverifiable_identity"
    if any(str(d.get("id")) == str(bid) for d in delivered):
        return "delivered_id_present"
    top = hits[:tutor_hits]
    def contains(h):
        return any(str(b.get("id")) == str(bid) for b in h.get("context_blocks", []))
    if any(contains(h) for h in top):
        return "actual_delivery_window_excluded"
    if any(str(h.get("block_id")) == str(bid) for h in top):
        return "expansion_omitted"
    if any(str(h.get("block_id")) == str(bid) or contains(h) for h in hits[tutor_hits:]):
        return "rank_excluded"
    return "not_seen_in_returned_top_k"

def quote_coverage(excerpt, delivered):
    bid = str(excerpt.get("block_id") or "")
    quote = excerpt.get("excerpt_verbatim") or ""
    pieces = [d.get("text", "") for d in delivered if str(d.get("id")) == bid]
    return {"ref": excerpt.get("ref"), "block_id": bid, "quote": quote,
            "exact_quote_delivered": bool(quote and any(quote in p for p in pieces)),
            "identity_present": bool(pieces), "semantic_support": "not_automatically_judged"}

def identity_issues(hits, delivered, facts):
    issues=[]
    for h in hits:
        bid=str(h.get("block_id") or "")
        f=facts.get(bid)
        if not bid or not f or any(h.get(k)!=f.get(v) for k,v in (("source","doc_id"),("page","page"),("text","text"))):
            issues.append({"kind":"hit_identity_or_fact_mismatch","id":bid or None})
    for d in delivered:
        bid=str(d.get("id") or "")
        f=facts.get(bid)
        if not bid or not f or d.get("source")!=f.get("doc_id") or d.get("page")!=f.get("page") or str(d.get("text") or "") not in f.get("text", ""):
            issues.append({"kind":"delivered_identity_or_fact_mismatch","id":bid or None})
    return issues

def audit_archive(folder):
    rows = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name == "summary.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        gold = {x for s in d["evidence_sets"] for x in s.get("all_of_block_ids", [])}
        direct = {x for s in d["evidence_sets"] for x in s.get("direct_hit", [])}
        rows.append({"id": d["annotation_id"], "sha256": digest(p),
          "returned_distinct_direct_ids": d["layer_1_direct_hits"],
          "specified_gold_direct_count": len(gold & direct),
          "specified_gold_count": len(gold), "id_presence_satisfied": d["satisfied_any"],
          "missing_recordings": ["actual_hits_and_ranks", "expanded_context", "delivered_text"],
          "archived_semantic_attribution": "unverifiable_missing_recordings"})
    return {"rows": rows, "specified_gold_any_direct": sum(r["specified_gold_direct_count"] > 0 for r in rows),
            "warning": "layer_1_direct_hits counts every returned identity, not specified-gold hits; no old rank or semantic reconstruction is possible."}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("annotation", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--freeze", type=Path, help="Verify every frozen input before and after the run")
    args = ap.parse_args()
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    cases = validate_cases(data)
    frozen_before = verify_freeze(args.freeze, args.annotation) if args.freeze else None
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["RAG_RERANK"] = "0"
    os.environ["RAG_QUERY_EXPANSION"] = "0"
    os.environ["EMBEDDING_PROVIDER"] = "bge"
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "backend"))
    from sqlalchemy import event, text
    from qdrant_client import QdrantClient
    from app.config import get_settings
    from app.db.postgres import get_engine
    from app.rag import retriever, query_expansion, rerank
    from app.agent import tools as agent_tools
    from app.agent.tools import retrieve_knowledge
    from app.agent.graph import delivered_rag_sources, RAG_TUTOR_BUDGET
    from app.services.rag_materials import policy
    settings = get_settings()
    if settings.embedding_provider != "bge" or query_expansion.ENABLED or rerank.ENABLED:
        raise RuntimeError("Free local BGE, disabled expansion/rerank required")
    paths = [p for p in (ROOT / "backend/app").rglob("*.py")] + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    manifest = {"kind": "new_free_replay_not_original_run", "material_policy":policy(), "annotation_sha256": digest(args.annotation),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,text=True).strip(),
        "source_hashes": hashes, "paid_requests": 0, "postgres_readonly": True,
        "qdrant_remote_only": True, "collection": settings.qdrant_collection,
        "process_only_overrides": {"RAG_RERANK": "0", "RAG_QUERY_EXPANSION": "0", "EMBEDDING_PROVIDER":"bge"},
        "embedding_model": settings.embedding_model,"returned_top_k":5,"tutor_hits":3,"budget":RAG_TUTOR_BUDGET,
        "freeze_before": frozen_before}
    write(args.out/"manifest.json",manifest)
    write(args.out/"annotation_snapshot.json",data)
    if args.archive:
        write(args.out/"archive_audit.json",audit_archive(args.archive))
    engine = get_engine()
    @event.listens_for(engine,"connect")
    def readonly(connection,_):
        with connection.cursor() as c:
            c.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        connection.commit()
    qd = QdrantClient(url=settings.qdrant_url,timeout=20,check_compatibility=False)
    qd.get_collection(settings.qdrant_collection)
    retriever.get_qdrant = lambda: qd
    captured_candidates = []
    production_retrieve = agent_tools.retrieve
    def capture_retrieve(*a, **kw):
        value = production_retrieve(*a, **kw)
        captured_candidates.append({"top_k":kw.get("top_k"),"hits":value})
        return value
    agent_tools.retrieve = capture_retrieve
    results = []
    for case in cases:
        start = time.perf_counter()
        try:
            captured_candidates.clear()
            hits = retrieve_knowledge(case["query"],top_k=5)
            delivered = delivered_rag_sources(hits,budget=RAG_TUTOR_BUDGET)
            gold = {str(bid): ref for s in case.get("acceptable_evidence_sets",[]) for ref,bid in zip(s["all_of"],s["all_of_block_ids"])}
            ids = set(gold) | {str(h["block_id"]) for h in hits if h.get("block_id")} | {str(b["id"]) for h in hits for b in h.get("context_blocks",[]) if b.get("id")}
            ids |= {str(bid) for call in captured_candidates for h in call["hits"]
                    for bid in [h.get("block_id"), *[b.get("id") for b in h.get("context_blocks",[])]] if bid}
            facts = {}
            with engine.connect() as conn:
                for bid in ids:
                    row=conn.execute(text("select id,doc_id,page,block_index,type,text,indexable,is_generated from document_blocks where id=:id"),{"id":bid}).mappings().first()
                    facts[bid]=dict(row) if row else None
            issues=identity_issues(hits,delivered,facts)
            sets=[]
            for s in case.get("acceptable_evidence_sets",[]):
                quotes=[quote_coverage(e,delivered) for e in s.get("excerpts",[])]
                sets.append({"all_of":s["all_of"],"declared_supports":s.get("supports"),"all_ids_delivered":all(any(str(d.get("id"))==str(b) for d in delivered) for b in s["all_of_block_ids"]),
                 "all_annotated_quotes_delivered": bool(quotes) and all(q["exact_quote_delivered"] for q in quotes),"quotes":quotes})
            blocks=[{"ref":ref,"id":bid,"category":classify(bid,hits,delivered,tutor_hits=3 if policy()=="legacy" else len(hits)),"direct_rank":next((i for i,h in enumerate(hits,1) if str(h.get("block_id"))==bid),None),
                     "pre_tool_return_direct_ranks":[{"call":ci,"rank":i} for ci,call in enumerate(captured_candidates,1) for i,h in enumerate(call["hits"],1) if str(h.get("block_id"))==bid]}
                    for bid,ref in gold.items()]
            row={"id":case["id"],"query":case["query"],"candidate_calls":list(captured_candidates),"hits":hits,"delivered":delivered,"source_facts":facts,
                 "required_answer_points":case.get("required_answer_points",[]),"blocks":blocks,"evidence_sets":sets,
                 "gold_direct_count":sum(b["direct_rank"] is not None for b in blocks),"total_direct_hits":len(hits),
                 "identity_issues":issues,"status":"unverifiable" if issues else "succeeded",
                 "semantic_sufficiency":"pending_manual_review","elapsed_sec":time.perf_counter()-start}
            write(args.out/(case["id"]+".json"),row)
            results.append({k:row[k] for k in ("id","gold_direct_count","total_direct_hits","elapsed_sec","status")}|{"category_counts":dict(Counter(b["category"] for b in blocks)),"id_presence_satisfied":None if issues else any(s["all_ids_delivered"] for s in sets),"quote_coverage_satisfied":None if issues else any(s["all_annotated_quotes_delivered"] for s in sets)})
            print(case["id"],results[-1],flush=True)
        except Exception as exc:
            failed={"id":case["id"],"status":"failed","error_type":type(exc).__name__,"elapsed_sec":time.perf_counter()-start,"id_presence_satisfied":None,"quote_coverage_satisfied":None}
            write(args.out/(case["id"]+".json"),failed)
            results.append(failed)
            print(case["id"],"failed",type(exc).__name__,flush=True)
    frozen_after = verify_freeze(args.freeze, args.annotation) if args.freeze else None
    write(args.out/"summary.json",{"rows":results,"semantic_sufficiency":"pending_manual_review","source_drift":[k for k,v in hashes.items() if digest(ROOT/k)!=v],"annotation_unchanged":digest(args.annotation)==manifest["annotation_sha256"],"paid_requests":0,
                                   "freeze_after": frozen_after})
    write(args.out/"artifact_hashes.json",{p.name:digest(p) for p in args.out.iterdir() if p.is_file()})
    qd.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
