"""Read-only, non-paid retrieval checks; misses remain in the denominator.

Only accepts explicitly AI-reviewed development data. Does not relabel samples,
change production settings, invoke LLM reranking, or promote a development set.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("status") != "AI_REVIEWED_DEVELOPMENT":
        raise ValueError("Requires AI_REVIEWED_DEVELOPMENT, not an unreviewed draft")
    cases = dataset["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    from sqlalchemy import event, text
    from qdrant_client import QdrantClient
    from app.config import get_settings
    from app.db.postgres import get_engine
    from app.rag import retriever

    settings = get_settings()
    if settings.embedding_provider.strip().lower() != "bge":
        raise ValueError("This non-paid checker requires local BGE embeddings")
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [p for base in ("backend/app", "prompts") for p in (ROOT / base).rglob("*")
             if p.suffix in {".py", ".md"}] + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    manifest = {"dataset_sha256": sha(args.dataset), "source_hashes": hashes,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "postgres_read_only": True, "qdrant_remote_only": True,
        "embedding_provider": "bge", "embedding_model": settings.embedding_model,
        "collection": settings.qdrant_collection, "paid_model_requests": 0,
        "top_k": 10, "scope": "AI-reviewed development retrieval, not held-out or teaching evaluation"}
    # Save identities before any retrieval, so failures cannot change the sample set.
    write(args.out / "manifest.json", manifest)
    write(args.out / "dataset_snapshot.json", dataset)
    engine = get_engine()
    @event.listens_for(engine, "connect")
    def read_only(connection, _):
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        connection.commit()

    qd = QdrantClient(url=settings.qdrant_url, timeout=15, check_compatibility=False)
    qd.get_collection(settings.qdrant_collection)  # No local fallback.
    points = qd.retrieve(settings.qdrant_collection,
        ids=[c["gold"]["block_id"] for c in cases], with_payload=True, with_vectors=False)
    point_map = {str(p.id): p.payload for p in points}
    presence = {}
    with engine.connect() as conn:
        for case in cases:
            gold = case["gold"]
            row = conn.execute(text("select doc_id,page,text,indexable,is_generated from "
                "document_blocks where id=:id"), {"id": gold["block_id"]}).mappings().first()
            payload = point_map.get(gold["block_id"])
            presence[case["id"]] = {
                "pg_exact": bool(row and all(row[k] == gold[k] for k in ("doc_id", "page", "text"))),
                "indexable": row["indexable"] if row else None,
                "is_generated": row["is_generated"] if row else None,
                "vector_exact": bool(payload and payload.get("source") == gold["doc_id"]
                    and payload.get("page") == gold["page"] and payload.get("text") == gold["text"])}
    write(args.out / "gold_presence.json", presence)
    # Bind the production retriever to this checked remote client only.
    retriever.get_qdrant = lambda: qd
    results = []
    for case in cases:
        start = time.perf_counter()
        gold = case["gold"]
        try:
            hits = retriever.retrieve(case["query"], top_k=10)
            exact = next((i for i, h in enumerate(hits, 1) if h["source"] == gold["doc_id"]
                and h["page"] == gold["page"] and h["text"] == gold["text"]), None)
            row = {"id": case["id"], "status": "succeeded", "hits": hits, "exact_rank": exact}
        except Exception as exc:
            row = {"id": case["id"], "status": "failed", "error_type": type(exc).__name__, "exact_rank": None}
        row["elapsed_sec"] = time.perf_counter() - start
        results.append(row)
        print(case["id"], row["status"], "exact_rank=", row["exact_rank"], flush=True)
    write(args.out / "results.json", results)
    drift = [str(p.relative_to(ROOT)) for p in paths if sha(p) != hashes[str(p.relative_to(ROOT))]]
    summary = {"cases": len(cases), "case_status": dict(Counter(r["status"] for r in results)),
        "exact_hits": {str(k): sum(r["exact_rank"] is not None and r["exact_rank"] <= k for r in results)
                       for k in (1, 3, 5, 10)},
        "missed_ids": [r["id"] for r in results if r["exact_rank"] is None],
        "gold_presence_all_valid": all(p["pg_exact"] and p["vector_exact"] and p["indexable"] is not False
                                       and not p["is_generated"] for p in presence.values()),
        "dataset_unchanged": sha(args.dataset) == manifest["dataset_sha256"], "source_drift": drift,
        "paid_model_requests": 0, "semantic_quality_verified": False,
        "miss_policy": "Retain every frozen case; a retrieval miss is not grounds for invalidating its gold"}
    write(args.out / "summary.json", summary)
    write(args.out / "artifact_hashes.json", {str(p.relative_to(args.out)): sha(p)
        for p in sorted(args.out.rglob("*")) if p.is_file()})
    print(json.dumps(summary, ensure_ascii=False))
    qd.close()
    return 0 if summary["gold_presence_all_valid"] and not drift and summary["dataset_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
