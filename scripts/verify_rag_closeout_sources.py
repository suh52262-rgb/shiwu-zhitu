"""Read-only identity/quote checks, not semantic or retrieval quality scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "backend"))
    from sqlalchemy import event, text
    from qdrant_client import QdrantClient
    from app.config import get_settings
    from app.db.postgres import get_engine
    engine = get_engine()

    @event.listens_for(engine, "connect")
    def readonly(connection, _):
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        connection.commit()

    ids = {str(bid) for c in data["drafts"] for s in c["acceptable_evidence_sets"]
           for bid in s["all_of_block_ids"]}
    facts = {}
    with engine.connect() as connection:
        for bid in sorted(ids):
            row = connection.execute(text("select id,doc_id,page,block_index,type,text,indexable,is_generated "
                                          "from document_blocks where id=:id"), {"id": bid}).mappings().first()
            if row:
                facts[bid] = dict(row)
                facts[bid]["id"] = str(row["id"])
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=20, check_compatibility=False)
    vectors = {str(p.id): p.payload for p in client.retrieve(settings.qdrant_collection, ids=sorted(ids),
                                                            with_payload=True, with_vectors=False)}
    client.close()
    errors, quotes, references = [], 0, 0
    for case in data["drafts"]:
        for group in case["acceptable_evidence_sets"]:
            group_ids = set(group["all_of_block_ids"])
            for ref, bid in zip(group["all_of"], group["all_of_block_ids"]):
                references += 1
                fact, payload = facts.get(bid), vectors.get(bid)
                if not fact:
                    errors.append([case["id"], ref, "missing PostgreSQL identity"])
                    continue
                actual = f"{fact['doc_id']}:p{fact['page']}:b{fact['block_index']}"
                if actual != ref or not fact["indexable"] or fact["is_generated"] or fact["type"] == "figure_caption":
                    errors.append([case["id"], ref, "identity/indexability/type mismatch"])
                if not payload or any(payload.get(k) != fact[f] for k, f in
                                      (("source", "doc_id"), ("page", "page"), ("text", "text"))):
                    errors.append([case["id"], ref, "remote vector payload mismatch"])
            for item in group.get("excerpts", []):
                quotes += 1
                bid = item["block_id"]
                fact = facts.get(bid)
                if bid not in group_ids or not fact or not item.get("excerpt_verbatim") or item["excerpt_verbatim"] not in fact["text"]:
                    errors.append([case["id"], item["ref"], "quote missing or outside group"])
                elif item["ref"] != f"{fact['doc_id']}:p{fact['page']}:b{fact['block_index']}":
                    errors.append([case["id"], item["ref"], "quote reference belongs to another identity"])
                elif hashlib.sha256(fact["text"].encode("utf-8")).hexdigest() != item["sha256"]:
                    errors.append([case["id"], item["ref"], "source text hash mismatch"])
    result = {"annotation_sha256": hashlib.sha256(args.annotation.read_bytes()).hexdigest(),
              "references": references, "unique_blocks": len(ids), "quotes": quotes,
              "errors": errors, "identity_quotes_verified": not errors, "semantic_validation": False,
              "postgres_read_only": True, "vector_operation": "remote retrieve IDs only",
              "embedding_requests": 0, "paid_requests": 0}
    for name, value in (("verification.json", result), ("source_facts.json", facts)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
