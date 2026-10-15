"""Read-only live RAG audit; writes only a new audit output directory.

Default: all existing retrieval regression cases, without paid LLM requests.
--paid additionally measures the actual Tool/reranker and paired hint generation.
This is an engineering regression audit, not a student/teacher effectiveness study.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(case, hits):
    gold = case["gold"]
    exact = page = None
    for i, hit in enumerate(hits, 1):
        if hit["source"] == gold["doc_id"] and hit["page"] == gold["page"]:
            page = page or i
            if hit["text"].strip() == gold["text"].strip():
                exact = exact or i
    return {"exact_rank": exact, "page_rank": page}


def metrics(rows, field="ranks"):
    out = {"n": len(rows)}
    for kind in ("exact", "page"):
        values = [r[field][kind + "_rank"] for r in rows]
        for k in (1, 3, 5, 10):
            out[f"{kind}_hit_at_{k}"] = sum(v is not None and v <= k for v in values)
        out[kind + "_mrr_at_10"] = sum(1 / v if v else 0 for v in values) / max(1, len(rows))
    return out


def anchor_status(hit):
    # The production tutor consumes context[:400]. A missing 80-character prefix
    # proves that this anchor wasn't sent; it is not a semantic relevance score.
    anchor = (hit.get("text") or "")[:80]
    context = hit.get("context") or hit.get("text", "")
    return {"anchor": anchor, "in_context_600": anchor in context,
            "in_tutor_400": anchor in context[:400]}


HINT_CASES = [
    {"id": "H1", "problem": "质量为2 kg的物块从高1 m的粗糙斜面顶端由静止下滑，摩擦力做功为-4 J，g=10 m/s²。求底端速度。",
     "answer": "mgh=1/2mv²，机械能守恒。", "issue": "题目给出摩擦力做负功，不能直接使用机械能守恒。",
     "concepts": ["动能定理", "摩擦力做功"],
     "expected": "提示考虑摩擦力做功，不能认可机械能守恒，不直接给出最终速度。"},
    {"id": "H2", "problem": "质量为0.2 kg的小球从离地5 m处由静止释放，不计空气阻力，g=10 m/s²。求落地前速度大小。",
     "answer": "v=gh=50 m/s。", "issue": "将重力加速度与高度直接相乘作为速度，关系及量纲不对。",
     "concepts": ["自由落体", "动能定理"],
     "expected": "引导建立速度与下落高度的正确关系，不把gh当速度，不直接报最终速度。"},
    {"id": "H3", "problem": "光滑水平面上，质量为1 kg的小车以4 m/s撞上静止的3 kg小车，随后粘在一起。求共同速度。",
     "answer": "1/2×1×4²=1/2×(1+3)×v²，碰撞前后动能守恒。",
     "issue": "粘连碰撞不满足动能守恒，应核查系统水平方向动量守恒。",
     "concepts": ["完全非弹性碰撞", "动量守恒"],
     "expected": "区分动量与动能守恒条件，不认可粘连碰撞动能守恒，不直接给最终共同速度。"},
    {"id": "H4", "problem": "6 Ω与3 Ω电阻并联接在6 V理想电源两端。求总电流。",
     "answer": "R总=6+3=9 Ω，所以I=6/9 A。", "issue": "并联电阻误用了串联电阻相加关系。",
     "concepts": ["电阻并联", "欧姆定律"],
     "expected": "提示并联支路电压相同及正确等效关系，不认可总电阻9 Ω，不直接报最终总电流。"},
]


def _is_balance_error(exc: BaseException | str) -> bool:
    """是不是**账户余额/配额**类错误（而不是偶发超时）。

    ⚠️ 为什么要单独认它（核查报告的建议）：**余额不足是"必然继续失败"的**，
    而超时是"可能重试成功"。不区分的话，付费探测会**继续烧在一个注定失败的账户上**。
    """
    text = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
    low = text.lower()
    return ("402" in low or "insufficient" in low or "balance" in low
            or "quota" in low or "余额" in text or "payment required" in low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--rerank-sample", type=int, default=24)
    parser.add_argument("--hint-repeats", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Use the already-installed embedding assets, without downloads.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from app.config import get_settings
    from app.db.postgres import get_engine
    from sqlalchemy import event, text
    from qdrant_client import QdrantClient
    settings = get_settings()
    configured = {k: getattr(settings, k) for k in
                  ("embedding_provider", "embedding_model", "qdrant_collection",
                   "llm_provider", "llm_model", "tutor_model")}
    # Every PG connection created in this audit refuses writes, including helpers.
    engine = get_engine()
    @event.listens_for(engine, "connect")
    def read_only(dbapi_connection, _):
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        dbapi_connection.commit()
    qd = QdrantClient(url=settings.qdrant_url, timeout=15)
    collection = qd.get_collection(settings.qdrant_collection)  # fail, no local fallback
    sources = {str(p.relative_to(ROOT)): sha(p) for base in ("backend/app", "prompts")
               for p in (ROOT / base).rglob("*") if p.suffix in {".py", ".md"}}
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    golden_path = ROOT / "evaluation/datasets/retrieval_golden_cases.json"
    cases = json.loads(golden_path.read_text(encoding="utf-8"))["cases"]
    with engine.connect() as conn:
        inventory = [dict(r._mapping) for r in conn.execute(text(
            "select doc_id,count(*) blocks,count(distinct page) pages,"
            "count(*) filter(where indexable) indexable,"
            "count(*) filter(where is_generated) generated from document_blocks "
            "group by doc_id order by doc_id"))]
        runs = [dict(r._mapping) for r in conn.execute(text(
            "select doc_id,status,count(*) runs from document_import_runs "
            "group by doc_id,status order by doc_id,status"))]
        gold_presence = {}
        for case in cases:
            row = conn.execute(text("select doc_id,page,text,indexable,is_generated "
                                    "from document_blocks where id=:id"),
                               {"id": case["gold"]["block_id"]}).mappings().first()
            gold_presence[case["id"]] = {
                "pg_exists": bool(row), "pg_exact": bool(row and
                    row["doc_id"] == case["gold"]["doc_id"] and
                    row["page"] == case["gold"]["page"] and
                    row["text"].strip() == case["gold"]["text"].strip()),
                "indexable": row["indexable"] if row else None,
                "generated": row["is_generated"] if row else None}
    points = qd.retrieve(settings.qdrant_collection,
                         ids=[c["gold"]["block_id"] for c in cases],
                         with_payload=True, with_vectors=False)
    point_map = {str(p.id): p.payload for p in points}
    for case in cases:
        payload = point_map.get(case["gold"]["block_id"])
        gold_presence[case["id"]]["vector_exists"] = payload is not None
        gold_presence[case["id"]]["vector_exact"] = bool(payload and
            payload.get("source") == case["gold"]["doc_id"] and
            payload.get("page") == case["gold"]["page"] and
            payload.get("text", "").strip() == case["gold"]["text"].strip())
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_hashes": sources, "dataset_sha256": sha(golden_path),
                "configured": configured, "collection": collection.model_dump(mode="json"),
                "inventory": inventory, "import_runs": runs, "seed": 20261010,
                "packages": {p: importlib.metadata.version(p) for p in
                             ("sentence-transformers", "qdrant-client", "sqlalchemy", "openai")},
                "scope": "Existing AI-authored retrieval regression; not held-out quality evidence",
                "postgres_read_only": True, "qdrant_remote_only": True}
    write(out / "manifest.json", manifest)
    write(out / "gold_presence.json", gold_presence)
    from app.rag import retriever
    retriever.get_qdrant = lambda: qd  # remote binding only; ranking unchanged
    from app.rag.embedding import get_embedding_client
    start = time.perf_counter()
    get_embedding_client().embed(["教材检索审计预热"])
    warmup = time.perf_counter() - start
    print(f"inventory docs={len(inventory)}, vectors={collection.points_count}, gold={len(cases)}, embedding warmup={warmup:.2f}s", flush=True)
    baseline = []
    with (out / "baseline.jsonl").open("w", encoding="utf-8") as handle:
        for i, case in enumerate(cases, 1):
            start = time.perf_counter()
            try:
                # The actual Tool recalls max(top_k,10) candidates before rerank.
                hits = retriever.retrieve(case["query"], top_k=10)
                row = {"id": case["id"], "query": case["query"], "gold": case["gold"],
                       "status": "succeeded", "hits": hits, "ranks": ranks(case, hits),
                       "anchors": [anchor_status(h) for h in hits[:3]]}
            except Exception as exc:
                row = {"id": case["id"], "status": "failed", "error_type": type(exc).__name__,
                       "ranks": {"exact_rank": None, "page_rank": None}, "anchors": []}
            row["elapsed_sec"] = time.perf_counter() - start
            baseline.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if i % 25 == 0:
                print(f"baseline {i}/{len(cases)}", flush=True)
    summary = {"baseline": metrics(baseline), "embedding_warmup_sec": warmup,
               "baseline_failures": sum(r["status"] != "succeeded" for r in baseline),
               "latency_sec": {"median": statistics.median(r["elapsed_sec"] for r in baseline),
                               "max": max(r["elapsed_sec"] for r in baseline)},
               "gold_presence": dict(Counter(tuple(sorted(v.items())) for v in gold_presence.values())),
               "anchor_checks": {"n": sum(len(r["anchors"]) for r in baseline),
                   "absent_from_context_600": sum(not a["in_context_600"] for r in baseline for a in r["anchors"]),
                   "absent_from_tutor_400": sum(not a["in_tutor_400"] for r in baseline for a in r["anchors"])}}
    # JSON object keys must be strings.
    summary["gold_presence"] = [{"properties": dict(k), "count": v}
                                for k, v in summary["gold_presence"].items()]
    summary["per_doc"] = {doc: metrics([r for r in baseline if r.get("gold", {}).get("doc_id") == doc])
                          for doc in sorted({c["gold"]["doc_id"] for c in cases})}
    write(out / "summary.json", summary)
    if args.paid:
        from run_t4_batch import capture_requests
        from app.agent import tools
        from app.agent.graph import generate_hint, tutor_graph
        from app.models.student import StudentState
        from app.rag import rerank, query_expansion
        from app.services.llm import get_llm
        # User-authorized model; do not edit .env or other processes.
        for field in ("llm_model", "tutor_model", "analyze_step_model", "diagnosis_model", "step_matcher_model"):
            setattr(settings, field, "deepseek-flash")
        settings.llm_provider = "deepseek"
        flags = {"rerank": rerank.ENABLED, "query_expansion": query_expansion.ENABLED,
                 "effective_model": get_llm("tutor").model}
        write(out / "paid_configuration.json", flags)
        rng = random.Random(20261010)
        grouped = defaultdict(list)
        for case in cases:
            grouped[case["gold"]["doc_id"]].append(case)
        selected = []
        for doc in sorted(grouped):
            rng.shuffle(grouped[doc])
        while len(selected) < min(args.rerank_sample, len(cases)):
            for doc in sorted(grouped):
                if grouped[doc] and len(selected) < args.rerank_sample:
                    selected.append(grouped[doc].pop())
        write(out / "rerank_selection.json", [c["id"] for c in selected])
        baseline_by_id = {r["id"]: r for r in baseline}
        reranked, negatives, hints, mapping, graph_runs = [], [], [], [], []
        paid_aborted = ""
        with capture_requests() as calls:
            def finish_phase(offset, phase, identity):
                for call in calls[offset:]:
                    call.update(audit_phase=phase, audit_id=identity)
                write(out / "provider_requests.json", calls)
            for i, case in enumerate(selected, 1):
                offset, start = len(calls), time.perf_counter()
                try:
                    hits = tools.retrieve_knowledge(case["query"], top_k=5)
                    row = {"id": case["id"], "status": "succeeded", "hits": hits,
                           "ranks": ranks(case, hits), "baseline_ranks": baseline_by_id[case["id"]]["ranks"]}
                except Exception as exc:
                    row = {"id": case["id"], "status": "failed", "error_type": type(exc).__name__,
                           "ranks": {"exact_rank": None, "page_rank": None},
                           "baseline_ranks": baseline_by_id[case["id"]]["ranks"]}
                    # 🔴 **批次熔断**（核查报告建议）：余额不足 ⇒ **必然继续失败**，
                    #    再跑下去只是把预算烧在一个注定失败的账户上。
                    #    ⚠️ 与"偶发超时"分开：超时是**可能重试成功**的，不该熔断。
                    if _is_balance_error(exc):
                        row["paid_aborted"] = True
                        reranked.append(row)
                        finish_phase(offset, "rerank", case["id"])
                        write(out / "reranked.json", reranked)
                        paid_aborted = f"余额/配额错误，已停止其余付费探测：{type(exc).__name__}"
                        print(f"[熔断] {paid_aborted}", flush=True)
                        break
                row.update(elapsed_sec=time.perf_counter()-start, physical_calls=len(calls)-offset)
                reranked.append(row)
                finish_phase(offset, "rerank", case["id"])
                write(out / "reranked.json", reranked)
                print(f"rerank {i}/{len(selected)}", flush=True)
            if paid_aborted:
                # 熔断后**跳过**其余两段付费探测（无关查询与提示配对）——
                # 它们比重排更贵，而在余额不足时必然全失败。
                write(out / "negative_queries.json", negatives)
                summary["paid"] = {"aborted": paid_aborted, "physical_requests": len(calls)}
                write(out / "summary.json", summary)
                print(f"[熔断] 跳过负例与提示配对。已写 {len(reranked)} 条重排记录。", flush=True)
                return 0
            for query in ("请告诉我明天北京天气和穿衣建议", "法国巴黎酒店的入住取消政策是什么"):
                offset = len(calls)
                hits = tools.retrieve_knowledge(query, top_k=3)
                negatives.append({"query": query, "hits": hits})
                finish_phase(offset, "out_of_domain", query)
            write(out / "negative_queries.json", negatives)
            for case in HINT_CASES:
                state = StudentState(student_id="rag_audit_no_persistence", problem_id=case["id"],
                    problem_content=case["problem"], current_answer=case["answer"],
                    current_step=case["answer"], diagnosed_issue=case["issue"], weak_concepts=case["concepts"],
                    issue_type="concept", has_obvious_error=True, hint_level=1, request_hint=True)
                offset = len(calls)
                hits = tools.retrieve_knowledge(case["issue"], knowledge_points=case["concepts"], top_k=3)
                finish_phase(offset, "hint_retrieval", case["id"])
                for rep in range(1, args.hint_repeats+1):
                    pair_id = f"{case['id']}-R{rep}"
                    conditions = [True, False]
                    rng.shuffle(conditions)
                    pair = {"pair_id": pair_id, "problem": case["problem"], "answer": case["answer"],
                            "diagnosis": case["issue"], "expected": case["expected"], "outputs": {}}
                    for label, enabled in zip(("A", "B"), conditions):
                        offset, start = len(calls), time.perf_counter()
                        try:
                            result = generate_hint(state.model_copy(update={"retrieved_context": hits if enabled else []}))
                            row = {"pair_id": pair_id, "label": label, "rag_enabled": enabled,
                                   "result": result, "hits": hits if enabled else [], "status": "succeeded"}
                        except Exception as exc:
                            row = {"pair_id": pair_id, "label": label, "rag_enabled": enabled,
                                   "status": "failed", "error_type": type(exc).__name__, "result": {}}
                        row.update(elapsed_sec=time.perf_counter()-start, physical_calls=len(calls)-offset,
                                   provider_call_indices=list(range(offset, len(calls))))
                        hints.append(row)
                        pair["outputs"][label] = {"hint": row["result"].get("hint", ""),
                            "degraded": bool(row["result"].get("degraded_reasons")),
                            "safety_substitutions": len(row["result"].get("safety_substitutions", []))}
                        mapping.append({"pair_id": pair_id, "label": label, "rag_enabled": enabled})
                        finish_phase(offset, "paired_hint", pair_id+"-"+label)
                        write(out / "paired_hints.json", hints)
                    review_path = out / "blind_hint_packets.json"
                    packets = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else []
                    packets.append(pair)
                    write(review_path, packets)
                    write(out / "restricted_condition_mapping.json", mapping)
                    print(f"hint pair {pair_id}", flush=True)
            # Invoke the real graph directly, deliberately bypassing attempt persistence.
            for identity, answer in (("wrong", HINT_CASES[0]["answer"]),
                                     ("correct", "由动能定理20-4=v²，故v=4 m/s。"), ("blank", "")):
                offset, start = len(calls), time.perf_counter()
                try:
                    result = tutor_graph.invoke({"student_id": "rag_audit_no_persistence",
                        "problem_id": "H1", "problem_content": HINT_CASES[0]["problem"],
                        "current_answer": answer, "request_hint": True, "graph_mode": False})
                    row = {"id": identity, "status": "succeeded", "result": result}
                except Exception as exc:
                    row = {"id": identity, "status": "failed", "error_type": type(exc).__name__}
                row.update(elapsed_sec=time.perf_counter()-start, physical_calls=len(calls)-offset)
                graph_runs.append(row)
                finish_phase(offset, "real_graph", identity)
                write(out / "graph_runs.json", graph_runs)
                print(f"graph {identity}", flush=True)
        summary["reranked_sample"] = metrics(reranked)
        summary["reranked_sample_baseline"] = metrics(reranked, "baseline_ranks")
        summary["paid"] = {"physical_requests": len(calls),
            "failed_requests": sum(c["status"] != "succeeded" for c in calls),
            "network_retries": sum(c["network_attempt"] > 1 for c in calls),
            "structured_second_pass_requests": sum(c["structured_pass"] > 1 for c in calls),
            "tokens": dict(sum((Counter(c.get("tokens") or {}) for c in calls), Counter())),
            "by_phase": {phase: {"physical_requests": len(rows),
                "median_sec": statistics.median(c["elapsed_sec"] for c in rows),
                "max_sec": max(c["elapsed_sec"] for c in rows)}
                for phase in sorted({c["audit_phase"] for c in calls})
                for rows in [[c for c in calls if c["audit_phase"] == phase]]}}
    summary["source_drift"] = [p for p, digest in sources.items() if sha(ROOT / p) != digest]
    summary["collection_points_at_end"] = qd.get_collection(settings.qdrant_collection).points_count
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
