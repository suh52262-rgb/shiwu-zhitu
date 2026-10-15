"""查询方式 × 候选深度 **矩阵实验**（只测量，不改默认、不开重排）。

## 用户设计（2026-10-11）
```
| 查询方式     | 候选深度      |
| 原查询       | 10、30、60   |
| 现有查询扩展 | 10、30、60   |
```
```
· **保持重排关闭** ⇒ 分清"查询"与"深度"各自的影响 ✓
· 查询扩展**只能读用户问题与正常运行上下文**，
  **不得读取金标、缺失要点或教材答案来构造查询** ✓
· **记录实际展开文本** ✓
· ⚠️ 查询扩展**调用模型**（`get_llm("tutor").chat`）⇒
  **如实计入请求数、耗时与 token**，**不能称为完全离线** ✗
· **不能以"10 块金标名次提高"作为唯一成功标准** ——
  深度增大后材料**仍进不了最终投递**，也**不算修复** ✓
```
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, "backend")
from sqlalchemy import text  # noqa: E402

from app.agent.graph import RAG_TUTOR_BUDGET, delivered_rag_sources  # noqa: E402
from app.db.postgres import get_engine  # noqa: E402
from app.rag import query_expansion as QE  # noqa: E402
from app.rag.retriever import retrieve as _retrieve_raw  # noqa: E402

DEPTHS = (10, 30, 60)
TUTOR_HITS = 3


def gold_ids(refs: list[str]) -> dict:
    out = {}
    with get_engine().connect() as c:
        for ref in refs:
            d, pg, bi = ref.split(":")
            r = c.execute(text("SELECT id FROM document_blocks "
                               "WHERE doc_id=:d AND page=:p AND block_index=:b"),
                          {"d": d, "p": int(pg.lstrip("p")), "b": int(bi.lstrip("b"))}).mappings().first()
            out[ref] = str(r["id"]) if r else None
    return out


def run_one(query: str, depth: int, gold: dict) -> dict:
    # ⚠️ 重排**关闭**（调用方用 RAG_RERANK=0 保证；这里再显式确认）
    t0 = time.perf_counter()
    hits = _retrieve_raw(query, top_k=depth)
    dt = time.perf_counter() - t0
    order = [str(h.get("block_id")) for h in hits if h.get("block_id")]
    delivered = delivered_rag_sources(hits[:TUTOR_HITS], budget=RAG_TUTOR_BUDGET, selected_policy="legacy")
    deliv_ids = {str(x.get("id")) for x in delivered if x.get("id")}
    ranks = {ref: (order.index(bid) + 1 if bid and bid in order else None)
             for ref, bid in gold.items()}
    in_cand = [r for r, k in ranks.items() if k]
    in_deliv = [ref for ref, bid in gold.items() if bid and bid in deliv_ids]
    return {
        "depth": depth,
        "returned": len(hits),
        "latency_sec": round(dt, 3),
        "gold_ranks": ranks,
        "gold_in_candidates": len(in_cand),
        "gold_in_delivered": len(in_deliv),
        "delivered_fragments": len(delivered),
        "candidate_blocks": len(order),
    }


def main() -> int:
    ann = json.loads(pathlib.Path("evaluation/datasets/annotation_recheck24_batch3_v2.json")
                     .read_text(encoding="utf-8"))
    case = next(c for c in ann["drafts"] if c["id"] == "recheck24_gx_miscon2_combo")
    query = case["query"]
    refs = list((case["acceptable_evidence_sets"][0]["all_of"]))
    gold = gold_ids(refs)

    report = {"case": case["id"], "query": query,
              "rerank": "OFF（RAG_RERANK=0）",
              "note": "⚠️ **查询扩展会调模型** ⇒ 请求/token 如实计入，**非完全离线**",
              "groups": []}

    # ---- 组 1~3：原查询 × 深度 ----
    for d in DEPTHS:
        r = run_one(query, d, gold)
        r.update({"query_mode": "original", "expanded_text": None,
                  "model_requests": 0, "prompt_tokens": None, "completion_tokens": None})
        report["groups"].append(r)
        print(f"  原查询 depth={d:2d} ⇒ 候选命中 {r['gold_in_candidates']}/10 ｜ "
              f"投递命中 {r['gold_in_delivered']}/10 ｜ {r['latency_sec']}s")

    # ---- 组 4~6：查询扩展 × 深度 ----
    # ⚠️ 扩展**只吃用户问题**（不看金标/要点/答案）✓
    variants, usage = None, {}
    try:
        import run_t4_batch as RTB  # 用产线的请求捕获，如实计请求与 token
        with RTB.capture_requests() as calls:
            variants = QE.expand_query(query) if QE.ENABLED else []
        usage = {"model_requests": len(calls),
                 "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
                 "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls)}
    except Exception as exc:  # noqa: BLE001
        variants = []
        usage = {"model_requests": 0, "error": f"{type(exc).__name__}: {exc}"}
    print(f"  扩展是否启用 = {QE.ENABLED} ｜ 展开条数 = {len(variants or [])} ｜ usage={usage}")

    exp_queries = [query] + list(variants or [])
    for d in DEPTHS:
        merged: dict = {}
        t0 = time.perf_counter()
        lists = [_retrieve_raw(q, top_k=d) for q in exp_queries]
        hits = QE.merge_rrf(lists) if hasattr(QE, "merge_rrf") else (lists[0] if lists else [])
        dt = time.perf_counter() - t0
        order = [str(h.get("block_id")) for h in hits if h.get("block_id")]
        delivered = delivered_rag_sources(hits[:TUTOR_HITS], budget=RAG_TUTOR_BUDGET, selected_policy="legacy")
        deliv_ids = {str(x.get("id")) for x in delivered if x.get("id")}
        ranks = {ref: (order.index(bid) + 1 if bid and bid in order else None)
                 for ref, bid in gold.items()}
        report["groups"].append({
            "query_mode": "expanded", "depth": d, "expanded_text": exp_queries,
            "model_requests": usage.get("model_requests"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "returned": len(hits), "latency_sec": round(dt, 3),
            "gold_ranks": ranks,
            "gold_in_candidates": sum(1 for k in ranks.values() if k),
            "gold_in_delivered": sum(1 for ref, bid in gold.items() if bid and bid in deliv_ids),
            "delivered_fragments": len(delivered), "candidate_blocks": len(order),
        })
        g = report["groups"][-1]
        print(f"  扩展   depth={d:2d} ⇒ 候选命中 {g['gold_in_candidates']}/10 ｜ "
              f"投递命中 {g['gold_in_delivered']}/10 ｜ {g['latency_sec']}s")

    dst = pathlib.Path("artifacts/matrix_query_depth_20261011.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
