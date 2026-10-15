"""第 1 步：对已复核条目做**失败归因**（逐条、逐证据块）。

用户口径（2026-10-11）——四类须分开：
```
· 未召回        —— 该块**根本没进任何候选目录**
· 排名被排除    —— 召回到了，但**名次在前 tutor_hits 条之外**
· 扩展遗漏      —— 在**前 tutor_hits 条命中的目录里没有它**
· 必要内容截断  —— 进了前 N 条目录，但**落在 400 字预算之外**
```
⚠️ 并**逐条记录**：直接命中 → 扩展进入 → 最终投递 → （留给语义判定的）必需要点覆盖
⚠️ **不能读取金标来决定取哪些块**（本脚本只**判定**，不参与组装）。
"""
import json
import pathlib
import sys

sys.path.insert(0, "backend")
from app.agent.graph import RAG_TUTOR_BUDGET, delivered_rag_sources  # noqa: E402
from app.agent.tools import retrieve_knowledge  # noqa: E402
from evidence_coverage import classify_block  # noqa: E402

ANN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "evaluation/datasets/annotation_miscon_merged_v1.json")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                   "artifacts/failure_attribution_20261011.json")
TOP_K, TUTOR_HITS = 5, 3

ann = json.loads(ANN.read_text(encoding="utf-8"))

# ⚠️ 增量文件（如 `*_v3.json`）只含**被改动的**条目 ⇒ 把**同前缀**的 part 文件并回来。
#    前缀从**本文件名**推导 —— 否则会把别的批次的 part 文件也混进来（实测踩过 15 条）。
_stem = ANN.stem
_prefix = _stem.rsplit("_v", 1)[0] if "_v" in _stem else _stem
_parts = sorted(ANN.parent.glob(f"{_prefix}_part*.json"))
if _parts:
    _have = {c["id"]: c for c in ann["drafts"]}
    for _f in _parts:
        for _c in json.loads(_f.read_text(encoding="utf-8")).get("drafts", []):
            cur = _have.get(_c["id"])
            if cur is None:
                ann["drafts"].append(_c)
                _have[_c["id"]] = _c
            else:
                # ⚠️ **缺字段时用 part 文件补全**（2026-10-11 修）：
                # 增量文件里若有条目**丢了 `query`**，而 part 文件里有 ⇒ 必须补上。
                # 原逻辑「id 已存在就不并入」会让**残缺的那条赢** ⇒
                # 该条被静默跳过（实测：recheck8 只纳入 6 条，dc_01/gx_03 缺席 ✗）
                for _k in ("query", "required_answer_points", "acceptable_evidence_sets"):
                    if not cur.get(_k) and _c.get(_k):
                        cur[_k] = _c[_k]
                        print(f"  ⚠️ 用 part 文件补全 {_c['id']} 的 {_k}")
    print(f"并入口 {[p.name for p in _parts]} ⇒ 条目数 {len(ann['drafts'])}")

rows = []
for case in ann["drafts"]:
    q = case.get("query") or (case.get("old") or {}).get("query")
    if not q:
        continue
    hits = retrieve_knowledge(q, top_k=TOP_K)
    order = [str(h.get("block_id")) for h in hits if h.get("block_id")]
    ctx_by_rank = {i + 1: {str(b.get("id")) for b in (h.get("context_blocks") or []) if b.get("id")}
                   for i, h in enumerate(hits)}
    top_ctx = set().union(*[v for k, v in ctx_by_rank.items() if k <= TUTOR_HITS]) or set()
    all_ctx = set().union(*ctx_by_rank.values()) if ctx_by_rank else set()
    delivered = delivered_rag_sources(hits[:TUTOR_HITS], budget=RAG_TUTOR_BUDGET, selected_policy="legacy")
    deliv = {str(x.get("id")) for x in delivered if x.get("id")}

    blocks = []
    for s in case.get("acceptable_evidence_sets") or []:
        for ref, bid in zip(s.get("all_of") or [], s.get("all_of_block_ids") or []):
            # ⚠️ 用**经过测试的**纯函数判落点（`scripts/evidence_coverage.py`）——
            #    原先这段逻辑内联在这里、**测不到**，正是出过错的地方
            cat = classify_block(bid, delivered=deliv, top_ctx=top_ctx,
                                 ctx_by_rank=ctx_by_rank, order=order, tutor_hits=TUTOR_HITS)
            blocks.append({"ref": ref, "block_id": bid,
                           "direct_rank": (order.index(str(bid)) + 1) if str(bid) in order else None,
                           "category": cat})
    counts = {}
    for b in blocks:
        counts[b["category"]] = counts.get(b["category"], 0) + 1
    rows.append({
        "id": case["id"], "query": q,
        "evidence_blocks": len(blocks),
        "category_counts": counts,
        "blocks": blocks,
        "delivered_fragments": len(delivered),
        # ⚠️ 必需要点覆盖**留给语义判定**，此处只给机械事实
        "required_points_coverage": "PENDING_SEMANTIC_JUDGEMENT",
    })

doc = {
    "step": "1. 失败归因（四类分开）",
    "source": str(ANN),
    "top_k": TOP_K, "tutor_hits": TUTOR_HITS, "tutor_budget_chars": RAG_TUTOR_BUDGET,
    "note": "⚠️ 本脚本**只判定不组装**，且**不读取金标来决定取哪些块**。"
            "`necessary_content_truncated` 表示**该块**落在预算外，"
            "**不等于**该块的**必需要点**被截掉（后者须逐条语义判定）。",
    "rows": rows,
    "aggregate": {},
}
agg = {}
for r in rows:
    for k, v in r["category_counts"].items():
        agg[k] = agg.get(k, 0) + v
doc["aggregate"] = agg
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"条目 {len(rows)} ｜ 证据块合计 {sum(r['evidence_blocks'] for r in rows)}")
print("四类合计：", json.dumps(agg, ensure_ascii=False))
for r in rows:
    print(f"  {r['id']:30s} 块{r['evidence_blocks']:2d}  " +
          " ".join(f"{k}={v}" for k, v in sorted(r["category_counts"].items())))
print(f"\n→ {OUT}")
