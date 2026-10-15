"""recheck8（8 条已标注开发案例）**正式评测** —— 两条线分开报告。

用户口径（2026-10-10）：
```
· **指定来源找回**：指定块的**直接命中排名**。
· **问题相关检索**：**实际投递片段**是否满足**任一充分证据集**；
  区分 **直接命中 / 扩展进入 / 截断**，并**单列不可验证项**。
· 这是 **8 条已标注开发案例**的正式运行，**不代表 24 条探针已完成**。
· **失败案例照常保留，不据结果临时改标注。**
```
三层判定（沿用 v2 工具）：直接命中（按稳定 block_id）/ 仅扩展进入 / 实际投递（产线函数）。
"""
# Historical experiment: explicit legacy material policy, never the current default.
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--tutor-hits", type=int, default=3)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.agent.graph import RAG_TUTOR_BUDGET, delivered_rag_sources
    from app.agent.tools import retrieve_knowledge

    ann = json.loads(args.annotation.read_text(encoding="utf-8"))

    # ⚠️ v3 是**增量文件**：未改动的条目其 query 只在原始 part 文件里
    #    ⇒ 建一个 id → query 的映射（按 old → part1/2/3 顺序回退）
    qmap: dict = {}
    for f in sorted((Path(args.annotation).parent).glob("annotation_recheck8_part*.json")):
        for c in json.loads(f.read_text(encoding="utf-8")).get("drafts", []):
            if c.get("query"):
                qmap[c["id"]] = c["query"]
    for c in ann["drafts"]:
        if c.get("query"):
            qmap[c["id"]] = c["query"]
        elif (c.get("old") or {}).get("query"):
            qmap[c["id"]] = c["old"]["query"]

    # ⚠️ v3 是**增量文件**：未改动的条目不在 `drafts` 里
    #    ⇒ 把**同前缀**的 part 文件里的其余条目并进来
    #    ⚠️ 前缀须从**本标注文件名**推导 —— 否则会并入**别的批次**的 part 文件
    #       （实测踩过：跑误概念 7 条时把 recheck8 的 8 条也混进来 ⇒ 变成 15 条）
    stem = Path(args.annotation).stem
    prefix = stem.rsplit("_v", 1)[0] if "_v" in stem else stem
    parts = sorted(Path(args.annotation).parent.glob(f"{prefix}_part*.json"))
    merged = list(ann["drafts"])
    have = {c["id"] for c in merged}
    for f in parts:
        for c in json.loads(f.read_text(encoding="utf-8")).get("drafts", []):
            if c["id"] not in have:
                merged.append(c)
                have.add(c["id"])
            else:
                # ⚠️ **缺字段时用 part 文件补全**（2026-10-11 修，同 attribute_failures.py）：
                # 增量文件里若丢了 `query`，原逻辑会让残缺的那条赢 ⇒ 该条被静默跳过
                cur = next(x for x in merged if x["id"] == c["id"])
                for k in ("query", "required_answer_points", "acceptable_evidence_sets"):
                    if not cur.get(k) and c.get(k):
                        cur[k] = c[k]
                        print(f"  ⚠️ 用 part 文件补全 {c['id']} 的 {k}")
    ann["drafts"] = merged
    print(f"并入口 {[p.name for p in parts]} ⇒ 条目数 {len(merged)}")

    rows = []
    for case in ann["drafts"]:
        q = qmap.get(case["id"])
        if not q:
            print(f"⚠️ {case['id']} 找不到 query，跳过")
            continue
        hits = retrieve_knowledge(q, top_k=args.top_k)
        order = [str(h.get("block_id")) for h in hits if h.get("block_id")]
        direct = set(order)
        # ⚠️ **按命中名次分别记目录**：这样才能把两种情况分开 ——
        #    · 在**前 tutor_hits 条**的目录里但超出 400 字预算 ⇒ **真截断**
        #    · 只在**第 tutor_hits+1 条及以后**的目录里 ⇒ **名次不够**（不是截断）
        ctx_by_rank = {}
        for rank, h in enumerate(hits, 1):
            ctx_by_rank[rank] = {str(b.get("id")) for b in (h.get("context_blocks") or [])
                                  if b.get("id")}
        ctx = set().union(*ctx_by_rank.values()) if ctx_by_rank else set()
        delivered = delivered_rag_sources(hits[:args.tutor_hits], budget=RAG_TUTOR_BUDGET, selected_policy="legacy")
        deliv = {str(x.get("id")) for x in delivered if x.get("id")}
        top_ctx = set().union(*[ctx_by_rank[r] for r in ctx_by_rank if r <= args.tutor_hits]) \
            if ctx_by_rank else set()

        def classify(bid: str) -> str:
            """一个证据块在**投递链**上的落点（四类分开，不合并）。"""
            if bid in deliv:
                return "delivered"
            if bid in top_ctx:
                return "truncated_beyond_budget"     # 进了前 N 条目录，但被 400 字预算截掉
            ranks = [r for r in sorted(ctx_by_rank) if bid in ctx_by_rank[r]]
            if ranks:
                return f"ranked_out_below_top{args.tutor_hits}"   # 名次不够，**不是截断**
            return "not_retrieved"                   # 根本没被检索到

        # ---- 线 1：指定来源找回（指定块的直接命中排名）----
        line1 = []
        for s in case.get("acceptable_evidence_sets") or []:
            for bid in (s.get("all_of_block_ids") or []):
                r = order.index(bid) + 1 if bid in direct else None
                line1.append({"block_id": bid, "direct_rank": r,
                              "found_directly": r is not None})

        # ---- 线 2：问题相关检索（实际投递是否满足任一充分证据集）----
        line2 = []
        for s in case.get("acceptable_evidence_sets") or []:
            ids = [str(x) for x in (s.get("all_of_block_ids") or [])]
            d_hit = [i for i in ids if i in direct]
            dl = [i for i in ids if i in deliv]
            # ⚠️ 逐块记**四类落点**，不再把"名次不够"混进"截断"
            placement = {i: classify(i) for i in ids}
            cut = [i for i in ids if placement[i] == "truncated_beyond_budget"]
            ranked_out = [i for i in ids if placement[i].startswith("ranked_out")]
            if not ids:
                v = "unverifiable_no_block_id"
            elif len(dl) == len(ids):
                v = "satisfied"
            elif cut:
                v = "not_satisfied_content_truncated"          # 真截断
            elif ranked_out:
                v = "not_satisfied_ranked_below_tutor_cut"     # **名次不够（非截断）**
            else:
                v = "not_satisfied_not_retrieved"              # 根本没检索到
            line2.append({"all_of": s.get("all_of"), "verdict": v,
                          "direct_hit": d_hit, "delivered": dl,
                          "truncated_beyond_budget": cut, "ranked_out": ranked_out,
                          "placement": placement, "supports": s.get("supports")})

        rows.append({
            "id": case["id"],
            "line_1_specified_source_recovery": line1,
            "line_2_question_relevant_retrieval": line2,
            "delivered_fragments": len(delivered),
            "unverifiable": [x for x in line2 if x["verdict"] == "unverifiable_no_block_id"],
        })

    out = {
        "tool": "recheck8 formal eval（两条线分开）",
        "annotation_version": ann.get("version"),
        "cases": len(rows),
        "RAG_TUTOR_BUDGET": RAG_TUTOR_BUDGET,
        "rows": rows,
        "summary": {
            "line_1_specified_block_direct_hit": {
                "checked": sum(len(r["line_1_specified_source_recovery"]) for r in rows),
                "hit": sum(1 for r in rows for x in r["line_1_specified_source_recovery"]
                           if x["found_directly"]),
            },
            "line_2_satisfied_cases": sum(
                1 for r in rows
                if any(x["verdict"] == "satisfied" for x in r["line_2_question_relevant_retrieval"])),
            "line_2_verdict_counts": {},
            "context_truncated_cases": [
                r["id"] for r in rows
                if any(x["verdict"] == "not_satisfied_content_truncated"
                       for x in r["line_2_question_relevant_retrieval"])],
            "unverifiable_items": sum(len(r["unverifiable"]) for r in rows),
        },
        "scope_note": "**这是 8 条已标注开发案例的正式运行，不代表 24 条探针已完成。**",
        "policy_note": "**失败案例照常保留；不据结果临时改标注。**",
    }
    for r in rows:
        for x in r["line_2_question_relevant_retrieval"]:
            k = x["verdict"]
            out["summary"]["line_2_verdict_counts"][k] = \
                out["summary"]["line_2_verdict_counts"].get(k, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    for r in rows:
        v = [x["verdict"] for x in r["line_2_question_relevant_retrieval"]]
        print(f"  {r['id']:22s} 投递 {r['delivered_fragments']:2d} 片段  ⇒ {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
