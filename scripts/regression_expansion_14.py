"""其余 14 条：**「原查询×10」vs「扩展×10」** 机械筛查（只测量）。

## 口径
```
· **重排关闭**（RAG_RERANK=0），候选深度固定 10 ⇒ 只比"查询方式"这一个变量 ✓
· ⚠️ **这是机械筛查**（看金标块是否进候选/进投递），
  **不能替代"必需要点覆盖"的语义判定** ✗（用户口径：不得以金标名次为唯一标准）
· 扩展**会调模型** ⇒ 请求数如实计入，**非完全离线** ✓
· 目的是**找出退化案例**，不是得出"扩展更好"的结论 ✓
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, "backend")
from app.agent.graph import RAG_TUTOR_BUDGET, delivered_rag_sources  # noqa: E402
from app.rag import query_expansion as QE  # noqa: E402
from app.rag.retriever import retrieve as R  # noqa: E402

DEPTH = 10
TUTOR_HITS = 3


def load_all() -> list[dict]:
    """载入 15 条（8 recheck8 + 7 误概念），排除偏振那条（它已单独判过）。"""
    DS = pathlib.Path("evaluation/datasets")
    out: dict[str, dict] = {}

    def add(cases, src):
        for c in cases:
            if c["id"] not in out:
                out[c["id"]] = {**c, "_src": src}

    v3 = json.loads((DS / "annotation_recheck8_v3.json").read_text(encoding="utf-8"))
    add(v3["drafts"], "recheck8_v3")
    for f in sorted(DS.glob("annotation_recheck8_part*.json")):
        d = json.loads(f.read_text(encoding="utf-8")).get("drafts", [])
        add(d, f.name)
        # 缺字段用 part 补全
        for c in d:
            cur = out.get(c["id"])
            if not cur:
                continue
            for k in ("query", "required_answer_points", "acceptable_evidence_sets"):
                if not cur.get(k) and c.get(k):
                    cur[k] = c[k]
    mis = json.loads((DS / "annotation_miscon_merged_v1.json").read_text(encoding="utf-8"))
    add(mis["drafts"], "miscon")

    # ⚠️ **校验放在所有来源合并之后**（2026-10-11 修）：
    #    若在 add() 内逐源校验，增量文件里"query 待 part 补全"的条目会被误判为缺失 ✗
    #    （实测：recheck8_dc_01 在 v3 里无 query、在 part1 里有 ⇒ 过早校验会直接失败）
    missing = [cid for cid, c in out.items()
               if not (c.get("query") or (c.get("old") or {}).get("query"))]
    if missing:
        raise RuntimeError(f"**缺 query**：{missing} ⇒ 拒绝继续（禁止静默跳过）")
    for cid, c in out.items():
        c["query"] = c.get("query") or c["old"]["query"]

    skip = {"recheck24_gx_miscon2_combo"}          # 偏振：已单独判过
    rows = [c for cid, c in out.items() if cid not in skip]
    print(f"载入 {len(out)} 条，排除 {skip} ⇒ 待测 {len(rows)} 条")
    return rows


def gold_of(case: dict) -> list[str]:
    ids: list[str] = []
    for s in case.get("acceptable_evidence_sets") or []:
        ids.extend(str(x) for x in (s.get("all_of_block_ids") or []))
    return ids


def measure(query: str, gold: list[str]) -> dict:
    t0 = time.perf_counter()
    hits = R(query, top_k=DEPTH)
    dt = time.perf_counter() - t0
    order = [str(h.get("block_id")) for h in hits if h.get("block_id")]
    delivered = delivered_rag_sources(hits[:TUTOR_HITS], budget=RAG_TUTOR_BUDGET, selected_policy="legacy")
    did = {str(x.get("id")) for x in delivered if x.get("id")}
    return {"in_cand": sum(1 for g in gold if g in order),
            "in_deliv": sum(1 for g in gold if g in did),
            "n_gold": len(gold), "frags": len(delivered),
            "latency_sec": round(dt, 3)}


def main() -> int:
    rows = load_all()
    report, calls_total = [], 0
    for c in rows:
        gold = gold_of(c)
        base = measure(c["query"], gold)
        variants = QE.expand_query(c["query"]) if QE.ENABLED else []
        calls_total += 1 if QE.ENABLED else 0
        lists = [R(q, top_k=DEPTH) for q in [c["query"]] + list(variants)]
        merged = QE.merge_rrf(lists) if hasattr(QE, "merge_rrf") else (lists[0] if lists else [])
        order = [str(h.get("block_id")) for h in merged if h.get("block_id")]
        delivered = delivered_rag_sources(merged[:TUTOR_HITS], budget=RAG_TUTOR_BUDGET, selected_policy="legacy")
        did = {str(x.get("id")) for x in delivered if x.get("id")}
        exp = {"in_cand": sum(1 for g in gold if g in order),
               "in_deliv": sum(1 for g in gold if g in did),
               "n_gold": len(gold), "frags": len(delivered),
               "expanded_text": list(variants)}
        delta = (exp["in_cand"] - base["in_cand"], exp["in_deliv"] - base["in_deliv"])
        report.append({"id": c["id"], "src": c["_src"], "base": base, "expanded": exp,
                       "delta_cand": delta[0], "delta_deliv": delta[1]})
        flag = "⚠️ 退化" if (delta[0] < 0 or delta[1] < 0) else ("✅ 改善" if sum(delta) > 0 else "—")
        print(f"  {c['id']:32s} 候选 {base['in_cand']}→{exp['in_cand']} "
              f"投递 {base['in_deliv']}→{exp['in_deliv']}  {flag}")

    reg = [r for r in report if r["delta_cand"] < 0 or r["delta_deliv"] < 0]
    imp = [r for r in report if r["delta_cand"] > 0 or r["delta_deliv"] > 0]
    out = {
        "scope": "其余 14 条的**机械筛查**（原查询×10 vs 扩展×10），重排关闭",
        "caveat": "⚠️ **不能替代必需要点覆盖的语义判定**；目的只是找出退化案例 ✓",
        "model_requests_for_expansion": calls_total,
        "rows": report,
        "regressions": [r["id"] for r in reg],
        "improvements": [r["id"] for r in imp],
        "summary": {"n": len(report), "regressions": len(reg), "improvements": len(imp),
                    "neutral": len(report) - len(reg) - len(imp)},
    }
    dst = pathlib.Path("artifacts/regression_expansion_14_20261011.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n合计 {len(report)} 条：**退化 {len(reg)}** ｜ 改善 {len(imp)} ｜ 中性 {len(report)-len(reg)-len(imp)}")
    print(f"扩展调用模型 {calls_total} 次（如实计入，非完全离线）")
    if reg:
        print("退化条目：", [r["id"] for r in reg])
    print(f"→ {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
