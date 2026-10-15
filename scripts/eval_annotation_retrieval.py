"""按标注跑**问题相关检索**评测（第二评测线）。工具版本 v2。

## 三层**分开记录**（用户 2026-10-10 要求；不许互相反推）
```
① 直接命中   ：命中列表里就有它 ⇒ **按稳定 `block_id` 判**（Qdrant point.id 透传）
② 扩展包含   ：只在上下文目录 `context_blocks` 里 ⇒ **只证明"经扩展进入"**
                ⚠️ **不得用它反推直接命中**
③ 实际投递   ：模型**真正收到**的片段（Tutor 侧 top-3 × 400 字预算）
                ⇒ 用**产线的同一个函数** `delivered_rag_sources()` 算，不自己重写一遍
```
## 四类自检（后两类分别记「不满足」「不可验证」，**不许混算**）
```
· direct_hit            直接命中
· context_only          仅扩展进入
· truncated_away        必要内容**被截断**（进了目录但没进投递窗口） ⇒ **不满足**
· identity_missing      命中**身份缺失**（拿不到 block_id）        ⇒ **不可验证**
```
## 结论口径（用户指定，不许升级）
```
即使结果为假，也只能写：
  **「本次检索投递材料未满足任何预先标注的充分证据集。」**
**不能**写"系统确实答不了" —— 因为
  · 本次**未验证生成回答**（只测了投递材料）
  · **不保证**可接受证据集合已穷尽
```
"""
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="按标注跑问题相关检索评测（v2）")
    ap.add_argument("annotation", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--tutor-hits", type=int, default=3,
                    help="历史兼容参数，不再控制投递；由产线材料策略决定")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.agent.graph import RAG_TUTOR_BUDGET, delivered_rag_sources  # noqa: PLC0415
    from app.agent.tools import retrieve_knowledge  # noqa: PLC0415

    ann = json.loads(args.annotation.read_text(encoding="utf-8"))
    query, sets = ann["query"], ann.get("acceptable_evidence_sets") or []
    hits = retrieve_knowledge(query, top_k=args.top_k)

    # ---- ① 直接命中：**只按 block_id** ----
    direct_ids, missing_identity = set(), []
    for h in hits:
        bid = h.get("block_id")
        if bid:
            direct_ids.add(str(bid))
        else:
            missing_identity.append({"source": h.get("source"), "page": h.get("page")})
    # ---- ② 扩展包含：只证明"经扩展进入" ----
    ctx_ids = {str(b.get("id")) for h in hits for b in (h.get("context_blocks") or [])
               if b.get("id")}
    # ---- ③ 实际投递：用产线函数算（top-3 × 400 字） ----
    delivered = delivered_rag_sources(hits, budget=RAG_TUTOR_BUDGET)
    delivered_ids = {str(d.get("id")) for d in delivered if d.get("id")}
    delivered_text = " ".join(str(d.get("text") or "") for d in delivered)

    results = []
    for s in sets:
        need = [str(x) for x in (s.get("all_of") or [])]
        # ⚠️ 标注里写的是 `doc:p316:b17` 这类 ref —— 需要映射到 block_id 才能比。
        ids = [str(x) for x in (s.get("all_of_block_ids") or [])]
        row = {"all_of": need, "all_of_block_ids": ids, "supports": s.get("supports")}
        if not ids:
            # **不可验证** ≠ 未命中
            row.update({"verdict": "identity_unverifiable",
                        "reason": "该证据集没有 block_id 映射 ⇒ 无法按稳定身份判定"})
        else:
            row["direct_hit"] = [i for i in ids if i in direct_ids]
            row["context_only"] = [i for i in ids if i in ctx_ids and i not in direct_ids]
            row["delivered"] = [i for i in ids if i in delivered_ids]
            row["truncated_away"] = [i for i in ids if i in ctx_ids and i not in delivered_ids]
            if len(row["delivered"]) == len(ids):
                row["verdict"] = "satisfied"
            elif row["truncated_away"]:
                row["verdict"] = "not_satisfied_content_truncated"      # **不满足**
            else:
                row["verdict"] = "not_satisfied_not_delivered"
        results.append(row)

    any_identity_gap = bool(missing_identity)
    out = {
        "tool_version": "v2（三层分开 + 四类自检）",
        "annotation_id": ann.get("id"), "annotation_version": ann.get("annotation_version"),
        "query": query, "top_k": args.top_k, "material_policy": __import__("app.services.rag_materials", fromlist=["policy"]).policy(),
        "tutor_budget_chars": RAG_TUTOR_BUDGET,
        "returned_hits": len(hits),
        "layer_1_direct_hits": len(direct_ids),
        "layer_2_context_only_ids": len(ctx_ids - direct_ids),
        "layer_3_delivered_fragments": len(delivered),
        "self_checks": {
            "identity_missing": missing_identity,                       # ⇒ 不可验证
            "identity_missing_is_not_a_miss": True,
            "context_blocks_not_used_to_infer_direct_hits": True,
        },
        "evidence_sets": results,
        "satisfied_any": any(r.get("verdict") == "satisfied" for r in results),
        "unverifiable_any": any(r.get("verdict") == "identity_unverifiable" for r in results),
        "conclusion_allowed": "本次检索投递材料未满足任何预先标注的充分证据集。",
        "conclusion_not_allowed": "系统确实答不了（未验证生成回答；不保证证据集已穷尽）",
        "line": "question_relevant_retrieval",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("tool_version", "returned_hits", "layer_1_direct_hits",
                       "layer_2_context_only_ids", "layer_3_delivered_fragments",
                       "satisfied_any", "unverifiable_any")},
                     ensure_ascii=False, indent=2))
    for r in results:
        print(f"  {str(r.get('supports')):16s} ⇒ {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
