"""第 3 步：量**材料组装**是否存在实际损失（只测量，不改业务代码）。

## 用户口径（2026-10-11）
```
逐条记录：实际投递字符数、**去重后**字符数、**重复占比**、
          **排名之外相关且新增**的内容、以及在**相同预算**下能否补足**必需要点**
⚠️ **按要点判断收益，不能只看多放进几个金标块**
⚠️ **重复多本身不构成修改理由**；**排名之外的内容也不能为提高金标覆盖率全部塞进去**
```

## 硬校验（新增，防止"把子集当全量报告"）
```
· **缺 query** ⇒ **明确失败**
· **重复 ID 冲突** ⇒ **明确失败**
· **合并后条数不符**（预期 vs 实际）⇒ **明确失败**
⇒ **禁止静默跳过**（上一轮正是静默跳过 2 条，才把 6 条当成了全量 8 条 ✗）
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, "backend")
sys.path.insert(0, "scripts")
from app.agent.graph import RAG_TUTOR_BUDGET, delivered_rag_sources  # noqa: E402
from app.agent.tools import retrieve_knowledge  # noqa: E402


class EntryError(RuntimeError):
    """数据入口的硬失败 —— **不许降级为跳过**。"""


def load(annotation: pathlib.Path, expected: int | None = None) -> list[dict]:
    """读标注并**硬校验**：缺 query / 重复 ID / 条数不符 ⇒ 抛错。"""
    ann = json.loads(annotation.read_text(encoding="utf-8"))
    stem = annotation.stem
    prefix = stem.rsplit("_v", 1)[0] if "_v" in stem else stem
    parts = sorted(annotation.parent.glob(f"{prefix}_part*.json"))

    merged: list[dict] = list(ann["drafts"])
    seen: dict[str, dict] = {}
    for c in merged:
        if c["id"] in seen:
            raise EntryError(f"**重复 ID**：{c['id']}（在同一文件里出现两次）")
        seen[c["id"]] = c
    for f in parts:
        for c in json.loads(f.read_text(encoding="utf-8")).get("drafts", []):
            cur = seen.get(c["id"])
            if cur is None:
                merged.append(c)
                seen[c["id"]] = c
            else:
                for k in ("query", "required_answer_points", "acceptable_evidence_sets"):
                    if not cur.get(k) and c.get(k):
                        cur[k] = c[k]

    # ⚠️ **缺 query ⇒ 明确失败**（不许静默跳过 —— 上一轮就是这么丢掉 2 条的）
    missing = [c["id"] for c in merged if not (c.get("query") or (c.get("old") or {}).get("query"))]
    if missing:
        raise EntryError(f"**缺 query**：{missing} ⇒ 拒绝继续（禁止静默跳过）")
    if expected is not None and len(merged) != expected:
        raise EntryError(f"**合并后条数不符**：预期 {expected}，实际 {len(merged)} ⇒ 拒绝继续")
    print(f"  ✅ {annotation.name}：{len(merged)} 条（并入口 {[p.name for p in parts]}），query 齐全")
    return merged


def measure(case: dict, top_k: int = 5, tutor_hits: int = 3) -> dict:
    q = case.get("query") or (case.get("old") or {}).get("query")
    hits = retrieve_knowledge(q, top_k=top_k)
    delivered = delivered_rag_sources(hits[:tutor_hits], budget=RAG_TUTOR_BUDGET, selected_policy="legacy")

    frags = [str(x.get("text") or "") for x in delivered]
    total = sum(len(t) for t in frags)
    uniq_texts = []
    dup_chars = 0
    for t in frags:
        if t in uniq_texts:
            dup_chars += len(t)
        else:
            uniq_texts.append(t)
    dedup = sum(len(t) for t in uniq_texts)

    # 排名之外（>tutor_hits）目录里、且**不在投递材料内**的证据块 ⇒ 候补信息
    beyond = set()
    for rank, h in enumerate(hits, 1):
        if rank <= tutor_hits:
            continue
        for b in (h.get("context_blocks") or []):
            if b.get("id"):
                beyond.add(str(b["id"]))
    delivered_ids = {str(x.get("id")) for x in delivered if x.get("id")}
    beyond_new = beyond - delivered_ids

    return {
        "id": case["id"],
        "delivered_fragments": len(delivered),
        "delivered_chars": total,
        "deduped_chars": dedup,
        "duplicate_chars": dup_chars,
        "duplicate_ratio": round(dup_chars / total, 4) if total else 0.0,
        "budget_chars": RAG_TUTOR_BUDGET * tutor_hits,
        "budget_used_ratio": round(total / (RAG_TUTOR_BUDGET * tutor_hits), 4),
        "beyond_top3_new_blocks": len(beyond_new),
        # ⚠️ **按要点判断收益**是语义判断，此处只给机械事实
        "required_points_coverage": "PENDING_SEMANTIC_JUDGEMENT",
    }


def main() -> int:
    sets = [
        ("recheck8（完整 8 条）", pathlib.Path("evaluation/datasets/annotation_recheck8_v3.json"), 8),
        ("误概念（7 条）", pathlib.Path("evaluation/datasets/annotation_miscon_merged_v1.json"), 7),
    ]
    all_rows = []
    for name, p, exp in sets:
        print(f"=== {name} ===")
        cases = load(p, expected=exp)
        for c in cases:
            all_rows.append({"set": name, **measure(c)})
    out = {
        "step": "3. 组装实际损失测量（只测量）",
        "hard_validation": "缺 query / 重复 ID / 条数不符 ⇒ **明确失败**（禁止静默跳过）",
        "rows": all_rows,
        "note": "⚠️ **重复多本身不构成修改理由**；是否值得改组装，取决于"
                "**是否挤掉了有用材料**、以及**同预算下能否补足必需要点**（后者须逐条语义判定）",
    }
    dst = pathlib.Path("artifacts/assembly_loss_20261011.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'条目':32s} {'片段':>4s} {'投递字':>6s} {'去重后':>6s} {'重复占比':>8s} {'预算用':>7s} {'>3条新增块':>9s}")
    for r in all_rows:
        print(f"{r['id']:32s} {r['delivered_fragments']:4d} {r['delivered_chars']:6d} "
              f"{r['deduped_chars']:6d} {r['duplicate_ratio']:8.1%} {r['budget_used_ratio']:7.1%} "
              f"{r['beyond_top3_new_blocks']:9d}")
    print(f"\n→ {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
