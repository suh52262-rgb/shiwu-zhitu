"""**描述性汇总器**（不产生 `passed`）—— 用户 2026-10-11 裁定。

## 为什么与 `summarize_t4_development.py` 并存而不替代
```
`summarize_t4_development.py` **硬编码 72 条 + `split=="development"`**，
且会输出 `passed`（依赖开发集的预先冻结判据 ✓）。
本脚本**不判通过与否** ✓ —— 因为**留出集没有预先冻结的验收判据**，
而"看过结果再定阈值"只能算**事后判定**，不能成为预先冻结的留出验收 ✗（用户口径）。
```

## 输出什么 / 不输出什么
```
✅ 统计：数量、分母、可评率、四项严重指标、识别数、逐 flag 一致性、未决项
✅ `acceptance_status = "not_predefined"`（明确：本批**无**预先冻结的判据）
❌ **不产生 `passed`** ✗ ｜ **不写任何通过阈值** ✗
```
⚠️ 新聚合器**不必写入通过阈值** ✓；下一轮的阈值**在接触新留出题之前**冻结 ✓。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

SEVERE = ("severe_false_accusation", "unsupported_cognitive_claim",
          "invalid_displayed_reference", "serious_physics_contradiction")
SCORES = ("correctness_1to5", "targeting_1to5", "actionability_1to5", "answer_leakage_1to5")
DIRECTION = {"correctness_1to5": "higher_is_better", "targeting_1to5": "higher_is_better",
             "actionability_1to5": "higher_is_better", "answer_leakage_1to5": "higher_is_worse"}


def _load(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("initial_ratings_locked") is not True:
        raise ValueError(f"{p.name} 未锁定初评 ⇒ 拒绝汇总")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="描述性汇总（不判通过）")
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--ratings", type=Path, nargs=2, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a, b = (_load(p) for p in args.ratings)
    ai = {x["blind_item_id"]: x for x in a["items"]}
    bi = {x["blind_item_id"]: x for x in b["items"]}
    if set(ai) != set(bi):
        raise ValueError("两份评审的条目集合不一致 ⇒ 拒绝汇总")
    ids = sorted(ai)

    # 逐 flag 一致性（含两个布尔与可评性）
    flags = list(SEVERE) + ["material_has_definite_error", "explicit_error_detected", "ratable"]
    agree = {f: sum(1 for i in ids if ai[i].get(f) == bi[i].get(f)) for f in flags}
    # 评分差异
    diffs = collections.defaultdict(list)
    for i in ids:
        for s in SCORES:
            if ai[i].get(s) != bi[i].get(s):
                diffs[s].append({"blind_item_id": i, "a": ai[i].get(s), "b": bi[i].get(s)})

    def dist(rev):
        out = {}
        for s in SCORES:
            c = collections.Counter(x.get(s) for x in rev["items"] if x.get(s) is not None)
            out[s] = dict(sorted(c.items()))
        return out

    report = {
        "kind": "descriptive_summary_only",
        "acceptance_status": "not_predefined",
        "acceptance_note": ("⚠️ 本批**没有预先冻结的验收判据** ⇒ 本汇总**不判通过与否** ✓。"
                            "看过结果再定阈值只能算**事后判定**，"
                            "不能成为预先冻结的留出验收 ✗。"),
        "thresholds_written": False,
        "batch": str(args.batch),
        "n_items": len(ids),
        "ratable": {r["reviewer_id"]: sum(1 for x in r["items"] if x.get("ratable")) for r in (a, b)},
        "severe_metrics": {r["reviewer_id"]: {k: r["metrics"].get(k) for k in
                                              [s + "s" for s in SEVERE]}
                           for r in (a, b)},
        "explicit_error": {r["reviewer_id"]: {"total": r["metrics"].get("explicit_error_total"),
                                              "recognized": r["metrics"].get("explicit_error_recognized")}
                           for r in (a, b)},
        "flag_agreement": {k: f"{v}/{len(ids)}" for k, v in agree.items()},
        "score_distributions": {r["reviewer_id"]: dist(r) for r in (a, b)},
        "score_direction": DIRECTION,
        "score_disagreements": {k: {"count": len(v), "items": v[:20]} for k, v in diffs.items()},
        "unresolved_fields": sorted({k for k, v in diffs.items() if v}),
        "note": ("⚠️ 评分差异**不等于**评审错误，也**不在此裁决** —— "
                 "本脚本只描述；分歧的裁决是另一道工序 ✓"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"条目 {len(ids)} ｜ acceptance_status = {report['acceptance_status']}（**不判通过** ✓）")
    print("四项严重指标：", json.dumps(report["severe_metrics"], ensure_ascii=False))
    print("识别数：", json.dumps(report["explicit_error"], ensure_ascii=False))
    print("逐 flag 一致率：", json.dumps(report["flag_agreement"], ensure_ascii=False))
    print("评分差异条目数：", {k: v["count"] for k, v in report["score_disagreements"].items()})
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
