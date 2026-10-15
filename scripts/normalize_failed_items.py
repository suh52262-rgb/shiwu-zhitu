"""把评分文件里"失败流程"的条目归一化为**不可评**（项目的硬不变式）。

⚠️ **不覆盖原件**：写新文件，并把被改动的字段原值留在 `_normalized_from` 里（可回溯）。
⚠️ **要重新算 metrics** —— 因为 `explicit_error_*` 的分母依赖可评性，
   不重算就会出现"自报 12、重算 11"的不一致。

为什么需要它：聚合器强制「Failed workflows must be unratable and excluded from means」，
而评审 B′ 主张"teaching 部分仍可评"、把两条 failed 条目打了分。
**规则是项目的，所以归一化；但改动必须留痕，不能悄悄改。**
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ("correctness_1to5", "targeting_1to5", "actionability_1to5", "answer_leakage_1to5")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--ratings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists():
        raise ValueError("输出已存在；永远不覆盖既有评分文件")
    data = json.loads(args.ratings.read_text(encoding="utf-8"))
    changed = []
    for item in data["items"]:
        packet = json.loads((args.batch / "review_packets" /
                             f"{item['blind_item_id']}.json").read_text(encoding="utf-8"))
        if packet.get("workflow_status") == "succeeded":
            continue
        if not item.get("ratable"):
            continue                      # 已经是不可评，无需动
        before = {k: item.get(k) for k in SCORES}
        item["_normalized_from"] = {"ratable": True, "scores": before,
                                    "reason": "workflow_status != succeeded；"
                                              "聚合器要求失败流程必须不可评（原值保留在此）"}
        item["ratable"] = False
        for k in SCORES:
            item[k] = None
        changed.append((item["blind_item_id"], before))

    # metrics 必须跟着重算，否则自报与重算不一致
    items = data["items"]
    data["metrics"] = {
        "severe_false_accusations": sum(i["severe_false_accusation"] for i in items),
        "unsupported_cognitive_claims": sum(i["unsupported_cognitive_claim"] for i in items),
        "invalid_displayed_references": sum(i["invalid_displayed_reference"] for i in items),
        "serious_physics_contradictions": sum(i["serious_physics_contradiction"] for i in items),
        "explicit_error_total": sum(i["material_has_definite_error"] for i in items if i["ratable"]),
        "explicit_error_recognized": sum(i["material_has_definite_error"] and i["explicit_error_detected"]
                                         for i in items if i["ratable"]),
    }
    data["normalization"] = {
        "policy": "workflow_status != succeeded ⇒ ratable=false（项目不变式）",
        "changed_items": [q for q, _ in changed],
        "note": "原值保留在各条目的 _normalized_from 里；metrics 已按新可评集合重算",
    }
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"归一化 {len(changed)} 条：")
    for qid, before in changed:
        print(f"  {qid}  原分数 {before} → null（原值已存在 _normalized_from）")
    print(f"重算后的 metrics：{json.dumps(data['metrics'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
