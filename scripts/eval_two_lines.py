"""逐条跑两条评测线并汇总（批处理驱动，输入路径由参数给）。

`eval_annotation_retrieval.py` 是单条输入且 `--out` 写**文件** ⇒ 本脚本按条拆开跑、再汇总。
两条线：① 指定来源找回 ② 实际投递证据充分性。保留失败与不可验证项。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/Scripts/python.exe"
RUNNER = ROOT / "scripts/eval_annotation_retrieval.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    ann = args.annotation if args.annotation.is_absolute() else ROOT / args.annotation
    out = args.out if args.out.is_absolute() else ROOT / args.out
    data = json.loads(ann.read_text(encoding="utf-8"))
    drafts = data["drafts"]
    out.mkdir(parents=True, exist_ok=True)

    rows, fails = [], []
    with tempfile.TemporaryDirectory() as td:
        for d in drafts:
            one = {"id": d["id"], "query": d["query"],
                   "acceptable_evidence_sets": d.get("acceptable_evidence_sets") or []}
            p = pathlib.Path(td) / f"{d['id']}.json"
            p.write_text(json.dumps(one, ensure_ascii=False), encoding="utf-8")
            f = out / d["id"]
            r = subprocess.run([str(PY), str(RUNNER), str(p), "--out", str(f)],
                               capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0 or not f.is_file():
                fails.append({"id": d["id"], "err": (r.stderr or "")[-200:]})
                print(f"  FAIL {d['id']}")
                continue
            j = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"id": d["id"], "cell_key": d.get("cell_key"),
                         "line1_direct_hits": j.get("layer_1_direct_hits"),
                         "line1_layer1_ids": j.get("layer_1_direct_hits"),
                         "satisfied_any": j.get("satisfied_any"),
                         "unverifiable_any": j.get("unverifiable_any"),
                         "conclusion_allowed": j.get("conclusion_allowed")})
            print(f"  OK   {d['id']}")

    n = len(rows)
    agg = {
        "n_items": len(drafts), "n_ok": n, "n_fail": len(fails), "fails": fails,
        "line1_指定来源找回（直接命中≥1）": f"{sum(1 for r in rows if r['line1_direct_hits'])}/{n}",
        "line2_实际投递证据充分（满足任一充分集）": f"{sum(1 for r in rows if r['satisfied_any'])}/{n}",
        "不可验证项": f"{sum(1 for r in rows if r['unverifiable_any'])}/{n}",
        "允许下结论": f"{sum(1 for r in rows if r['conclusion_allowed'])}/{n}",
        "per_item": rows,
        "annotation_file": str(ann.relative_to(ROOT)),
        "note": ("⚠️ 补样评测：样本新增/修订 ⇒ **不与旧批次混算** ✓；"
                 "**保留失败与不可验证项** ✓；**未改任何默认配置** ✓。"
                 "若样本尚未冻结 ⇒ 本结果**不是**冻结版本的正式成绩 ✓。"),
    }
    (out / "summary.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    for k in ("line1_指定来源找回（直接命中≥1）", "line2_实际投递证据充分（满足任一充分集）",
              "不可验证项", "允许下结论"):
        print(f"  {k}: {agg[k]}")
    print("→", out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
