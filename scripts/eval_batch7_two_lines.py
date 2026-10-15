"""逐条跑两条评测线并汇总（batch7_v2）。

`eval_annotation_retrieval.py` 是**单条**输入 ⇒ 本脚本按条拆开跑、再汇总 ✓
两条线：① 指定来源找回（指定块 Hit@k）② 实际投递证据充分性 ✓
保留失败与不可验证项 ✓
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/Scripts/python.exe"
RUNNER = ROOT / "scripts/eval_annotation_retrieval.py"
ANN = ROOT / "evaluation/datasets/annotation_recheck24_batch7_v2.json"
OUT = ROOT / "artifacts/rag_eval_batch7_v2_20261011"

data = json.loads(ANN.read_text(encoding="utf-8"))
drafts = data["drafts"]
OUT.mkdir(parents=True, exist_ok=True)
rows, fails = [], []

with tempfile.TemporaryDirectory() as td:
    for d in drafts:
        one = {"id": d["id"], "query": d["query"],
               "acceptable_evidence_sets": d.get("acceptable_evidence_sets") or []}
        p = pathlib.Path(td) / f"{d['id']}.json"
        p.write_text(json.dumps(one, ensure_ascii=False), encoding="utf-8")
        o = OUT / d["id"]
        r = subprocess.run([str(PY), str(RUNNER), str(p), "--out", str(o)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            fails.append({"id": d["id"], "err": (r.stderr or "")[-200:]})
            print(f"  FAIL {d['id']}")
            continue
        j = json.loads((o / "result.json").read_text(encoding="utf-8")) \
            if (o / "result.json").is_file() else None
        rows.append({"id": d["id"], "cell": d.get("dimensions", {}).get(
            "evidence_completeness", ""), "result": j})
        print(f"  OK   {d['id']}")

summary = {
    "annotation_file": str(ANN.relative_to(ROOT)),
    "n_drafts": len(drafts), "n_ok": len(rows), "n_fail": len(fails),
    "fails": fails,
    "line1_指定来源找回": "见各条 result（指定块 Hit@k / 命中位次 / 不可验证单列）",
    "line2_实际投递充分性": "见各条 result（是否满足任一充分证据集；区分直接命中/扩展进入/截断）",
    "per_item": rows,
    "note": ("⚠️ 本轮为**补样评测**：样本系新增/修订，**不与旧批次混算** ✓；"
             "失败与不可验证项**保留** ✓；**未改任何默认配置** ✓"),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n∑ 跑完 {len(rows)}/{len(drafts)} 条（失败 {len(fails)}）")
print("→", OUT / "summary.json")
