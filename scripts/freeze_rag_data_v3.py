"""冻结 RAG 补样数据为 `rag_data_v3`（2026-10-11）。

## 登记要点
```
· 版本号：现有数据版本 v1（13 条）／v2（recheck8+24 · 15 条有效）⇒ 本次 **v3** ✓
· **不覆盖旧清单** ✓ ｜ 冻结：标注文件 + 其复核产物 + 两条评测线产物 + 运行器 ✓
· 交付口径：**实际有效条数 + 缺口清单** ✓（**不称"24 条已补齐"** ✗）
```
⚠️ 本次为**数据冻结**（不是 T4 的运行身份冻结 ✓）；评测在**未冻结前已跑**
（v2/v3/v4/v5 各一轮 ✓），**v5 为定稿版** ⇒ 以 v5 那一轮为正式成绩 ✓。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/datasets/rag_data_freeze_v3.json"
if OUT.exists():
    raise SystemExit(f"{OUT.name} 已存在 => 拒绝覆盖")

FILES = [
    "evaluation/datasets/annotation_recheck24_batch7_v5.json",
    "evaluation/datasets/annotation_recheck24_batch7_v4_reviewed.json",
    "evaluation/datasets/annotation_recheck24_batch7_v3_reviewed.json",
    "evaluation/datasets/annotation_recheck24_batch7_v2_reviewed.json",
    "artifacts/rag_eval_batch7_v5_20261011/summary.json",
    "scripts/eval_two_lines.py",
    "scripts/eval_annotation_retrieval.py",
    "scripts/evidence_coverage.py",
]
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
files = {p: sha(p) for p in FILES}
v5 = json.loads((ROOT / FILES[0]).read_text(encoding="utf-8"))
summary = json.loads((ROOT / FILES[4]).read_text(encoding="utf-8"))

manifest = {
    "schema_version": 1,
    "kind": "rag_annotation_data_freeze",
    "data_version": "v3",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    "annotation_file": FILES[0],
    "files": files,
    "n_items": len(v5["drafts"]),
    "cells_filled": sum(1 for c in v5.get("cell_ledger", {}).get("cells", []) if c.get("entries")),
    "line1_指定来源找回": summary["line1_指定来源找回（直接命中≥1）"],
    "line2_实际投递证据充分": summary["line2_实际投递证据充分（满足任一充分集）"],
    "不可验证项": summary["不可验证项"],
    "允许下结论": summary["允许下结论"],
    "delivery_naming": "实际有效条数 7 条 + 缺口清单（**不得称 24 条已补齐**）",
    "gap_cells": ["em|normal|single", "em|miscon|single",
                  "optics|normal|comb", "optics|miscon|single"],
    "review_rounds": 4,
    "self_corrections": {
        "write_collision": ("⚠️ 两个 agent 曾同时写 v5 路径 ✗；磁盘定稿为 b1791626 版"
                            "（456,658B／sha 50bc7f2d…／自校验 218/218 ✓）；"
                            "另一份未完成的孤儿文件已移除，**来历记录在此，不抹去** ✓"),
        "third_round_review_errors": ["p325:b32 从未进过证据组（第三轮复核半句有误，第四轮已撤）",
                                      "条4 v3 组内已有 b23 极限命题（“只收首句”与送审件不符）",
                                      "cell_ledger 有 entries 的格数为 4，非 5"],
    },
    "note": ("⚠️ 属**补样评测**：样本新增/修订 ⇒ **不与旧批次混算** ✓；"
             "**保留失败与不可验证项** ✓；**未改任何默认配置** ✓。"
             "业务缺口只登记、不改默认 ✓"),
}
OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
chk = json.loads(OUT.read_text(encoding="utf-8"))
print("OK 写入", OUT.name)
print(f"  data_version={chk['data_version']} ｜ items={chk['n_items']} ｜ 已填格={chk['cells_filled']}")
print(f"  git_head={chk['git_head'][:12]} ｜ dirty={chk['git_dirty']}")
print(f"  线1={chk['line1_指定来源找回']} ｜ 线2={chk['line2_实际投递证据充分']}"
      f" ｜ 不可验证={chk['不可验证项']}")
print(f"  冻结文件数={len(chk['files'])}")
