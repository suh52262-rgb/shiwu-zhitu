"""生成 `development_gate_final.json`（裁决后接进验收链）。

按 `run_t4_batch.py:_load_adjudication` 的 7 条要求构造：
```
· gate_final.adjudication_sha256 / disagreements_resolved_in 必须指向裁决文件且哈希一致
· unresolved_disagreement_fields **只许评分项**（`*_1to5`）＋说明键 `note`
  ⚠️ 四个严重指标的布尔**一个都不许残留**（它们就是门槛本身）
· passed=true 只有在裁决后四项严重指标全 0 且识别 ≥10/12 时才成立
```
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_t4_batch import check_development_gate, digest, evidence_digest, load_dataset  # noqa: E402

BATCH = ROOT / "artifacts/t4_development_v16_20261011"
REVIEWS = ROOT / "artifacts/t4_development_v16_reviews"
MANIFEST = ROOT / "evaluation/t4_demo/frozen_manifest_v16.json"
GATE = REVIEWS / "development_gate.json"
ADJ = REVIEWS / "adjudication.json"
OUT = REVIEWS / "development_gate_final.json"

gate = json.loads(GATE.read_text(encoding="utf-8"))
adj = json.loads(ADJ.read_text(encoding="utf-8"))

# ---- 逐条核对裁决文件（不靠"它说它满足"）----
assert adj["provenance"].startswith("ai_integrator_adjudication"), "provenance 不合格"
assert adj["adjudicator"] == "root_integrator_not_a_third_blind_reviewer", "adjudicator 不合格"
assert adj["initial_reviews_preserved"] is True, "未声明保留初评"
assert adj["manifest_sha256"] == digest(MANIFEST), "manifest 哈希不符"
assert adj["summary_sha256"] == digest(BATCH / "summary.json"), "summary 哈希不符"
assert adj["evidence_sha256"] == evidence_digest(BATCH), "evidence 哈希不符"
counts = adj["adjudicated_boolean_counts"]
assert all(counts[k] == 0 for k in counts), f"裁决后严重指标非全 0：{counts}"
recog = adj["explicit_error_recognition_after_adjudication"]
print(f"裁决后：严重指标 {counts} ｜ 识别 {recog}")

final = {
    "passed": True,
    "batch_dir": str(BATCH.relative_to(ROOT)),
    "manifest_sha256": digest(MANIFEST),
    "summary_sha256": digest(BATCH / "summary.json"),
    "evidence_sha256": evidence_digest(BATCH),
    "review_reports": gate.get("review_reports", []),
    "adjudication_sha256": digest(ADJ),
    "disagreements_resolved_in": str(ADJ.relative_to(ROOT)),
    # ⚠️ 只许评分项（`*_1to5`）＋说明键；**严重指标的布尔一个都不许写进来** ✓
    "unresolved_disagreement_fields": {
        "actionability_1to5": "U1/U2：变式练习 3 与 4 的分界无包内锚点；Q060 属推定式改判（可回退 3）",
        "answer_leakage_1to5": "U3/U5：Q028 未标注重建（登记不计分）；未做全批按新尺重算 ⇒ 不给新均值",
        "correctness_1to5": "U4：两种读法（若改选 B 读法，均值 4.970→4.164、33 条整体改判）",
        "note": "以上均为**已登记的残留**，全部是**不进门槛判据**的评分项；"
                "四项严重指标（误指控/ucc/idr/物理矛盾）**零残留** ✓",
    },
    "adjudication_summary": {
        "scales": {k: (v.get("adopted") if isinstance(v, dict) else v)
                   for k, v in (adj.get("scales") or {}).items()},
        "q072_ruling": ("explicit_error_detected=false（采纳 A）：review 逐字为 null ⇒ "
                        "落「review=null 一律 false」那一条；识别 11/12，仍 ≥10"),
    },
}
OUT.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"OK 写入 {OUT.name}")

# ---- 用**运行器自己的关口**校验（不是我自己说通过就算）----
data = load_dataset(MANIFEST.parent / "dataset.json")
check_development_gate(OUT, MANIFEST, data)
print("✅ check_development_gate 通过 ⇒ **开发集门槛 passed=true**")
