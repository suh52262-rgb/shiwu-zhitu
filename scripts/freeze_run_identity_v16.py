"""冻结**当前运行身份**为 v16（T4，2026-10-11）。

## 与 `run_t4_batch.py --freeze` 的区别（不要混）
```
`--freeze`   冻结【样例 + 两份独立样本评审】=> 要求两份 verdict=accept 的评审
本脚本       冻结【当前运行身份】：源码 / 提示词 / 配置 / 依赖 /
             评测数据 / 运行器哈希 + 回归记录
=> `--freeze` 给"这批样例能不能用"背书；
   本冻结回答"这次 72 条重跑对应的是哪一版代码"
```

## 用户口径（2026-10-11）
```
· 查版本登记、使用新编号（现有到 v15 => 本次 v16）
· 冻结：源码、提示词、配置、依赖、评测数据及运行器哈希；保存 1798 项回归记录
· 切换默认提示词不改变 v1-v15 历史批次的身份 ——
  旧批次继续绑定原冻结清单，只是不能代表当前版本
· 不得覆盖旧清单
"""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_t4_batch as RTB  # noqa: E402

OUT = ROOT / "evaluation/t4_demo/frozen_manifest_v16.json"
DATASET = ROOT / "evaluation/t4_demo/dataset.json"

if OUT.exists():
    raise SystemExit(f"{OUT.name} 已存在 => 拒绝覆盖（never overwrite a frozen batch）")

versions = sorted(p.stem.replace("frozen_manifest_", "")
                  for p in (ROOT / "evaluation/t4_demo").glob("frozen_manifest_v*.json"))
print(f"现有冻结清单：{versions}")
if "v16" in versions:
    raise SystemExit("v16 已登记 => 换个编号")

manifest = {
    "schema_version": 1,
    "kind": "source_and_run_identity_freeze",
    "version": "v16",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    "dataset": str(DATASET.resolve().relative_to(ROOT)),
    "dataset_sha256": RTB.digest(DATASET),
    "files": RTB.source_files(DATASET),
    # ⚠️ `verify()` 会读 `review_files`（`{**files, **review_files}`）⇒ 必须有这个键 ✓
    #    本冻结是**运行身份**冻结，不含样本评审 ⇒ 空字典 + 说明 ✓
    #    （要冻结样例评审请用 `--freeze`，那条路要求两份 verdict=accept 的评审 ✓）
    "review_files": {},
    "review_files_note": ("空：本清单是**运行身份**冻结，不含样本评审。"
                          "样例评审冻结走 `run_t4_batch.py --freeze` ✓"),
    "configuration": RTB.model_configuration(),
    "dependencies": {d.metadata["Name"]: d.version
                     for d in importlib.metadata.distributions() if d.metadata["Name"]},
    "regression": {
        "command": '.venv\\Scripts\\python.exe -m pytest tests -q -m "not real_db and not local_asset"',
        "passed": 1798, "failed": 0, "deselected": 37,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    },
    "supersedes_as_current_run_identity": "v15",
    "historical_batches_unchanged": True,
    "note": ("切换默认提示词不改变 v1-v15 历史批次的身份 —— "
             "旧批次继续绑定原冻结清单，只是不能代表当前版本。"
             "本清单回答的是「这一次 72 条重跑对应哪一版代码」。"),
    "holdout_policy": "On first holdout failure retire this holdout as development evidence",
    # ⚠️ 运行器实际会用到的字段（2026-10-11 补：首跑因缺 `seed` 直接 KeyError ✗）
    "seed": 20261009,                      # 与既有各批一致 ⇒ 可比
    "cases": 36, "repetitions": 3, "planned_runs": 108,
    "origin": "ai_authored; independently AI-reviewed; no teacher/student research",
}

# holdout 身份：**照抄 `freeze()` 的算法**（不自己发明）
data = RTB.load_dataset(DATASET)
_holdout_sources = sorted(
    (p["statement"], RTB.digest(DATASET.parent / p["diagram"]) if p.get("diagram") else None)
    for p in data["problems"] if p["split"] == "holdout")
manifest["holdout_identity"] = hashlib.sha256(
    json.dumps(_holdout_sources, ensure_ascii=False).encode("utf-8")).hexdigest()
manifest["holdout_question_identities"] = [
    hashlib.sha256(json.dumps(p, ensure_ascii=False).encode("utf-8")).hexdigest()
    for p in _holdout_sources]

OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

chk = json.loads(OUT.read_text(encoding="utf-8"))
print(f"OK 写入 {OUT.name}")
print(f"   version={chk['version']} | files={len(chk['files'])} | deps={len(chk['dependencies'])}")
print(f"   git_head={chk['git_head'][:12]} | dirty={chk['git_dirty']}")
print(f"   回归记录：{chk['regression']['passed']} passed / {chk['regression']['failed']} failed")
print(f"   覆盖诊断提示词 v3：{any(k.replace(chr(92), '/') == 'prompts/diagnosis/v3.md' for k in chk['files'])}")
