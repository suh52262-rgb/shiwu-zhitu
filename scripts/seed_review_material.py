"""为**评测样本**写入合成文本复盘材料（契约 §2.5）。

## 它解决什么
T4 的样本是**文本作答**，而复盘材料此前只能由视频流程产出 ——
`scope` 被硬校验为 `video_run_snapshot`，没有任何受支持的"文本 → 复盘"入口。
这个脚本就是那个入口。

## 它**不**做什么
```
· 不生成复盘（那是 POST /video/review 的事，请跑真实端点）
· 不提供 scope 参数 —— 合成路径**不可能**写出 video_run_snapshot
· 不改题干、不改图：都取自真实题库与真实作答状态
· 不改写作答：`--answer` 传什么就存什么，逐字
```

## 用法
```powershell
# 1) 先用真实端点建一次作答（这一步是真实教学链路）
#    POST /attempt/analyze {student_id, problem_id, answer}
# 2) 再用本脚本把同一条作答写成"可复盘的材料"
.venv\\Scripts\\python.exe scripts/seed_review_material.py `
    --student-id stu_t4_001 --problem-id phy_002 --attempt-id <attempt_id> `
    --answer "P、Q 开始时一起运动，可以看作一个整体。"
# 3) 然后正常调复盘
#    POST /video/review {student_id, problem_id, attempt_id, run_id}
```
脚本会打印 `run_id` 与材料路径，并**明确提示这是合成材料**。
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.video import RUNS_DIR  # noqa: E402
from app.services import review_material  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--answer", action="append", required=True,
                        help="可重复；每次一行作答，**逐字存入**，不要润色")
    parser.add_argument("--run-id", help="不给就随机生成（32 位十六进制）")
    parser.add_argument("--ts", type=float, default=0.0, help="写成视频时刻，纯文本样本填 0")
    args = parser.parse_args()

    run_id = args.run_id or uuid.uuid4().hex
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        parser.error("--run-id 必须是 32 位十六进制（契约里的 run_id 形状）")

    rounds = [{"round": i, "ts": args.ts, "answer": text}
              for i, text in enumerate(args.answer, start=1)]
    out_dir = RUNS_DIR / run_id

    # ⚠️ **先校验、再建目录** —— 反过来的话，一次"作答不存在"的失败
    # 会在 data/video_runs/ 里留下一个**空目录**（第一版就是这么写的，实测留了一个）。
    try:
        payload = review_material.build_synthesized_payload(
            student_id=args.student_id, problem_id=args.problem_id,
            attempt_id=args.attempt_id, rounds=rounds)
    except ValueError as exc:
        print(f"[X] 写材料失败：{exc}")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    path = review_material.write_material(out_dir, payload)

    saved = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "run_id": run_id,
        "material": str(path),
        "scope": saved["input"]["scope"],
        "fingerprint": saved["fingerprint"],
        "rounds": len(saved["input"]["rounds"]),
        "timeline_length": saved["input"]["timeline_length"],
        "source_revisions": saved["input"]["source_revisions"],
    }, ensure_ascii=False, indent=2))
    print("\n⚠️ 这是**合成文本材料**（scope=synthesized_text_sample），"
          "不是真实视频运行；\n   复盘响应的 review_scope 会如实回带这个值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
