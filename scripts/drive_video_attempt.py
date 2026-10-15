"""把一段视频的作答**驱动进教学链路**。

## 为什么是这个形状
实测确认：`AnalyzeRequest.answer` **就是纯文本**，所以视频快照
**不需要任何新接口** —— 缺的只是"**谁驱动、什么节奏、attempt 怎么串**"。

本脚本把这三件事做成参数，于是"视频 → 教学链路"从**方案**变成**可执行的工具**。

## 数据来源
优先用感知层已经存下的 `results.json`（`run_pipeline` 的产物）——
**离线可重跑、不重复花 VL 的钱**。也支持直接给视频（现场跑一遍感知层）。

## 三个设计决定（都可调）
| 决定 | 参数 | 默认 | 为什么这个默认 |
|---|---|---|---|
| **谁驱动** | 本脚本 | — | 离线批处理够验证；实时要等感知层提速 |
| **提交节奏** | `--every-sec` | **20** | 实测验证过的值（9 个快照跑通 demo_03） |
| **attempt 生命周期** | `--student-id` / `--attempt-id` / `--reset` | 新建 | 一次作答 = 一个 attempt，脚本串起来 |

## 用法
```bash
# 离线：用现成的感知层结果
python scripts/drive_video_attempt.py data/video_runs/full_03平板车/results.json \
    --problem-id demo_03 --student-id stu_01

# 换节奏 / 先重置
python scripts/drive_video_attempt.py <results.json> --problem-id demo_03 \
    --student-id stu_01 --every-sec 30 --reset
```
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import httpx  # noqa: E402


def load_snapshots(results_json: Path, every_sec: float, *, deduplicate: bool = True) -> list[dict]:
    """从感知层的 `results.json` 取快照。

    **复用 `merge_by_timestamp`** —— ROI 与整帧同一时刻要合并（ROI 文字优先），
    否则会把同一时刻读两遍。
    """
    from app.api.video import _snapshots_from

    data = json.loads(results_json.read_text(encoding="utf-8"))
    snapshots = _snapshots_from(data, every_sec)
    if deduplicate:
        from app.vision.teaching_snapshots import coalesce_snapshots
        snapshots, _ = coalesce_snapshots(snapshots)
    return snapshots



def drive(base: str, problem_id: str, student_id: str, snaps: list[dict],
          attempt_id: str = "", request_hint: bool = False,
          log=print) -> dict:
    """按顺序提交快照，返回整段轨迹。"""
    traj: list[dict] = []
    with httpx.Client(timeout=180) as c:
        for i, g in enumerate(snaps, 1):
            body = {"student_id": student_id, "problem_id": problem_id,
                    "answer": g["text"].strip(), "attempt_id": attempt_id,
                    "request_hint": request_hint}
            try:
                r = c.post(f"{base}/attempt/analyze", json=body)
            except Exception as exc:  # noqa: BLE001
                log(f"  [{i}] ✗ 请求失败：{type(exc).__name__}: {exc}")
                break
            if r.status_code != 200:
                log(f"  [{i}] ✗ HTTP {r.status_code}：{r.text[:80]}")
                break
            d = r.json()
            attempt_id = d.get("attempt_id") or attempt_id
            traj.append({"round": i, "ts": g["ts"], **d})
            log(f"  [{i:>2}] t={g['ts']:>7.1f}s  "
                f"node={str(d.get('current_node') or '—'):<8}"
                f"L{d.get('hint_level', 0)}  卡顿{d.get('stuck_score', 0):.2f}  "
                f"{(d.get('hint') or d.get('issue') or '')[:44]}")
    return {"attempt_id": attempt_id, "trajectory": traj}


def main() -> int:
    ap = argparse.ArgumentParser(description="把视频作答驱动进教学链路")
    ap.add_argument("results_json", help="感知层产出的 results.json")
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--every-sec", type=float, default=20.0,
                    help="提交节奏（秒）。实测 20 验证过；调小=更密")
    ap.add_argument("--attempt-id", default="", help="续做已有 attempt")
    ap.add_argument("--request-hint", action="store_true",
                    help="每个快照都当作主动请求提示")
    ap.add_argument("--reset", action="store_true",
                    help="先把该学生这道题的未结束 attempt 重置")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="", help="把轨迹写到这个 json")
    a = ap.parse_args()

    rj = Path(a.results_json)
    if not rj.is_file():
        print(f"  ✗ 找不到 {rj}")
        return 1
    snaps = load_snapshots(rj, a.every_sec, deduplicate=not a.request_hint)
    print(f"  快照 {len(snaps)} 个（每 {a.every_sec:g} 秒一个）"
          f"   题 {a.problem_id}   学生 {a.student_id}")
    if not snaps:
        print("  ✗ 没有可提交的快照")
        return 1

    if a.reset:
        try:
            httpx.post(f"{a.base}/attempt/reset", timeout=15, json={
                "student_id": a.student_id, "problem_id": a.problem_id})
            print("  ✓ 已重置该学生这道题的进行中 attempt")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 重置失败（继续）：{exc}")

    out = drive(a.base, a.problem_id, a.student_id, snaps,
                a.attempt_id, a.request_hint)
    t = out["trajectory"]
    print(f"\n  === 汇总 ===")
    print(f"    attempt_id = {out['attempt_id']}")
    print(f"    提交 {len(t)}/{len(snaps)} 轮")
    if t:
        off = sum(1 for x in t if x.get("round_off_graph"))
        done = sorted({n for x in t for n in (x.get("visited_nodes") or [])})
        weak = sorted({w for x in t for w in (x.get("weak_concepts") or [])})
        hint = [x.get("hint_level", 0) for x in t]
        print(f"    离图轮次 {off}/{len(t)}")
        print(f"    走过节点 {done}")
        print(f"    薄弱点   {weak}")
        print(f"    提示级别 {hint}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"    轨迹已存 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
