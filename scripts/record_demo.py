"""录制一段演示：跑真实模型，把每轮结果存成可回放的 JSON。

## 为什么要有这个脚本

P25 的回放必须是**真实跑出来的**，不能手写假数据——
否则"演示"就变成了"编演"。这个脚本跑一次真实模型（会花钱，但只跑一次），
把每轮的学生输入、诊断、提示、节点变化、Timeline 落盘。

之后现场回放**一次模型都不调**，断网也能跑。

用法：
    python scripts/record_demo.py spring_concept_error
    python scripts/record_demo.py --list
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

OUT_DIR = ROOT / "demo" / "recordings"


def main() -> int:
    ap = argparse.ArgumentParser(description="录制演示回放")
    ap.add_argument("case", nargs="?", help="simulation/ 下的用例名")
    ap.add_argument("--list", action="store_true", help="列出可用用例")
    ap.add_argument("--student", default="", help="学生 id（默认按用例名生成）")
    args = ap.parse_args()

    cases = sorted(p.stem for p in (ROOT / "simulation").glob("*.json"))
    if args.list or not args.case:
        print("  可用用例：")
        for c in cases:
            print(f"    {c}")
        return 0
    if args.case not in cases:
        print(f"  ✗ 没有用例 {args.case!r}；可用：{cases}")
        return 1

    spec = json.loads((ROOT / "simulation" / f"{args.case}.json").read_text(encoding="utf-8"))
    steps_in = spec.get("steps") or []
    problem_id = spec.get("problem_id", "")
    student_id = args.student or f"demo_{args.case}"

    print(f"  录制 {args.case}（{len(steps_in)} 轮，problem={problem_id}）")
    print("  ⚠ 这一步会**真实调用模型**，请确认 API Key 与预算。")

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    recorded: list[dict] = []
    attempt_id = ""
    for i, step in enumerate(steps_in, 1):
        payload = {
            "student_id": student_id,
            "problem_id": problem_id,
            "answer": step.get("input", ""),
            "idle_time": step.get("idle_time", 0),
            "request_hint": step.get("request_hint", False),
        }
        if attempt_id:
            payload["attempt_id"] = attempt_id
        r = client.post("/attempt/analyze", json=payload, timeout=300)
        if r.status_code != 200:
            print(f"    ✗ 第 {i} 轮 HTTP {r.status_code}：{r.text[:160]}")
            return 1
        body = r.json()
        attempt_id = body.get("attempt_id") or attempt_id
        recorded.append({
            "index": i,
            "student_input": step.get("input", ""),
            "note": step.get("note", ""),
            "diagnosis": {
                "issue_type": body.get("issue_type", ""),
                "hint": body.get("hint", ""),
                "hint_level": body.get("hint_level", 0),
            },
            "current_node": body.get("current_node", ""),
            "visited_nodes": list(body.get("visited_nodes") or []),
            "finished": bool(body.get("finished")),
            "timeline": list(body.get("events") or []),
        })
        print(f"    ✓ 第 {i} 轮：node={body.get('current_node', '-')} "
              f"L{body.get('hint_level', 0)} 事件 {len(body.get('events') or [])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.case}.json"
    out.write_text(json.dumps({
        "case_id": args.case,
        "title": spec.get("title", args.case),
        "problem_id": problem_id,
        "student_id": student_id,
        # **如实标明来源**：现场要能回答"这是真跑出来的吗"
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/record_demo.py（真实模型跑出的结果）",
        "steps": recorded,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✓ 已写入 {out.relative_to(ROOT)}（{len(recorded)} 轮）")
    print("    现场回放：POST /demo/playback/" + args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
