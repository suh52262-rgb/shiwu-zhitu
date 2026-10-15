"""P-Q 弹簧题完整多轮 Demo（P10 验收，任务书第 15 节剧本）。

模拟一个真实的多轮解题过程，验证任务书第 18 节的核心验收目标：
学生连续多轮作答，系统持续维护解题状态、识别 Graph 节点、发现遗漏与错误、
结合 Learning Events 判断卡顿、按需调用 RAG、分级提示直至完成。

前置条件：
1. 后端已启动：cd backend && uvicorn app.main:app --reload
2. .env 已配置 LLM Key（真实调用）与 RAG 索引已构建

运行：.venv/Scripts/python scripts/pq_demo.py [backend_url]
"""
import sys
import uuid

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
STUDENT = f"stu_pqdemo_{uuid.uuid4().hex[:6]}"

checks: list[tuple[str, bool, str]] = []
rounds: list[dict] = []


def check(step: str, ok: bool, note: str = "") -> None:
    checks.append((step, ok, note))
    print(f"    {'✅' if ok else '❌'} {step}" + (f" — {note}" if note else ""))


def analyze(answer: str, request_hint: bool = False) -> dict:
    r = httpx.post(f"{BASE}/attempt/analyze", timeout=240, json={
        "student_id": STUDENT, "problem_id": "phy_002",
        "answer": answer, "idle_time": 30, "request_hint": request_hint,
    })
    r.raise_for_status()
    return r.json()


def run() -> None:
    print(f"P-Q 弹簧题多轮 Demo ｜ 学生 {STUDENT}\n")

    # Step 1：识别初始状态与共同运动阶段
    print("[Step 1] 学生：P、Q 开始时一起运动。")
    r1 = analyze("P、Q 开始时一起运动，可以看作一个整体。")
    rounds.append(r1)
    matched = r1["step_match"]["matched_nodes"]
    check("识别到初始/共同运动节点", bool({"S1", "S2"} & set(matched)), f"matched={matched}")

    # Step 2：跳步——直接给分离位置结论，缺 S3/S4 推理
    print("\n[Step 2] 学生：弹簧恢复原长时二者分离。")
    r2 = analyze("弹簧恢复原长时二者分离。")
    rounds.append(r2)
    sm = r2["step_match"]
    touched = set(sm["matched_nodes"]) | set(sm["attempted_nodes"])
    check("定位到分离位置节点（S5 附近）", bool({"S3", "S4", "S5", "S6"} & touched), f"matched={sm['matched_nodes']} attempted={sm['attempted_nodes']}")
    check("指出缺失的关键推理（跳步）", bool(sm["missing_reasoning"]), "；".join(sm["missing_reasoning"])[:60])
    check("Tutor 给出针对性提示", bool(r2["hint"]), r2["hint"][:60])

    # Step 3：补齐分离条件与受力分析推理链
    print("\n[Step 3] 学生：分离时 N=0，对 Q 有 N=m2·a，因此 a=0。")
    r3 = analyze("分离时接触力 N=0，对 Q 有 N = m2·a，因此 a = 0，即弹簧恢复原长时分离。")
    rounds.append(r3)
    check("分离条件推理链匹配（S3/S4）", bool({"S3", "S4"} & set(r3["step_match"]["matched_nodes"])),
          f"matched={r3['step_match']['matched_nodes']}")

    # Step 4：求分离速度（任选能量法或简谐运动法）
    print("\n[Step 4] 学生：由能量守恒求出 v。")
    r4 = analyze("由能量守恒 1/2·kd² = 1/2·(m1+m2)·v²，得 v = d·√(k/(m1+m2))。")
    rounds.append(r4)
    check("求出分离速度（S6a/S6b + S7）",
          bool({"S6a", "S6b", "S7"} & set(r4["step_match"]["matched_nodes"])),
          f"matched={r4['step_match']['matched_nodes']}")

    # Step 5：典型错误——认为分离后振幅仍是 d（E4）
    print("\n[Step 5] 学生：分离后 P 的振幅还是 d。（典型错误 E4）")
    r5 = analyze("分离后 P 的振幅还是 d。", request_hint=True)
    rounds.append(r5)
    sm5 = r5["step_match"]
    check("识别为错误作答（correct=False）", sm5["correct"] is False)
    check("定位到振幅相关节点", bool({"S8", "S9", "S10"} & set(sm5["attempted_nodes"])),
          f"attempted={sm5['attempted_nodes']}")
    check("Hint Policy 升级提示（L>1）", r5["hint_level"] > 1, f"L{r5['hint_level']}：{r5['hint'][:56]}")

    # Step 6：修正——系统边界变化 + 新初始条件 + 振幅能量
    print("\n[Step 6] 学生修正：分离后振子只剩 P，按新初始条件求振幅与能量。")
    r6 = analyze(
        "明白了：分离后 Q 匀速离开，新振子系统只有弹簧和 P。分离瞬间 x0=0、v0=v，"
        "由 1/2·kA² = 1/2·m1·v² 得 A = d·√(m1/(m1+m2))，"
        "能量 E = 1/2·m1·v² = m1kd²/(2(m1+m2))。"
    )
    rounds.append(r6)
    check("修正被接受（S8-S11）", bool({"S8", "S9", "S10", "S11"} & set(r6["step_match"]["matched_nodes"])),
          f"matched={r6['step_match']['matched_nodes']}")
    check("整题完成判定（终端节点）", r6["finished"] is True)

    # ---- 完整 Learning Timeline 回查 ----
    print("\n[Timeline] 完整学习过程（PostgreSQL 回查）")
    detail = httpx.get(f"{BASE}/attempt/{r6['attempt_id']}", timeout=15).json()
    events = detail.get("events", [])
    hints = detail.get("hints", [])
    check("全部作答事件已落库", len(events) >= 6, f"{len(events)} 条事件")
    check("提示历史已落库", len(hints) >= 2, f"{len(hints)} 条提示（L{hints[0]['hint_level']}→L{hints[-1]['hint_level']}）")

    visited = r6.get("visited_nodes", [])
    path = " → ".join(visited) if visited else "（空）"
    print(f"\n  最终解题路径：{path}")
    for i, (event, _) in enumerate(zip(events, rounds)):
        minute, sec = divmod(int(event["timestamp"]), 60)
        print(f"  {minute:02d}:{sec:02d}  {event['event_type']}: {event['content'][:44]}")

    # ---- 汇总 ----
    failed = [c for c in checks if not c[1]]
    print(f"\n{'=' * 50}")
    print(f"P-Q 弹簧题 Demo 验收：{len(checks) - len(failed)}/{len(checks)} 项通过")
    if failed:
        for step, _, note in failed:
            print(f"  ❌ {step}: {note}")
        sys.exit(1)
    print("PQ_DEMO_OK — 多轮过程感知教学闭环完整跑通")


if __name__ == "__main__":
    run()
