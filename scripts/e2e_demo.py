"""端到端验收脚本：对应开发文档第十一节的 11 步验收目标。

前置条件：
1. 后端已启动：cd backend && uvicorn app.main:app --reload
2. .env 已配置 LLM API Key（会发起真实 LLM 调用）
3. RAG 索引已构建（scripts/build_rag.py）

运行：.venv/Scripts/python scripts/e2e_demo.py
"""
import sys
import uuid
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
STUDENT = f"stu_accept_{uuid.uuid4().hex[:6]}"

checks: list[tuple[str, bool, str]] = []


def check(step: str, ok: bool, note: str = "") -> None:
    checks.append((step, ok, note))
    mark = "✅" if ok else "❌"
    print(f"  {mark} {step}" + (f" — {note}" if note else ""))


def main() -> None:
    print(f"验收学生：{STUDENT}\n")

    # 1. 系统显示一道大学物理题
    print("[1] 系统显示一道大学物理题")
    problem = httpx.get(f"{BASE}/problems/demo", timeout=10).json()
    # **字段名**：API 返回的是 `statement`（Pydantic 模型字段名），
    # 而 `data/problems/*/problem.json` 文件里用的是 `content`。
    # 这里原本只查 `content`，于是**这个检查从来就不可能通过**
    # （两个字段都接受，两种来源都能兼容）。
    check("题目可获取",
          bool(problem.get("statement") or problem.get("content")),
          problem.get("title", ""))

    # 2-3. 学生提交第一步答案 → 后端记录事件
    print("\n[2-3] 提交错误作答 → 记录事件")
    wrong = "物体受重力，mgh = 1/2·mv²，机械能守恒，v = √(2gh)"
    r1 = httpx.post(f"{BASE}/attempt/analyze", timeout=180, json={
        "student_id": STUDENT, "problem_id": "phy_001",
        "answer": wrong, "idle_time": 30,
    }).json()
    check("作答被受理并返回 attempt_id", bool(r1.get("attempt_id")), r1.get("attempt_id", ""))

    # 4-5. Agent 判断步骤 / 发现潜在错误
    print("\n[4-5] Agent 判断步骤、发现错误")
    check("返回当前步骤", bool(r1.get("current_step")), r1.get("current_step", ""))
    check("发现明显错误", bool(r1.get("issue")), f"{r1.get('issue_type')}：{r1.get('issue', '')[:40]}")

    # 6. RAG 从教材中检索相关知识
    print("\n[6] RAG 教材检索")
    kb = httpx.get(f"{BASE}/knowledge/search",
                   params={"query": r1.get("issue") or "机械能守恒条件", "top_k": 2},
                   timeout=60).json()
    check("检索到教材依据", bool(kb.get("items")),
          f"{kb['items'][0]['chapter']} p{kb['items'][0]['page']}" if kb.get("items") else "知识库为空？先跑 build_rag.py")

    # 7. Tutor 给一级提示而不是答案
    print("\n[7] 一级提示（不给最终答案）")
    hint1 = r1.get("hint", "")
    check("返回一级提示", r1.get("hint_level") == 1 and bool(hint1), hint1[:60])
    gives_away = "1/2·mv² - mgh" in hint1 or "W_f = " in hint1
    check("未直接给出最终答案", not gives_away)

    # 8-9. 学生根据提示修改答案 → 系统继续分析
    print("\n[8-9] 根据提示修正 → 继续分析")
    r2 = httpx.post(f"{BASE}/attempt/analyze", timeout=180, json={
        "student_id": STUDENT, "problem_id": "phy_001",
        "answer": "斜面粗糙，摩擦力做负功，机械能不守恒。由动能定理 mgh + W_f = 1/2·mv²，"
                  "得 W_f = 1/2·mv² - mgh",
        "idle_time": 60,
    }).json()
    check("修正后被判定完成", r2.get("finished") is True, r2.get("hint", "")[:50])

    # 10. 完整 Learning Timeline 已落库
    print("\n[10] 完整 Learning Timeline")
    detail = httpx.get(f"{BASE}/attempt/{r2['attempt_id']}", timeout=15).json()
    check("事件已落库", len(detail.get("events", [])) >= 2,
          f"{len(detail['events'])} 条事件")
    # **期望 1 条而不是 2 条。**
    # 第 [8-9] 步提交的是**修正后的正确答案**，会被判定完成——
    # 完成了就不再给提示，所以整个流程**按设计只产生 1 条提示**。
    # 原来的 `>= 2` 是个错误假设（实测 9/11 时挂在这条上）。
    # 检查的本意是"提示历史被落库"，用 ≥1 + 层级字段存在来表达。
    hints = detail.get("hints", [])
    check("提示历史已落库",
          len(hints) >= 1 and all(h.get("hint_level") is not None for h in hints),
          f"{len(hints)} 条提示（L{hints[0]['hint_level']}→L{hints[-1]['hint_level']}）"
          if hints else "无提示")

    # 11. 输出简单学习总结
    print("\n[11] 学习总结")
    profile = httpx.get(f"{BASE}/students/{STUDENT}/profile", timeout=15).json()
    attempt = detail
    print(f"  学生 {STUDENT}：提交 {len(attempt['events'])} 次，"
          f"获得 {len(attempt['hints'])} 级提示，"
          f"结果 {'正确 ✅' if attempt.get('final_correct') else '未完成'}，"
          f"薄弱点：{'、'.join(profile.get('weak_concepts', [])) or '无'}")
    check("画像与记录可回查", len(profile.get("attempts", [])) == 1)

    # ---- 汇总 ----
    failed = [c for c in checks if not c[1]]
    print(f"\n{'=' * 46}")
    print(f"验收结果：{len(checks) - len(failed)}/{len(checks)} 项通过")
    if failed:
        for step, _, note in failed:
            print(f"  ❌ {step}: {note}")
        sys.exit(1)
    print("ACCEPTANCE_OK — 11 步最小教学闭环全部通过")


if __name__ == "__main__":
    main()
