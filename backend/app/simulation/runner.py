"""多轮学生学习模拟器（P12，任务书第 P12 节）。

在成员 C 的真实 Learning Event 接入之前，用脚本化的"学生"驱动完整教学闭环，
用于：测试多轮 Agent、State 演化、Graph 路径稳定性、Hint Policy、回归测试，
以及比赛现场不依赖摄像头/视频模型的备用演示模式。

为什么走 HTTP（TestClient）而不是直接调函数：
模拟器应当验证**真实的接口层编排**（Attempt 加载/写回、卡顿分计算、落库、
Trace），而不是复制一份等价逻辑。TestClient 在进程内运行，不需要起 uvicorn。
"""
import time
from dataclasses import dataclass, field
from typing import Any

from app.simulation.cases import SimulationCase, SimulationStep


@dataclass
class StepResult:
    """一轮的完整可观测结果（对应任务书要求的逐 Step 输出）。"""

    index: int
    student_input: str
    note: str = ""
    matched_nodes: list[str] = field(default_factory=list)
    attempted_nodes: list[str] = field(default_factory=list)
    missing_reasoning: list[str] = field(default_factory=list)
    current_step: str = ""
    current_node: str = ""
    visited_nodes: list[str] = field(default_factory=list)
    issue: str = ""
    issue_type: str = ""
    stuck_score: float = 0.0
    hint_level: int = 0
    hint: str = ""
    finished: bool = False
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)


@dataclass
class SimulationReport:
    """一次模拟运行的汇总（对应任务书要求的最终汇总）。"""

    case_id: str
    title: str
    profile: str
    problem_id: str
    student_id: str
    attempt_id: str
    mode: str
    steps: list[StepResult] = field(default_factory=list)
    visited_path: list[str] = field(default_factory=list)
    mistakes: list[dict] = field(default_factory=list)
    hints: list[dict] = field(default_factory=list)
    weak_concepts: list[str] = field(default_factory=list)
    finished: bool = False
    status: str = ""
    summary_checks: list[tuple[str, bool, str]] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for step in self.steps if step.ok)

    @property
    def total(self) -> int:
        return len(self.steps)

    @property
    def failed_checks(self) -> list[str]:
        failed = [
            f"Step {s.index}.{name}: {detail}"
            for s in self.steps
            for name, ok, detail in s.checks
            if not ok
        ]
        failed += [f"汇总.{name}: {detail}" for name, ok, detail in self.summary_checks if not ok]
        return failed

    @property
    def ok(self) -> bool:
        return not self.failed_checks


# ---------------------------------------------------------------- 断言

def _subset(expected: list[str], actual: list[str]) -> tuple[bool, str]:
    missing = [item for item in expected if item not in actual]
    return (not missing, f"期望包含 {expected}，实际 {actual}")


def _check_step(step: SimulationStep, body: dict[str, Any]) -> list[tuple[str, bool, str]]:
    expected = step.expected
    match = body.get("step_match") or {}
    graphless = body.get("graphless") or {}
    matched = list(match.get("matched_nodes") or [])
    attempted = list(match.get("attempted_nodes") or [])
    # P23：graphless 没有 step_match，缺失推理在 graphless 块里。
    # 不兜这一下，无图题目的剧本就永远无法断言"缺失推理被指出了"。
    missing = list(match.get("missing_reasoning") or graphless.get("missing_reasoning") or [])
    checks: list[tuple[str, bool, str]] = []

    # ---- 节点归属断言：**必须显式选口径** ----
    #
    # 支持三个字段名，各自含义不同（见 app/agent/node_semantics.py）：
    #   · `completed_nodes` —— 正确建立的节点（= state.visited_nodes 口径）
    #   · `touched_nodes`   —— 碰到就算（对应上 ∪ 尝试过）
    #   · `nodes`           —— **已废弃的别名，等价于 `touched_nodes`**
    #
    # 为什么 `nodes` 映射到 "touched" 而不是 "matched"：
    # 现有 6 个用例是按"碰到了就算"写的（例如"学生说出了 S5 的结论但未论证"
    # 也期望 `[S5]`）。此前断言用的是 `matched`，比用例的意图更严，
    # 于是出现"教学语义正确但断言失败"。**改断言口径去对齐用例意图，
    # 而不是放宽匹配器**——匹配器"只承认被论证成立的步骤"是教学上正确的。
    from app.agent.node_semantics import completed_nodes, touched_nodes

    match_obj = match if match else {}
    if getattr(expected, "completed_nodes", None):
        got = completed_nodes(match_obj)
        checks.append(("completed_nodes",
                       *_subset(expected.completed_nodes, got)))
    if getattr(expected, "touched_nodes", None):
        got = touched_nodes(match_obj)
        checks.append(("touched_nodes", *_subset(expected.touched_nodes, got)))
    if expected.nodes:
        got = touched_nodes(match_obj)
        checks.append(("nodes", *_subset(expected.nodes, got)))
    if expected.attempted:
        checks.append(("attempted", *_subset(expected.attempted, attempted)))
    if expected.missing_min:
        checks.append(
            (
                "missing_reasoning",
                len(missing) >= expected.missing_min,
                f"期望至少 {expected.missing_min} 条，实际 {len(missing)} 条",
            )
        )
    if expected.off_graph:
        checks.append(("off_graph", bool(match.get("off_graph")), f"实际 {match.get('off_graph')}"))

    detected_error = body.get("issue_type") not in ("", "none", None)
    checks.append(
        ("error", detected_error == expected.error, f"期望 error={expected.error}，实际 {detected_error}")
    )
    checks.append(
        ("finished", bool(body.get("finished")) == expected.complete,
         f"期望 finished={expected.complete}，实际 {body.get('finished')}")
    )

    level = int(body.get("hint_level", 0))
    if expected.hint_level is not None:
        checks.append(("hint_level", level == expected.hint_level, f"期望 L{expected.hint_level}，实际 L{level}"))
    if expected.hint_min_level is not None:
        checks.append(
            ("hint_min_level", level >= expected.hint_min_level,
             f"期望 ≥L{expected.hint_min_level}，实际 L{level}")
        )
    if expected.hint_max_level is not None:
        checks.append(
            ("hint_max_level", level <= expected.hint_max_level,
             f"期望 ≤L{expected.hint_max_level}，实际 L{level}")
        )
    return checks


def _check_summary(case: SimulationCase, detail: dict, report: SimulationReport) -> list[tuple[bool, str, str]]:
    expected = case.expected_summary
    checks: list[tuple[bool, str, str]] = []

    if "finished" in expected:
        want = bool(expected["finished"])
        got = bool(detail.get("final_correct")) or report.finished
        checks.append(("finished", got == want, f"期望 {want}，实际 {got}"))
    if "visited_path" in expected:
        want = list(expected["visited_path"])
        checks.append(("visited_path", report.visited_path == want, f"期望 {want}，实际 {report.visited_path}"))
    if "visited_contains" in expected:
        ok, detail_text = _subset(list(expected["visited_contains"]), report.visited_path)
        checks.append(("visited_contains", ok, detail_text))
    if "visited_excludes" in expected:
        present = [node for node in expected["visited_excludes"] if node in report.visited_path]
        checks.append(
            ("visited_excludes", not present, f"不应出现但出现了 {present}（实际路径 {report.visited_path}）")
        )
    if "mistakes_min" in expected:
        want = int(expected["mistakes_min"])
        checks.append(("mistakes_min", len(report.mistakes) >= want, f"期望 ≥{want}，实际 {len(report.mistakes)}"))
    if "hints_min" in expected:
        want = int(expected["hints_min"])
        checks.append(("hints_min", len(report.hints) >= want, f"期望 ≥{want}，实际 {len(report.hints)}"))
    if "max_hint_level" in expected:
        want = int(expected["max_hint_level"])
        got = max([s.hint_level for s in report.steps] or [0])
        checks.append(("max_hint_level", got <= want, f"期望 ≤L{want}，实际最高 L{got}"))
    if "min_hint_level" in expected:
        want = int(expected["min_hint_level"])
        got = max([s.hint_level for s in report.steps] or [0])
        checks.append(("min_hint_level", got >= want, f"期望 ≥L{want}，实际最高 L{got}"))
    return checks


# ---------------------------------------------------------------- 运行

def run_case(
    case: SimulationCase,
    *,
    offline: bool = False,
    client: Any | None = None,
    quiet: bool = False,
    on_step: Any | None = None,
) -> SimulationReport:
    """跑完一类学生的完整剧本，返回逐轮结果与汇总。

    offline=True 时用脚本化 LLM（见 offline.py），完全离线、结果确定。
    """
    from app.main import app

    if client is None:
        from fastapi.testclient import TestClient

        client = TestClient(app)

    restore = None
    if offline:
        from app.simulation.offline import install_scripted_llm

        restore = install_scripted_llm(case)

    student_id = case.simulator_student_id
    report = SimulationReport(
        case_id=case.case_id,
        title=case.title,
        profile=case.profile,
        problem_id=case.problem_id,
        student_id=student_id,
        attempt_id="",
        mode="offline" if offline else "real",
    )
    started = time.time()
    try:
        # 每次都从一次全新作答开始，保证可重复（否则会续做上一轮遗留的 attempt）
        client.post("/attempt/reset", json={"student_id": student_id, "problem_id": case.problem_id})

        index = 0
        for step in case.steps:
            for _ in range(max(1, step.repeat)):
                index += 1
                body = client.post(
                    "/attempt/analyze",
                    json={
                        "student_id": student_id,
                        "problem_id": case.problem_id,
                        "answer": step.input,
                        "idle_time": step.idle_time,
                        "rewrite_count": step.rewrite_count,
                        "request_hint": step.request_hint,
                    },
                ).json()
                match = body.get("step_match") or {}
                result = StepResult(
                    index=index,
                    student_input=step.input,
                    note=step.note,
                    matched_nodes=list(match.get("matched_nodes") or []),
                    attempted_nodes=list(match.get("attempted_nodes") or []),
                    missing_reasoning=list(match.get("missing_reasoning") or []),
                    current_step=body.get("current_step", ""),
                    current_node=body.get("current_node", ""),
                    visited_nodes=list(body.get("visited_nodes") or []),
                    issue=body.get("issue", ""),
                    issue_type=body.get("issue_type", ""),
                    stuck_score=float(body.get("stuck_score") or 0.0),
                    hint_level=int(body.get("hint_level") or 0),
                    hint=body.get("hint", ""),
                    finished=bool(body.get("finished")),
                    checks=_check_step(step, body),
                )
                report.attempt_id = body.get("attempt_id", report.attempt_id)
                report.steps.append(result)
                if on_step is not None:
                    on_step(result)
                elif not quiet:
                    print_step(result)

        detail = _attempt_detail(client, report.attempt_id)
        report.visited_path = list(detail.get("visited_nodes") or [])
        report.mistakes = list(detail.get("mistakes") or [])
        report.hints = list(detail.get("hint_history") or [])
        report.weak_concepts = list(detail.get("weak_concepts") or [])
        report.status = detail.get("status", "")
        report.finished = bool(detail.get("final_correct")) or any(s.finished for s in report.steps)
        report.summary_checks = _check_summary(case, detail, report)
    finally:
        if restore is not None:
            restore()

    report.elapsed = round(time.time() - started, 1)
    return report


def _attempt_detail(client: Any, attempt_id: str) -> dict:
    """取作答汇总：优先走 GET /attempt/{id}，不可用时退回进程内 AttemptStore。

    为什么需要退路：模拟器要能在**没有 PostgreSQL** 的环境下跑（项目一直承诺
    "无 DB 也能跑"，CI 里也未必起 Docker）。此时接口返回 404/503，
    而状态其实在 AttemptStore 的内存降级存储里，直接读它即可。
    """
    if not attempt_id:
        return {}
    response = client.get(f"/attempt/{attempt_id}")
    if response.status_code == 200:
        data = response.json()
        if data.get("visited_nodes") is not None:
            return data

    from app.services import attempt_store as store_module

    record = store_module.get_attempt_store().load_attempt(attempt_id)
    if record is None:
        return {}
    return {
        "visited_nodes": record.visited_nodes,
        "mistakes": record.mistakes,
        "hint_history": record.hint_history,
        "weak_concepts": record.weak_concepts,
        "status": record.status,
        "final_correct": record.final_correct,
    }


# ---------------------------------------------------------------- 打印

def print_step(result: StepResult) -> None:
    """按任务书 P12 要求的字段逐轮打印。"""
    print(f"\n── Step {result.index} " + "─" * 46)
    print(f"Student Input : {result.student_input}")
    if result.note:
        print(f"剧本          : {result.note}")
    print(f"Matched Node  : {result.matched_nodes or '（无）'}"
          + (f"  尝试未完成: {result.attempted_nodes}" if result.attempted_nodes else ""))
    if result.missing_reasoning:
        print(f"缺失推理      : {result.missing_reasoning}")
    print(f"Diagnosis     : [{result.issue_type or 'none'}] {result.issue or '无明显错误'}")
    print(f"Stuck Score   : {result.stuck_score:.2f}")
    print(f"Hint Level    : L{result.hint_level}")
    print(f"Tutor Output  : {result.hint or '（本轮不提示）'}")
    print(f"Updated State : current={result.current_node or '-'} "
          f"visited={'→'.join(result.visited_nodes) or '-'} finished={result.finished}")
    if result.checks:
        marks = " ".join(f"{'✓' if ok else '✗'}{name}" for name, ok, _ in result.checks)
        print(f"核对          : {marks}")


def print_summary(report: SimulationReport) -> None:
    """按任务书 P12 要求的最终汇总打印。"""
    print("\n" + "=" * 62)
    print(f"模拟汇总 · {report.case_id}（{report.title}）  模式={report.mode}")
    print(f"学生画像: {report.profile}")
    print(f"attempt : {report.attempt_id}   状态={report.status}   用时={report.elapsed}s")
    print("-" * 62)
    print(f"Visited Path  : {' → '.join(report.visited_path) or '（无）'}")
    print(f"Mistakes      : {len(report.mistakes)} 条")
    for item in report.mistakes:
        print(f"    - [{item.get('node') or '-'}] {item.get('issue_type') or '-'}: {item.get('issue') or ''}")
    print(f"Hints         : {len(report.hints)} 条")
    for item in report.hints:
        print(f"    - L{item.get('hint_level')} @{item.get('node') or '-'}: {str(item.get('hint'))[:60]}")
    print(f"Weak Concepts : {report.weak_concepts or '（无）'}")
    print(f"Result        : {'Finished ✓' if report.finished else 'Failed ✗（未判定完成）'}")
    print("-" * 62)
    for name, ok, detail in report.summary_checks:
        print(f"  {'✓' if ok else '✗'} {name}: {detail}")
    print(f"逐轮核对: {report.passed}/{report.total} 通过")
    if report.mode == "offline":
        print(
            "⚠ 离线模式：脚本化 LLM 按用例 expected 回放，因此 nodes / attempted /\n"
            "  off_graph / error 这几类断言属**冒烟**性质，不代表模型准确率。\n"
            "  真实准确率请去掉 --offline 用真实模型运行。"
        )
    if report.failed_checks:
        print("未通过项：")
        for item in report.failed_checks:
            print(f"  ✗ {item}")
    print("=" * 62)
