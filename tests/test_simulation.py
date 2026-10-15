"""多轮学生学习模拟器测试（P12）。

全部走 `--offline`（脚本化 LLM）：不调用真实模型，结果确定，可直接作为回归基线。
真实模型的准确率必须用 `scripts/run_simulation.py <case>` 手动跑——脚本化 LLM
按用例期望回放，节点断言在离线模式下必然成立，不代表模型能力。
"""
import pytest

from app.simulation import list_cases, load_case, run_case
from app.simulation.offline import ScriptedLLM

EXPECTED_CASES = [
    "spring_alternative_path",
    "spring_concept_error",
    "spring_good_student",
    "spring_missing_reasoning",
    "spring_stuck_student",
]

# 同一用例会被多条断言复用；离线结果确定，跑一次缓存即可（否则整个文件要跑 15 遍）
_REPORT_CACHE: dict[str, object] = {}


def offline_report(case_id: str):
    """跑一次某用例的离线模拟并缓存（结果与执行顺序无关）。"""
    if case_id not in _REPORT_CACHE:
        _REPORT_CACHE[case_id] = run_case(load_case(case_id), offline=True, quiet=True)
    return _REPORT_CACHE[case_id]


# ---------- 用例数据 ----------

def test_all_five_student_cases_exist():
    """任务书 P12 要求至少模拟 5 类学生。

    这里断言"P12 的五个都在"，而**不是**题库式的穷举相等——
    后续阶段会继续加剧本（P23 就加了 graphless_basic），
    穷举会让"加用例"这件事无端变成"改测试"。
    """
    available = list_cases()

    assert set(EXPECTED_CASES) <= set(available), f"P12 的五类学生剧本不齐：{available}"
    assert len(available) >= len(EXPECTED_CASES)


def test_graphless_case_has_no_node_expectations():
    """P23 的 graphless 剧本不该出现节点期望——无图题目没有节点可匹配。

    这条防止有人误把 phy_002 的期望复制过来，让一条"永远通过"的假用例混进来。
    """
    case = load_case("graphless_basic")

    assert case.problem_id == "phy_001"
    for step in case.steps:
        assert step.expected.nodes == [], f"步骤 {step.note} 不该期望节点"
        assert step.expected.attempted == []
    assert case.expected_summary.get("visited_path") == []


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_case_files_are_well_formed(case_id):
    case = load_case(case_id)
    assert case.case_id == case_id
    assert case.title and case.profile and case.notes
    assert case.problem_id == "phy_002"
    assert len(case.steps) >= 5
    for step in case.steps:
        assert step.input.strip()
        assert step.idle_time >= 0
        assert step.repeat >= 1


def test_unknown_case_reports_available_options():
    with pytest.raises(ValueError, match="可用用例"):
        load_case("no_such_case")


# ---------- 脚本化 LLM ----------

def test_scripted_llm_picks_current_step_not_stale_one():
    """回归：提示词里含历史作答，必须取**最后**出现的那一步，否则整条剧本错位。"""
    case = load_case("spring_good_student")
    llm = ScriptedLLM(case)
    first, second = case.steps[0].input, case.steps[1].input
    # 模拟第二轮提示词：当前输入 + 时间线里带着第一轮的输入
    prompt = f"【近期学习事件】t=6s submit: {first}\n\n【学生当前输入】\n{second}"

    assert llm._step_for(prompt) is case.steps[1]


# ---------- 端到端（离线） ----------

@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_case_passes_offline(case_id):
    """每个用例逐轮核对与汇总核对必须全通过。"""
    report = offline_report(case_id)

    assert report.failed_checks == [], f"{case_id} 未通过：{report.failed_checks}"
    assert report.total >= 5
    assert report.passed == report.total
    assert report.attempt_id  # 复用了 P11 的 Attempt 生命周期
    assert report.status == "finished"


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_case_finishes_the_problem(case_id):
    """五类学生最终都应把题做完（任务书 P12 的 Finished/Failed 汇总）。"""
    report = offline_report(case_id)
    assert report.finished is True
    assert report.visited_path[-1] == "S11"


def test_good_student_needs_no_hint():
    report = offline_report("spring_good_student")
    assert report.hints == []
    assert report.mistakes == []


def test_alternative_path_uses_harmonic_method():
    """Case E：走 S6b 简谐运动法，且**不应**出现能量法节点 S6a。"""
    report = offline_report("spring_alternative_path")
    assert "S6b" in report.visited_path
    assert "S6a" not in report.visited_path


def test_missing_reasoning_records_mistake_and_hint():
    """Case B：跳步被记入 mistakes 并给出一条提示。"""
    report = offline_report("spring_missing_reasoning")
    assert [item["node"] for item in report.mistakes] == ["S5"]
    assert len(report.hints) == 1
    assert report.hints[0]["hint_level"] == 1


def test_concept_error_is_located_and_recovered():
    """Case C：E2 与 E4 两处概念错误都被定位，且学生最终修正。"""
    report = offline_report("spring_concept_error")
    nodes = {item["node"] for item in report.mistakes}
    assert nodes == {"S5", "S10"}
    assert report.visited_path[-1] == "S11"


def test_stuck_student_escalates_to_full_explanation():
    """Case D：长时间停顿 + 反复改写 + 主动请求 → 提示一路升到 L5。"""
    report = offline_report("spring_stuck_student")
    levels = [step.hint_level for step in report.steps if step.hint_level > 0]
    assert levels[0] == 1                      # 首次卡顿给方向提示
    assert levels == sorted(levels)            # 只升不降
    assert max(levels) == 5                    # 最终给完整讲解
    assert any(step.stuck_score > 0 for step in report.steps)  # 卡顿分确实被触发


def test_stuck_score_reflects_idle_and_rewrite_signals():
    """卡顿分应随 idle/rewrite 信号出现（P4 图感知卡顿检测）。"""
    report = offline_report("spring_stuck_student")
    assert max(step.stuck_score for step in report.steps) >= 0.5
