"""Step Matcher 测试（P2）：假 LLM，覆盖任务书第 5 节要求的六种场景。"""
import pytest

from app.agent.step_matcher import StepMatch, match_step
from app.services.problem_store import load_problem, load_problem_graph
from fakes import install_fake_llm


@pytest.fixture
def pq():
    problem = load_problem("phy_002")
    graph = load_problem_graph("phy_002")
    assert problem and graph
    return problem, graph


def _match(monkeypatch, step_match, visited=None, timeline=None, text="学生输入") -> StepMatch:
    install_fake_llm(monkeypatch, step_match=step_match)
    problem, graph = load_problem("phy_002"), load_problem_graph("phy_002")
    from app.models.student import StudentState

    state = StudentState(visited_nodes=visited or [])
    return match_step(problem, graph, state, text, timeline=timeline)


def test_does_not_mutate_the_object_returned_by_llm(monkeypatch, pq):
    """回归：后校验曾就地改写 LLM 返回的对象。

    一旦某个 LLM 实现复用同一个对象（测试夹具就是如此），那份对象会被永久改坏，
    后续调用全部拿到空匹配。这里固定住"入参不被修改、返回的是校验后的副本"。
    """
    problem, graph = pq
    from app.models.student import StudentState

    canned = StepMatch(
        matched_nodes=["S5", "不存在节点"], confidence=0.9, correct=True
    )
    install_fake_llm(monkeypatch, step_match=canned)

    result = match_step(problem, graph, StudentState(), "弹簧恢复原长时二者分离")

    assert result.matched_nodes == ["S5"]          # 非法节点被过滤
    assert result is not canned                    # 返回的是副本
    assert canned.matched_nodes == ["S5", "不存在节点"]   # 原对象未被改动
    assert canned.off_graph is False


def test_single_node_match(monkeypatch, pq):
    result = _match(
        monkeypatch,
        StepMatch(matched_nodes=["S5"], confidence=0.9, correct=True),
    )
    assert result.matched_nodes == ["S5"]
    assert result.correct is True
    assert result.off_graph is False


def test_multi_node_skip(monkeypatch, pq):
    """一步覆盖多个节点（跳步）：S3+S4+S5 一次匹配。"""
    result = _match(
        monkeypatch,
        StepMatch(
            matched_nodes=["S3", "S4", "S5"],
            confidence=0.95,
            correct=True,
            current_method="牛顿第二定律",
        ),
    )
    assert result.matched_nodes == ["S3", "S4", "S5"]
    assert result.current_method == "牛顿第二定律"


def test_correct_but_missing_reasoning(monkeypatch, pq):
    """结论正确但关键推理缺失（E2 场景）。"""
    result = _match(
        monkeypatch,
        StepMatch(
            matched_nodes=["S5"],
            correct=True,
            missing_reasoning=["缺少接触力 N=0 的判断", "缺少对 Q 的受力分析"],
        ),
    )
    assert result.correct is True
    assert len(result.missing_reasoning) == 2


def test_alternative_method_path(monkeypatch, pq):
    """替代路径：能量法（S6a）与简谐运动法（S6b）都可匹配。"""
    result = _match(monkeypatch, StepMatch(matched_nodes=["S6a"], correct=True, current_method="能量法"))
    assert "S6a" in result.matched_nodes
    result_b = _match(monkeypatch, StepMatch(matched_nodes=["S6b"], correct=True, current_method="简谐运动法"))
    assert "S6b" in result_b.matched_nodes


def test_off_graph_input(monkeypatch, pq):
    result = _match(
        monkeypatch,
        StepMatch(matched_nodes=[], confidence=0.1, correct=False, explanation="与解题无关"),
        text="老师，这道题什么时候交？",
    )
    assert result.matched_nodes == []
    assert result.off_graph is True


def test_wrong_answer_is_attempted_not_off_graph(monkeypatch, pq):
    """错误作答：不算完成，但要点出学生卡在哪个节点（E4 场景）。"""
    result = _match(
        monkeypatch,
        StepMatch(
            matched_nodes=[],
            attempted_nodes=["S10"],
            confidence=0.95,
            correct=False,
            missing_reasoning=["分离后系统边界变化：振子只剩弹簧+P"],
        ),
        text="分离后 P 的振幅还是 d。",
    )
    assert result.matched_nodes == []
    assert result.attempted_nodes == ["S10"]
    assert result.off_graph is False


def test_attempted_nodes_filtered_and_deduped(monkeypatch, pq):
    """attempted 中非法 id 被过滤、与 matched 重复的被剔除。"""
    result = _match(
        monkeypatch,
        StepMatch(matched_nodes=["S5"], attempted_nodes=["S5", "S99"]),
    )
    assert result.matched_nodes == ["S5"]
    assert result.attempted_nodes == []


def test_invalid_node_ids_filtered(monkeypatch, pq):
    """后校验：LLM 编造的节点 id 必须被过滤。"""
    result = _match(monkeypatch, StepMatch(matched_nodes=["S5", "S99"], confidence=0.9))
    assert result.matched_nodes == ["S5"]


def test_all_invalid_ids_become_off_graph(monkeypatch, pq):
    result = _match(monkeypatch, StepMatch(matched_nodes=["X1", "X2"], confidence=0.9))
    assert result.matched_nodes == []
    assert result.off_graph is True


def test_confidence_clamped(monkeypatch, pq):
    result = _match(monkeypatch, StepMatch(matched_nodes=["S1"], confidence=1.7))
    assert result.confidence == 1.0


def test_prompt_contains_graph_and_visited(monkeypatch, pq):
    fake = install_fake_llm(monkeypatch)
    from app.models.student import StudentState

    state = StudentState(visited_nodes=["S1", "S2", "S3"])
    match_step(pq[0], pq[1], state, "学生输入")
    system, user = fake.prompts[-1]
    assert "步骤匹配器" in system
    assert "S6a" in user and "完整路径1" in user      # Graph 结构进入了提示词
    assert "S1、S2、S3" in user                        # 已完成节点进入了提示词
