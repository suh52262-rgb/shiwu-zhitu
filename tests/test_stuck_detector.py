"""卡顿检测器测试：行为规则（文档第五节）+ Graph 感知增强（P4）。"""
import pytest

from app.models.student import StudentState
from app.services.problem_store import load_problem_graph
from app.services.stuck_detector import (
    GraphSignals,
    StuckSignals,
    build_graph_signals,
    compute_graph_aware_stuck_score,
    compute_stuck_score,
    is_stuck,
    stuck_reasons,
)


def test_no_signals_means_zero():
    assert compute_stuck_score(StuckSignals()) == 0.0


def test_idle_only():
    score = compute_stuck_score(StuckSignals(idle_time=41.0))
    assert score == 0.30


def test_all_rules_capped_at_one():
    signals = StuckSignals(idle_time=60.0, error_count=3, rewrite_count=4, hint_count=2)
    assert compute_stuck_score(signals) == 1.0


def test_thresholds_are_strict():
    # 边界值不计分：idle 需严格大于 40，error 需 >= 2
    assert compute_stuck_score(StuckSignals(idle_time=40.0)) == 0.0
    assert compute_stuck_score(StuckSignals(error_count=1)) == 0.0
    assert compute_stuck_score(StuckSignals(error_count=2)) == 0.25
    assert compute_stuck_score(StuckSignals(rewrite_count=3)) == 0.25
    assert compute_stuck_score(StuckSignals(hint_count=1)) == 0.20


def test_is_stuck_threshold():
    assert is_stuck(0.5) is True
    assert is_stuck(0.49) is False


# ---------- Graph 感知增强（P4） ----------

def test_graphless_scores_equal_base():
    signals = StuckSignals(idle_time=50.0)
    assert compute_graph_aware_stuck_score(signals, GraphSignals()) == pytest.approx(0.30)


def test_key_node_retry_bonus():
    """在关键节点上反复尝试（>=2 次）加分；只尝试 1 次不加。"""
    base = GraphSignals(current_node="S3", is_key_node=True, node_retry_count=2)
    assert compute_graph_aware_stuck_score(StuckSignals(), base) == pytest.approx(0.10)
    once = GraphSignals(current_node="S3", is_key_node=True, node_retry_count=1)
    assert compute_graph_aware_stuck_score(StuckSignals(), once) == pytest.approx(0.0)


def test_node_error_repeat_bonus():
    repeated = GraphSignals(current_node="S10", node_retry_count=2, node_error_count=2)
    assert compute_graph_aware_stuck_score(StuckSignals(), repeated) == pytest.approx(0.15)


def test_all_graph_bonuses_and_cap():
    signals = StuckSignals(idle_time=60.0, error_count=3)  # 行为分 0.55
    graph = GraphSignals(
        current_node="S3",
        is_key_node=True,
        node_retry_count=4,
        node_error_count=3,
        weak_concept_overlap=True,
    )  # 图加成 0.45
    # 0.55 + 0.45 = 1.0，封顶生效
    assert compute_graph_aware_stuck_score(signals, graph) == 1.0


def test_stuck_reasons_lists_triggered_rules():
    signals = StuckSignals(idle_time=50.0, error_count=2)
    graph = GraphSignals(current_node="S3", is_key_node=True, node_retry_count=3, node_error_count=2)
    reasons = stuck_reasons(signals, graph)
    assert any("空闲" in r for r in reasons)
    assert any("累计明显错误" in r for r in reasons)
    assert any("关键节点 S3" in r for r in reasons)
    assert any("S3" in r and "出错" in r for r in reasons)


def test_stuck_reasons_empty_when_fine():
    assert stuck_reasons(StuckSignals(), GraphSignals()) == []


def test_build_graph_signals_from_real_pq_graph():
    """用真实 phy_002 Graph 验证信号提取：S3 是关键推理节点且涉及薄弱知识点。"""
    graph = load_problem_graph("phy_002")
    state = StudentState(
        current_node="S3",
        node_attempts={"S3": 2},
        node_errors={"S3": 2},
        weak_concepts=["物体分离条件"],
        error_evidence=[{"concept":"物体分离条件", "issue":"关系错误", "student_text":"T=mg"}],
    )
    signals = build_graph_signals(graph, state)
    assert signals.current_node_type == "key_reasoning"
    assert signals.is_key_node is True
    assert signals.node_retry_count == 2
    assert signals.node_error_count == 2
    assert signals.weak_concept_overlap is True

    score = compute_graph_aware_stuck_score(StuckSignals(), signals)
    # key_node_stuck 0.10 + node_error_repeat 0.15 + weak_concept 0.10
    assert score == pytest.approx(0.35)


def test_build_graph_signals_graphless_or_unknown_node():
    assert build_graph_signals(None, StudentState(current_node="S3")) == GraphSignals()
    assert build_graph_signals(load_problem_graph("phy_002"), StudentState()) == GraphSignals()


def test_legacy_help_label_does_not_raise_graph_score():
    state = StudentState(current_node="S3", weak_concepts=["物体分离条件"])
    signals = build_graph_signals(load_problem_graph("phy_002"), state)
    assert signals.weak_concept_overlap is False
