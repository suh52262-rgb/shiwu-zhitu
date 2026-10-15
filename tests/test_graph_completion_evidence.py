"""Completion requires evidence for a whole route, not only its terminal node."""
import pytest

from app.agent.graph import update_state
from app.models.problem_graph import GraphNode, ProblemGraph
from app.models.student import StudentState
from app.services.problem_store import load_problem_graph


def finish(graph=None, **updates):
    state = StudentState(student_id="s", problem_id="demo_03", **updates)
    return update_state(state, {"configurable": {"problem_graph": graph}})


def test_recorded_demo03_terminal_hit_does_not_complete_unanswered_second_question():
    graph = load_problem_graph("demo_03")
    result = finish(graph, graph_mode=True, round_correct=True, current_node="S4",
                    visited_nodes=["S2", "S2a", "S4"],
                    missing_reasoning=["第(2)问未做：未求平均冲力（S3）"])
    assert result["finished"] is False
    assert result["is_complete"] is False


@pytest.mark.parametrize("visited", [["S4"], ["S2", "S2a", "S4"], ["S1", "S2", "S2a", "S4"]])
def test_terminal_hit_cannot_hide_incomplete_route_even_if_matcher_reports_no_missing_reasoning(visited):
    graph = load_problem_graph("demo_03")
    result = finish(graph, graph_mode=True, round_correct=True, current_node="S4", visited_nodes=visited)
    assert result["finished"] is False


@pytest.mark.parametrize("branch", ["S2a", "S2b"])
def test_either_complete_alternative_route_finishes_without_requiring_both_methods(branch):
    graph = load_problem_graph("demo_03")
    route = ["S1", "S2", branch, "S3", "S4"]
    result = finish(graph, graph_mode=True, round_correct=True, current_node="S4", visited_nodes=route)
    assert result["finished"] is True and result["is_complete"] is True
    assert result["visited_nodes"] == route


@pytest.mark.parametrize("blocker", [
    {"has_obvious_error": True},
    {"degraded_reasons": ["step matcher timeout"]},
    {"missing_reasoning": ["方向解释未交代完整"]},
])
def test_complete_route_does_not_override_current_round_blockers(blocker):
    graph = load_problem_graph("demo_03")
    result = finish(graph, graph_mode=True, round_correct=True, current_node="S4",
                    visited_nodes=graph.valid_paths[0], **blocker)
    assert result["finished"] is False


def test_missing_valid_paths_do_not_infer_and_or_completion_semantics():
    graph = ProblemGraph(problem_id="p", nodes=[GraphNode(id="S1", name="start"), GraphNode(id="S2", name="finish")], edges=[["S1", "S2"]])
    result = finish(graph, graph_mode=True, round_correct=True, current_node="S2", visited_nodes=["S1", "S2"])
    assert result["finished"] is False


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("blocker", [{"has_obvious_error": True}, {"degraded_reasons": ["analysis timeout"]}])
def test_model_completion_cannot_override_obvious_error_or_degradation(mode, blocker):
    result = finish(load_problem_graph("demo_03"), graph_mode=mode, is_complete=True, **blocker)
    assert result["finished"] is False


def test_explicit_graph_model_completion_cannot_override_known_missing_reasoning():
    result = finish(load_problem_graph("demo_03"), graph_mode=True, is_complete=True,
                    missing_reasoning=["第(2)问未做"])
    assert result["finished"] is False


def test_reliable_graphless_model_completion_keeps_existing_contract():
    result = finish(graph_mode=False, is_complete=True, has_obvious_error=False)
    assert result["finished"] is True
