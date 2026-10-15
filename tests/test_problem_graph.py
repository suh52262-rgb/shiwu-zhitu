"""ProblemGraph Schema 与存储测试（P1/P2）。"""
import pytest

from app.models.problem_graph import ProblemGraph
from app.services.problem_store import load_problem_graph


@pytest.fixture
def pq_graph() -> ProblemGraph:
    graph = load_problem_graph("phy_002")
    assert graph is not None, "data/problem_graphs/phy_002.json 缺失"
    return graph


def test_load_pq_spring_graph(pq_graph):
    assert pq_graph.problem_id == "phy_002"
    assert len(pq_graph.nodes) == 12
    ids = pq_graph.node_ids()
    assert {"S3", "S5", "S6a", "S6b", "S8", "S10"} <= ids


def test_branching_edges_and_valid_paths(pq_graph):
    # S5 处分叉为能量法 / 简谐运动法两条合理路径
    assert pq_graph.successors("S5") == ["S6a", "S6b"]
    assert pq_graph.successors("S6a") == ["S7"]
    assert pq_graph.successors("S6b") == ["S7"]
    assert len(pq_graph.valid_paths) == 2
    assert all(path[0] == "S1" and path[-1] == "S11" for path in pq_graph.valid_paths)


def test_unknown_node_has_no_successors(pq_graph):
    assert pq_graph.successors("S99") == []


def test_key_nodes_carry_common_errors(pq_graph):
    node_map = pq_graph.node_map()
    assert any(err.startswith("E1") for err in node_map["S3"].common_errors)
    assert any(err.startswith("E4") for err in node_map["S10"].common_errors)


def test_render_for_prompt_contains_graph_structure(pq_graph):
    text = pq_graph.render_for_prompt()
    assert "S3 确定分离条件" in text
    assert "S6a" in text and "S6b" in text
    assert "推进关系" in text
    assert "完整路径1" in text


def test_graph_schema_roundtrip(pq_graph):
    restored = ProblemGraph(**pq_graph.model_dump())
    assert restored.node_ids() == pq_graph.node_ids()
    assert restored.edges == pq_graph.edges
