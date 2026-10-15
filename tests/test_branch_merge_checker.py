"""Merge review regression fixtures are disclosed examples, not blind accuracy data."""
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

from app.models.problem_graph import GraphNode, ProblemGraph
from app.services.branch_merge_checker import inspect_branch_merges
from app.services.graph_validator import validate_graph

FIXTURES = FIXTURE_ROOT / "branch_merges"
ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return ProblemGraph.model_validate_json(path.read_text(encoding="utf-8"))


def diamond(left="x = 2", right="x = 3", target="核对结果", description="x = 2"):
    return ProblemGraph(problem_id="independent-example", nodes=[
        GraphNode(id="start", name="建模", description="初始条件"),
        GraphNode(id="left", name="路线一", description=left),
        GraphNode(id="right", name="路线二", description=right),
        GraphNode(id="end", name=target, description=description, type="result")],
        edges=[["start", "left"], ["start", "right"], ["left", "end"], ["right", "end"]],
        valid_paths=[["start", "left", "end"], ["start", "right", "end"]])


@pytest.mark.parametrize("problem,node,rule", [
    ("demo_03", "S3", "repeated_numeric_calculation"),
    ("demo_04", "S6", "root_selection_after_amplitude"),
])
def test_disclosed_before_fixtures_produce_path_evidence(problem, node, rule):
    graph = load(FIXTURES / f"{problem}_before.json")
    result = inspect_branch_merges(graph)
    finding = next(f for f in result["findings"] if f["node"] == node)
    assert finding["path_index"] == 2
    assert any(e["rule"] == rule for e in finding["evidence"])
    assert graph.metadata["needs_review"] is True


def test_legal_demo02_symbolic_routes_do_not_trigger_recalculation():
    result = inspect_branch_merges(load(PROBLEMS / "demo_02/graph.json"))
    assert result["findings"] == []
    assert result["unknown_count"] == 1  # No proof of semantic validity.


def test_independent_repeated_calculation_and_verification_are_distinct():
    assert inspect_branch_merges(diamond(target="求 x"))["findings"]
    assert inspect_branch_merges(diamond())["findings"] == []


def test_independent_amplitude_route_does_not_need_common_root_selection():
    graph = diamond(left="列二次方程并取正根", right="由振幅关系求位移",
                    target="取正根", description="取两个根中正根")
    result = inspect_branch_merges(graph)
    assert len(result["findings"]) == 1
    assert result["findings"][0]["prefix"] == ["start", "right"]


def test_explicit_contract_checks_every_full_prefix():
    graph = diamond()
    graph.metadata["merge_contracts"] = {
        "start": {"produces": ["mass"]}, "left": {"produces": ["speed"]},
        "right": {"produces": []}, "end": {"requires": ["mass", "speed"]}}
    result = inspect_branch_merges(graph)
    routes = result["merges"][0]["routes"]
    assert routes[0]["status"] == "annotated_pass"
    assert routes[1]["evidence"][0]["missing"] == ["speed"]
    assert result["findings"][0]["path_index"] == 2


@pytest.mark.parametrize("invalid", [None, [], {"end": {"requires": "speed"}},
                                     {"end": {"requires": [None]}}])
def test_missing_or_malformed_contract_is_unknown(invalid):
    graph = diamond()
    graph.metadata["merge_contracts"] = invalid
    result = inspect_branch_merges(graph)
    assert result["unknown_count"] == 1
    assert result["findings"] == []


def test_invalid_paths_and_edges_do_not_crash_or_count_as_evidence():
    graph = diamond()
    graph.valid_paths = [["start", "ghost", "end"], ["end", "right"]]
    graph.edges.append(["bad"])
    result = inspect_branch_merges(graph)
    assert result["merges"][0]["routes"] == []
    assert result["unknown_count"] == 1


def test_no_merge_is_vacuously_passed():
    graph = diamond()
    graph.edges = [["start", "left"]]
    report = validate_graph(graph, use_llm=False)
    assert report.checks["branch_merges"] == "pass"


def test_report_stores_compact_summary_and_preserves_human_review_flag():
    graph = diamond(target="求 x")
    graph.metadata["needs_review"] = True
    report = validate_graph(graph, use_llm=False)
    assert report.checks["branch_merges"] == "issues"
    assert report.valid is True  # Suspicion cannot make a student's route invalid.
    assert report.recommended_action == "teacher_review"
    assert report.metadata["branch_merges"]["findings"]
    compact = report.as_metadata()["branch_merges"]
    assert compact["merge_count"] == 1
    assert "findings" not in compact
    assert graph.metadata["needs_review"] is True


def test_unknown_semantics_is_not_reported_as_pass():
    report = validate_graph(diamond(), use_llm=False)
    assert report.checks["branch_merges"] == "skipped"
    assert "unknown" in report.summary
    assert report.as_metadata()["branch_merges"]["unknown_count"] == 1


def test_complete_labels_mean_annotation_consistency_only():
    graph = diamond()
    graph.metadata["merge_contracts"] = {
        "start": {"produces": []}, "left": {"produces": ["result"]},
        "right": {"produces": ["result"]}, "end": {"requires": ["result"]}}
    result = inspect_branch_merges(graph)
    assert result["unknown_count"] == 0
    assert result["merges"][0]["status"] == "annotated_pass"
    assert "不证明物理正确性" in result["merges"][0]["routes"][0]["reason"]
