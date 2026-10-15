"""Integration checks for false-branch warnings in the standard validator and CLI."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.models.graph_validation import ISSUE_FALSE_BRANCH
from app.models.problem_graph import GraphNode, ProblemGraph
from app.services.graph_validator import validate_graph


def branching_graph():
    return ProblemGraph(
        problem_id="fork",
        nodes=[
            GraphNode(id="S1", name="建模", type="modeling", description="识别粒子"),
            GraphNode(id="S2", name="静电力", type="calculation", description="库仑定律"),
            GraphNode(id="S3", name="万有引力", type="calculation", description="引力定律"),
            GraphNode(id="S4", name="结果", type="result", description="比较结果"),
        ],
        edges=[["S1", "S2"], ["S1", "S3"], ["S2", "S4"], ["S3", "S4"]],
        valid_paths=[["S1", "S2", "S4"], ["S1", "S3", "S4"]],
        metadata={"needs_review": True},
    )


def load_cli(name):
    spec = importlib.util.spec_from_file_location(
        f"test_cli_{name}", Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structurally_legal_false_branch_is_warning_with_evidence():
    graph = branching_graph()
    graph.metadata["quantities_asked"] = ["静电力", "万有引力"]
    before = graph.model_dump()
    report = validate_graph(graph, use_llm=False)
    assert report.checks["structure"] == report.checks["paths"] == "pass"
    assert report.checks["false_branches"] == "issues"
    assert report.valid is True
    assert report.recommended_action == "teacher_review"
    issue, = report.by_type(ISSUE_FALSE_BRANCH)
    assert issue.severity == "warning" and issue.node == "S1"
    assert "疑似" in issue.description and "需人工复核" in issue.description
    assert report.metadata["false_branches"]["findings"][0]["hit"] == {
        "S2": ["静电力"], "S3": ["万有引力"]}
    assert graph.model_dump() == before


@pytest.mark.parametrize("annotation", [None, [], "静电力", [""], ["静电力", 3]])
def test_missing_or_invalid_annotations_are_skipped(annotation):
    graph = branching_graph()
    graph.metadata["quantities_asked"] = annotation
    report = validate_graph(graph, use_llm=False)
    assert report.checks["false_branches"] == "skipped"
    assert "假分叉未检查" in report.summary
    assert "缺少有效" in report.as_metadata()["false_branches"]["reason"]


def test_explicit_annotation_overrides_metadata_and_same_target_is_not_flagged():
    graph = branching_graph()
    graph.metadata["quantities_asked"] = ["静电力", "万有引力"]
    graph.nodes[2].name = "静电力另一解法"
    report = validate_graph(graph, use_llm=False, quantities_asked=["静电力", "张力"])
    assert report.checks["false_branches"] == "pass"
    assert not report.by_type(ISSUE_FALSE_BRANCH)
    assert report.metadata["false_branches"]["source"] == "argument"
    skipped = validate_graph(graph, use_llm=False, quantities_asked=[])
    assert skipped.checks["false_branches"] == "skipped"


def test_independent_cli_accepts_explicit_targets_without_mutating_graph(monkeypatch, capsys):
    cli = load_cli("validate_graph")
    graph = branching_graph()

    class Repo:
        def get_graph(self, _):
            return graph

        def get_problem(self, _):
            return None

    monkeypatch.setattr(cli, "get_problem_repository", Repo)
    monkeypatch.setattr(sys, "argv", ["validate_graph", "fork", "--rules-only", "--json",
                                     "--quantity-asked", "静电力", "--quantity-asked", "万有引力"])
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["false_branches"] == "issues"
    assert graph.metadata == {"needs_review": True}


def test_generate_cli_saves_targets_and_validation(monkeypatch, tmp_path):
    cli = load_cli("generate_graph")
    saved = {}

    class Repo:
        def has_graph(self, _):
            return False

        def save_graph(self, pid, graph):
            saved[pid] = graph.model_copy(deep=True)
            return tmp_path / "graph.json"

        def get_graph(self, pid):
            return saved[pid].model_copy(deep=True)

        def get_problem(self, _):
            return None

        def merge_knowledge_points(self, *_):
            return []

    monkeypatch.setattr(cli, "get_problem_repository", Repo)
    monkeypatch.setattr(sys, "argv", ["generate_graph", "fork", "--statement", "测试题面",
                                     "--demo", "--save", "--validate",
                                     "--quantity-asked", "静电力", "--quantity-asked", "万有引力"])
    assert cli.main() == 0
    assert saved["fork"].metadata["quantities_asked"] == ["静电力", "万有引力"]
    assert saved["fork"].metadata["validation"]["checks"]["false_branches"] == "pass"
    assert saved["fork"].metadata["needs_review"] is True


def test_unreviewed_exam_cannot_bypass_publication_and_build_graph(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app.services import graph_generator, problem_repository

    cli = load_cli("import_exam")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"test")
    problem = SimpleNamespace(number="1", pages=[1], statement="测试题目内容足够超过十个字", solution="")
    monkeypatch.setattr(cli.exam_split, "split_pdf", lambda _: [problem])
    monkeypatch.setattr(cli.exam_split, "to_problem_payload",
                        lambda _, problem_id: {"problem_id": problem_id, "title": "测试题"})
    graph = branching_graph()
    saved = []

    class Repo:
        def get_problem(self, _):
            return problem

        def save_graph(self, _, result):
            saved.append(result)

    monkeypatch.setattr(problem_repository, "get_problem_repository", Repo)
    monkeypatch.setattr(graph_generator, "generate_and_save", lambda _: (graph, graph))
    monkeypatch.setattr(sys, "argv", ["import_exam", str(pdf), "--build-graph"])
    assert cli.main() == 1
    assert saved == []
    assert not (tmp_path / 'data' / 'problems').exists()
