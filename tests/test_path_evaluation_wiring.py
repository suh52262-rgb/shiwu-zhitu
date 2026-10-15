"""Real report wiring, with model IO replaced by deterministic predictions."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.agent.step_matcher import StepMatch
from app.evaluation.path_metrics import aggregate_node_recall, labeled_node_recall
from app.models.problem_graph import GraphNode, ProblemGraph
from app.services.graph_versions import graph_hash


@pytest.fixture
def graph():
    return ProblemGraph(problem_id="p", nodes=[
        GraphNode(id=n, name=n, description="a documented step", type="result")
        for n in ["start", "energy", "force", "end"]
    ], edges=[["start", "energy"], ["energy", "end"],
              ["start", "force"], ["force", "end"]],
        valid_paths=[["start", "energy", "end"], ["start", "force", "end"]])


@pytest.fixture
def runner(monkeypatch, graph):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("path_eval_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_problem_graph", lambda _: graph)
    monkeypatch.setattr(module, "load_problem", lambda _: object())
    return module


def reviewed(graph):
    # Synthetic test-only review record, never added to project datasets.
    return {"status": "reviewed", "review_source": "human",
            "reviewer": "synthetic-test-fixture",
            "reviewed_at": "2026-10-08T00:00:00+00:00", "graph_hash": graph_hash(graph)}


@pytest.mark.parametrize("label", [
    {"gold_path": ["start", "force", "end"]},
    {"method": "newton"},
    {"gold_path": "newton"},
    {"gold_nodes": ["start", "force", "end"]},
    {"expected": {"gold_nodes": ["start", "force", "end"]}},
])
def test_only_labeled_route_counts(graph, label):
    r = labeled_node_recall(label, {"start", "force", "end"}, graph,
                            {"newton": ["start", "force", "end"]})
    assert (r.numerator, r.denominator, r.value) == (3, 3, 1.0)
    assert "energy" not in str(r.detail)


@pytest.mark.parametrize("label", [{}, {"method": "missing"},
    {"gold_nodes": ["unknown"]}, {"gold_path": ["start", "end"]},
    {"method": []}])
def test_missing_or_invalid_labels_never_guessed(graph, label):
    assert labeled_node_recall(label, graph.node_ids(), graph).value is None


def test_aggregation_reports_counts_and_excluded(graph):
    labeled = labeled_node_recall({"gold_nodes": ["force", "end"]}, {"force"}, graph)
    missing = labeled_node_recall({}, {"energy"}, graph)
    r = aggregate_node_recall([("labeled", labeled), ("unlabeled", missing)])
    assert (r.numerator, r.denominator, r.value) == (1, 2, .5)
    assert "1 条" in r.note
    assert "unlabeled: n/a" in r.detail[1]


def test_zero_coverage_still_displays_counts_when_labels_excluded(graph):
    r = aggregate_node_recall([
        ("valid", labeled_node_recall({"gold_nodes": ["force"]}, set(), graph)),
        ("missing", labeled_node_recall({}, set(), graph)),
    ])
    assert "0/1" in r.render()


def test_alternative_batch_excludes_control_and_unlabeled(runner, monkeypatch):
    monkeypatch.setattr(runner, "match_step", lambda *a: StepMatch(
        matched_nodes=["force"], correct=a[3] != "rejected"))
    cases = {"problem_id": "p", "cases": [
        {"id": "control", "student_input": "ok", "is_alternative": False,
         "expected": {"correct": True}},
        {"id": "alt", "student_input": "rejected", "is_alternative": True,
         "expected": {"correct": True}, "gold_nodes": ["force"]},
        {"id": "unknown", "student_input": "ok", "is_alternative": True},
    ]}
    for case in cases["cases"]:
        case["annotation"] = reviewed(runner.load_problem_graph("p"))
    passed, total, notes = runner.eval_alternative_path(cases, 99)
    assert (passed, total) == (1, 2)
    assert runner.ALTERNATIVE_LABELS == [("alt", False)]
    assert "SKIP unknown" in notes[-1]
    assert len(runner.PATH_CASE_RESULTS) == 4  # graph plus all three examples


def test_graph_round_coverage_uses_touched_not_just_completed(runner, monkeypatch):
    monkeypatch.setattr(runner, "match_step", lambda *a: StepMatch(
        matched_nodes=["start"], attempted_nodes=["force"], correct=True))
    runner.eval_graph({"problem_id": "p", "gold_nodes": ["start", "force"],
        "annotation": reviewed(runner.load_problem_graph("p")),
        "rounds": [{"id": "r1", "student_input": "x"}],
        "expected_contains_any_of": [["start"]]}, 99)
    final = runner.PATH_CASE_RESULTS[-1]
    assert final["node_recall"]["value"] == 1.0
    assert final["touched"] == ["force", "start"]


def test_offline_cli_report_structure_only_and_no_cross_run_mix(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runner, "load_cases", lambda _: {"problem_id": "p", "rounds": []})
    monkeypatch.setattr(runner, "match_step", lambda *a: pytest.fail("offline called model"))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "graph", "--offline"])
    runner.ALTERNATIVE_LABELS.append(("stale", True))
    for _ in range(2):
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 0
    report = json.loads((tmp_path / "reports" / "latest-suite-graph.json").read_text("utf-8"))
    structure = report["metrics"]["structure_valid_rate"]
    assert (structure["numerator"], structure["denominator"]) == (1, 1)
    assert "不含物理" in structure["semantics"]
    assert report["path_cases"][0]["physics_checked"] is False
    assert report["path_cases"][0]["source"] == "author_candidate"
    assert report["path_cases"][0]["graph_hash"] == graph_hash(runner.load_problem_graph("p"))
    assert report["graph_source"] == "author_candidate"
    assert "不代表已发布版本质量" in report["graph_scope_note"]
    assert any("模型匹配 SKIPPED" in line and "结构检查已执行" in line
               for line in report["console"])
    assert report["metrics"]["node_recall"]["value"] is None
    assert report["metrics"]["alternative_path_acceptance"]["value"] is None
    assert not (tmp_path / "reports" / "latest-full.json").exists()


def test_alternative_cli_report_has_actual_prediction_counts(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runner, "load_cases", lambda _: {"problem_id": "p", "cases": [
        {"id": "alt", "student_input": "force route", "is_alternative": True,
         "gold_path": ["start", "force", "end"], "expected": {"correct": True},
         "annotation": reviewed(runner.load_problem_graph("p"))},
        {"id": "unlabeled", "student_input": "other", "expected": {"correct": False}},
    ]})
    monkeypatch.setattr(runner, "match_step", lambda *a: StepMatch(
        matched_nodes=["start", "force", "end"], correct=a[3] == "force route"))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "alternative_path"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 0
    payload = json.loads((tmp_path / "reports" / "latest-suite-alternative_path.json").read_text("utf-8"))
    recall = payload["metrics"]["node_recall"]
    assert (recall["numerator"], recall["denominator"]) == (3, 3)
    assert "1 条" in recall["note"]
    accepted = payload["metrics"]["alternative_path_acceptance"]
    assert (accepted["numerator"], accepted["denominator"]) == (1, 1)
    assert payload["path_cases"][1]["alternative_accepted"] is True
    expected_hash = graph_hash(runner.load_problem_graph("p"))
    assert all(row["graph_hash"] == expected_hash and row["source"] == "author_candidate"
               for row in payload["path_cases"])
    assert payload["path_cases"][2]["node_recall"]["value"] is None


def test_structure_records_distinct_content_versions(runner, graph):
    runner._record_structure("p", graph)
    changed = graph.model_copy(deep=True)
    changed.nodes[0].description = "new author content"
    runner._record_structure("p", changed)
    runner._record_structure("p", changed)
    assert len(runner.STRUCTURE_RESULTS) == 2
    assert {row["graph_hash"] for row in runner.PATH_CASE_RESULTS} == {
        graph_hash(graph), graph_hash(changed)}


def test_proposed_labels_never_call_model_by_default(runner, monkeypatch, graph):
    monkeypatch.setattr(runner, "match_step", lambda *a: pytest.fail("unreviewed model call"))
    case = {"id": "candidate", "student_input": "force", "gold_nodes": ["force"],
            "is_alternative": True, "expected": {"correct": True},
            "annotation": {"status": "proposed", "graph_hash": graph_hash(graph)}}
    dataset = {"problem_id": "p", "cases": [case]}
    assert runner.eval_step_matching(dataset, 99)[:2] == (0, 0)
    assert runner.eval_alternative_path(dataset, 99)[:2] == (0, 0)
    assert runner.ALTERNATIVE_LABELS == []
    assert all(row["node_recall"]["value"] is None for row in runner.PATH_CASE_RESULTS
               if "node_recall" in row)


def test_candidate_trial_report_cannot_replace_reviewed_baseline(runner, monkeypatch, tmp_path, graph):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runner, "load_cases", lambda _: {"problem_id": "p", "cases": [{
        "id": "candidate", "student_input": "force", "gold_nodes": ["force"],
        "is_alternative": True, "expected": {"correct": True},
        "annotation": {"status": "proposed", "graph_hash": graph_hash(graph)}}]})
    monkeypatch.setattr(runner, "match_step", lambda *a: StepMatch(matched_nodes=["force"], correct=True))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "alternative_path", "--include-proposed-labels"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 0
    candidate = tmp_path / "reports" / "candidate-latest-suite-alternative_path.json"
    report = json.loads(candidate.read_text("utf-8"))
    assert report["label_scope"] == "proposed_exploratory"
    assert report["metrics"]["node_recall"]["denominator"] == 1
    assert not (tmp_path / "reports" / "latest-suite-alternative_path.json").exists()


@pytest.mark.parametrize("scope,approved,expected_code", [
    ("proposed_exploratory", True, 0),
    ("reviewed_only", True, 2),
    ("proposed_exploratory", False, 2),
])
def test_manifest_enforces_label_scope_and_budget_before_model(
        runner, monkeypatch, tmp_path, scope, approved, expected_code):
    import hashlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from app.config import get_settings
    settings = get_settings().model_copy(deep=True)
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    source = tmp_path / "input.json"
    source.write_text("frozen", encoding="utf-8")
    manifest = {
        "status": "prepared_not_executed", "suites": ["alternative_path"], "limit": 99,
        "files": {"input.json": hashlib.sha256(source.read_bytes()).hexdigest()},
        "evaluation_config": {"min_accuracy": 0.7, "label_scope": scope},
        "budget": {"approved": approved},
        "model_config": {"llm_model": "synthetic-test-model"}, "prompt_versions": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runner, "load_cases", lambda _: {"problem_id": "p", "cases": []})
    monkeypatch.setattr(runner, "match_step", lambda *a: pytest.fail("no model expected"))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "alternative_path",
        "--manifest", str(path), "--include-proposed-labels"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == expected_code


def test_model_failure_still_saves_usage_and_error_report(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runner, "load_cases", lambda _: {"problem_id": "p", "cases": []})
    def fail(*a, **kw):
        raise RuntimeError("synthetic upstream failure")
    monkeypatch.setitem(runner.EVALUATORS, "alternative_path", fail)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "alternative_path"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    report = json.loads((tmp_path / "reports" / "latest-suite-alternative_path.json").read_text("utf-8"))
    assert report["overall_ok"] is False
    assert "synthetic upstream failure" in report["suite_results"]["alternative_path"]["error"]
    assert report["token_usage"]["alternative_path"]["calls"] == 0
