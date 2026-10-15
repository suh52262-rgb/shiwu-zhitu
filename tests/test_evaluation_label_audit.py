"""Label gates use synthetic reviews; no human review is asserted for real cases."""
import importlib.util
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

from app.evaluation.label_audit import audit_dataset, label_eligibility
from app.models.problem_graph import GraphNode, ProblemGraph
from app.services.graph_versions import graph_hash


@pytest.fixture
def graph():
    return ProblemGraph(problem_id="p", nodes=[
        GraphNode(id=n, name=n, description=n, type="result") for n in ["a", "b"]
    ], edges=[["a", "b"]], valid_paths=[["a", "b"]])


def annotation(graph, **updates):
    return {"status": "reviewed", "review_source": "human", "reviewer": "test-fixture-only",
            "reviewed_at": "2026-10-08T00:00:00+00:00", "graph_hash": graph_hash(graph), **updates}


@pytest.mark.parametrize("updates,reason", [
    ({"status": "proposed"}, "pending_review"),
    ({"review_source": "AI"}, "missing_human_review"),
    ({"reviewer": None}, "missing_reviewer"),
    ({"reviewed_at": "2026-10-08"}, "invalid_review_date"),
    ({"graph_hash": "old"}, "graph_hash_drift"),
])
def test_unreviewed_invalid_or_stale_not_scored(graph, updates, reason):
    assert label_eligibility({"annotation": annotation(graph, **updates)}, graph) == (False, reason)


def test_candidate_trial_explicit_and_drift_still_excluded(graph):
    case = {"annotation": annotation(graph, status="proposed")}
    assert label_eligibility(case, graph, allow_proposed=True) == (True, "candidate_only")
    case["annotation"]["graph_hash"] = "old"
    assert not label_eligibility(case, graph, allow_proposed=True)[0]


def test_coverage_distinguishes_candidate_unknown_and_reviewed(graph):
    dataset = {"problem_id": "p", "cases": [
        {"id": "reviewed", "gold_nodes": ["a"], "annotation": annotation(graph)},
        {"id": "candidate", "gold_nodes": ["b"], "annotation": annotation(graph, status="proposed")},
        {"id": "unknown"},
    ]}
    report = audit_dataset(dataset, lambda _: graph)
    assert report["counts"] == {"total": 3, "reviewed": 1, "pending_review": 1,
        "unknown": 1, "hash_drift": 0, "candidate_node_labels": 2, "reviewed_node_labels": 1,
        "unresolved_node_labels": 1}


def test_manifest_verify_detects_changed_missing_and_escaping_files(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_eval.py"
    spec = importlib.util.spec_from_file_location("prepare_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "dataset.json"
    source.write_text("original", encoding="utf-8")
    manifest = {"files": {"dataset.json": module.digest(source)}}
    assert module.verify(manifest, tmp_path) == []
    source.write_text("changed", encoding="utf-8")
    assert module.verify(manifest, tmp_path) == ["input_drift:dataset.json"]
    source.unlink()
    assert module.verify(manifest, tmp_path) == ["input_drift:dataset.json"]
    manifest["files"] = {"../secret": "hash"}
    assert module.verify(manifest, tmp_path) == ["invalid_path:../secret"]


def test_synthetic_dataset_candidates_never_claim_human_review():
    from app.services.problem_store import load_problem_graph
    root = Path(__file__).resolve().parents[1]
    counts = []
    for name in ["step_matching", "alternative_path", "graph"]:
        dataset = json.loads((EVALUATION / "datasets" / f"{name}_cases.json").read_text("utf-8"))
        report = audit_dataset(dataset, load_problem_graph)
        assert report["counts"]["reviewed_node_labels"] == 0
        assert report["counts"]["hash_drift"] == 0
        counts.append(report["counts"]["candidate_node_labels"])
        for case in dataset.get("cases", [dataset]):
            assert case["annotation"]["status"] == "proposed"
            assert case["annotation"]["reviewer"] is None
    assert sum(counts) == 5
