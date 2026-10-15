"""Synthetic aggregator fixtures; never evaluation measurements."""
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import summarize_t4_development as mod
import run_t4_batch as runner


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    batch = tmp_path / "batch"
    mapping, records, items = {}, [], []
    categories = sorted(runner.CATEGORIES)
    problems = []
    for p in range(4):
        cases = []
        for n, category in enumerate(categories):
            case_id = f"p{p}c{n}"
            cases.append({"id": case_id, "category": category})
            for repeat in range(1, 4):
                blind_id = f"Q{len(items) + 1:03d}"
                identity = {"case_id": case_id, "repeat": repeat, "split": "development", "category": category}
                mapping[blind_id] = identity
                records.append(identity)
                items.append({"blind_item_id": blind_id, "ratable": True,
                    **{k: 5 for k in mod.SCORES}, **{k: False for k in mod.FLAGS},
                    "material_has_definite_error": category == "explicit_error",
                    "explicit_error_detected": category == "explicit_error", "evidence": "Synthetic evidence"})
        problems.append({"split": "development", "cases": cases})
    mod.write(tmp_path / "dataset.json", {"problems": problems})
    manifest = tmp_path / "manifest.json"
    mod.write(manifest, {"dataset": "dataset.json", "files": {"dataset.json": mod.digest(tmp_path / "dataset.json")}})
    for n, r in enumerate(records):
        mod.write(batch / "runs" / f"r{n}.json", {**r, "status": "succeeded", "manifest_sha256": mod.digest(manifest)})
    mod.write(batch / "restricted_mapping.json", mapping)
    summary_path = batch / "summary.json"
    mod.write(summary_path, {"dry_run": False, "smoke": False, "records": 72, "succeeded": 72, "failed": 0,
                            "evidence_sha256": mod.evidence_digest(batch)})
    bindings = {"manifest_sha256": mod.digest(manifest), "summary_sha256": mod.digest(summary_path),
                "evidence_sha256": mod.evidence_digest(batch)}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for n, target in enumerate((a, b)):
        mod.write(target, {**bindings, "reviewer_id": f"reviewer{n}", "provenance": "independent_ai_review",
                           "initial_ratings_locked": True, "items": items})
    return batch, manifest, a, b, tmp_path / "out"


def edit(path, mutate):
    value = mod.read(path)
    mutate(value)
    mod.write(path, value)


def test_pass_is_gate_compatible_and_initial_bytes_preserved(fixture):
    batch, manifest, a, b, out = fixture
    initial = a.read_bytes()
    gate = mod.aggregate(*fixture)
    assert gate["passed"] and not gate["adjudication_required"]
    assert (out / "ratings_1_initial.json").read_bytes() == initial
    runner.check_development_gate(out / "development_gate.json", manifest,
                                  mod.read(manifest.parent / "dataset.json"))
    report = mod.read(out / "reviewer_1_report.json")
    assert report["metrics"]["explicit_error_total"] == 12
    assert len(report["by_category"]) == 6
    with pytest.raises(ValueError, match="Output exists"):
        mod.aggregate(*fixture)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(initial_ratings_locked=False),
    lambda r: r.update(summary_sha256="wrong"),
    lambda r: r["items"].pop(),
    lambda r: r["items"][0].update(ratable=False, correctness_1to5=0),
    lambda r: r["items"][0].update(severe_false_accusation="false"),
])
def test_bad_initials_rejected_before_output(fixture, mutation):
    edit(fixture[2], mutation)
    with pytest.raises(ValueError):
        mod.aggregate(*fixture)
    assert not fixture[4].exists()


def test_disagreements_block_even_when_both_reviewers_accept(fixture):
    edit(fixture[3], lambda r: r["items"][0].update(correctness_1to5=3))
    gate = mod.aggregate(*fixture)
    assert not gate["passed"] and gate["adjudication_required"]
    assert mod.read(fixture[4] / "reviewer_2_report.json")["verdict"] == "accept"
    assert gate["disagreements"][0]["fields"] == ["correctness_1to5"]


def test_severity_failure_and_detection_threshold(fixture):
    for path in fixture[2:4]:
        def mutate(r):
            r["items"][0]["serious_physics_contradiction"] = True
            explicit = [i for i in r["items"] if i["material_has_definite_error"]]
            for item in explicit[:3]:
                item["explicit_error_detected"] = False
        edit(path, mutate)
    gate = mod.aggregate(*fixture)
    assert not gate["passed"] and not gate["adjudication_required"]
    report = mod.read(fixture[4] / "reviewer_1_report.json")
    assert report["verdict"] == "reject"
    assert report["metrics"]["explicit_error_recognized"] == 9


def test_unratable_excluded_from_means(fixture):
    for path in fixture[2:4]:
        edit(path, lambda r: r["items"][0].update(ratable=False, **{k: None for k in mod.SCORES}))
    mod.aggregate(*fixture)
    metrics = mod.read(fixture[4] / "reviewer_1_report.json")["metrics"]
    assert metrics["ratable"] == 71
    assert metrics["unratable"] == 1
    assert metrics["mean_scores_ratable_only"]["correctness_1to5"] == 5


def test_raw_drift_rejected(fixture):
    mod.write(fixture[0] / "runs" / "extra.json", {})
    with pytest.raises(ValueError, match="unchanged raw evidence"):
        mod.aggregate(*fixture)


def change_failed_batch(fixture, count):
    batch, _, a, b, _ = fixture
    for path in sorted((batch / "runs").glob("*.json"))[:count]:
        edit(path, lambda r: r.update(status="failed"))
    edit(batch / "summary.json", lambda s: s.update(succeeded=72-count, failed=count,
                                                   evidence_sha256=mod.evidence_digest(batch)))
    for path in (a, b):
        edit(path, lambda r: r.update(summary_sha256=mod.digest(batch / "summary.json"),
                                     evidence_sha256=mod.evidence_digest(batch)))


def test_failed_workflows_cannot_enter_means(fixture):
    change_failed_batch(fixture, 1)
    with pytest.raises(ValueError, match="Failed workflows must be unratable"):
        mod.aggregate(*fixture)


def test_availability_failures_remain_in_denominator(fixture):
    change_failed_batch(fixture, 8)
    for path in fixture[2:4]:
        def mutate(review):
            for item in review["items"]:
                item.update(ratable=False, **{k: None for k in mod.SCORES})
        edit(path, mutate)
    gate = mod.aggregate(*fixture)
    report = mod.read(fixture[4] / "reviewer_1_report.json")
    assert not gate["passed"]
    assert report["metrics"]["failed"] == 8
    assert report["metrics"]["recorded"] == 72
    assert report["metrics"]["mean_scores_ratable_only"]["correctness_1to5"] is None


def test_frozen_dataset_drift_rejected(fixture):
    edit(fixture[1].parent / "dataset.json", lambda d: d["problems"][0]["cases"][0].update(category="changed"))
    with pytest.raises(ValueError, match="Frozen dataset drift"):
        mod.aggregate(*fixture)
