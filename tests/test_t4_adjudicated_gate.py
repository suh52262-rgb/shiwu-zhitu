"""裁决后的"通过"必须能过自动关口 —— 但**每一条前提缺了就都不许过**。

## 为什么单开一个模块
`check_development_gate` 是**守住留出集的那道关口**。2026-10-09 发现：
它原先**只认初评**（要求每份评审 `verdict == "accept"`），
⇒ 裁决**推翻**了一条标记之后，「裁决后通过」的文件**仍然过不了关口** ——
**裁决等于写了个没人看的文件**（实测复现：`Independent development review does not meet
the frozen engineering gate`）。

接入裁决时最容易犯的错是"看到 `passed: true` 就放行"。
所以这里把**每一条前提**都做成一条会失败的测试。
"""
import importlib.util
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t4_runner_gate", ROOT / "scripts/run_t4_batch.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

MANIFEST = "manifest.json"


def _fixture(tmp_path, monkeypatch):
    """造一份**初评两份都是 reject**、但**裁决后通过**的完整门禁。"""
    data = runner.load_dataset(T4_DATASET)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    batch = tmp_path / "batch"
    manifest = tmp_path / MANIFEST
    runner.write(manifest, {"frozen": True})
    for item in runner.plan(data, "development", 1):
        runner.write(batch / "runs" / f"{item['case_id']}-{item['repeat']}.json",
                     {**item, "manifest_sha256": runner.digest(manifest),
                      "status": "succeeded", "output": "fixture"})
    summary = batch / "summary.json"
    runner.write(summary, {"dry_run": False, "smoke": False,
                           "evidence_sha256": runner.evidence_digest(batch)})

    reviews, initial_hashes = [], {}
    for i, key in ((0, "a"), (1, "b")):
        ratings = tmp_path / f"ratings_{i}.json"
        runner.write(ratings, {"reviewer_id": f"ai_{key}", "items": []})
        initial_hashes[key] = runner.digest(ratings)
        report = tmp_path / f"reviewer_{i}.json"
        runner.write(report, {
            "reviewer_id": f"ai_{key}", "manifest_sha256": runner.digest(manifest),
            "summary_sha256": runner.digest(summary),
            "evidence_sha256": runner.evidence_digest(batch),
            "provenance": "independent_ai_review",
            # ⚠️ **两份初评都 reject** —— 正是"裁决把结论翻过来"的那种情形
            "verdict": "reject", "initial_ratings_locked": True,
            "initial_ratings_sha256": runner.digest(ratings),
            "blind_item_ids": [f"Q{n:03d}" for n in range(1, 73)],
            "metrics": {"severe_false_accusations": 0, "unsupported_cognitive_claims": 1,
                        "invalid_displayed_references": 0, "serious_physics_contradictions": 0,
                        "explicit_error_total": 12, "explicit_error_recognized": 10}})
        reviews.append({"path": report.name, "sha256": runner.digest(report)})

    adjudication = tmp_path / "adjudication.json"
    runner.write(adjudication, {
        "provenance": "ai_integrator_adjudication",
        "adjudicator": "root_integrator_not_a_third_blind_reviewer",
        "initial_reviews_preserved": True,
        "manifest_sha256": runner.digest(manifest),
        "summary_sha256": runner.digest(summary),
        "evidence_sha256": runner.evidence_digest(batch),
        "original_initial_sha256": initial_hashes,
        "resolved_item_count": 1,
        "adjudicated_boolean_counts": {"severe_false_accusation": 0,
                                       "unsupported_cognitive_claim": 0,
                                       "invalid_displayed_reference": 0,
                                       "serious_physics_contradiction": 0},
        "explicit_error_recognition_after_adjudication": {"total": 12, "recognized": 10,
                                                          "met": True},
        "passed": True, "holdout_started": False})

    gate = tmp_path / "gate.json"
    runner.write(gate, {
        "passed": True, "manifest_sha256": runner.digest(manifest), "batch_dir": "batch",
        "summary_sha256": runner.digest(summary), "review_reports": reviews,
        "adjudication_sha256": runner.digest(adjudication),
        "disagreements_resolved_in": adjudication.name,
        # 残留的**评分**分歧允许（不进门槛判据）；说明键也允许
        "unresolved_disagreement_fields": {"answer_leakage_1to5": ["Q001"], "note": "分数留待下一轮"}})
    return data, gate, manifest, adjudication


def _patch(path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_a_reject_verdict_pair_can_still_pass_after_adjudication(tmp_path, monkeypatch):
    """**这就是那条真实的洞**：两份初评都 reject，裁决翻过来之后必须能过关。"""
    data, gate, manifest, _ = _fixture(tmp_path, monkeypatch)
    runner.check_development_gate(gate, manifest, data)      # 不抛就是通过


@pytest.mark.parametrize("mutate,message", [
    (lambda a: a.update(adjudicated_boolean_counts={
        "severe_false_accusation": 0, "unsupported_cognitive_claim": 1,
        "invalid_displayed_reference": 0, "serious_physics_contradiction": 0}),
     "severe metrics are not all zero"),
    (lambda a: a.update(explicit_error_recognition_after_adjudication={
        "total": 12, "recognized": 9, "met": False}),
     "recognition below the frozen threshold"),
    (lambda a: a.update(provenance="something_else"), "ai_integrator_adjudication"),
    (lambda a: a.update(adjudicator="third_blind_reviewer"), "not a third blind review"),
    (lambda a: a.update(initial_reviews_preserved=False), "preserve the initial locked reviews"),
    (lambda a: a.update(original_initial_sha256={}), "register the initial review hashes"),
    (lambda a: a.update(summary_sha256="0" * 64), "bindings do not match"),
])
def test_adjudication_preconditions_are_enforced(tmp_path, monkeypatch, mutate, message):
    """**每一条前提缺了都不许过** —— 尤其"自称是第三名盲评者"和"没保留初评"。"""
    data, gate, manifest, adjudication = _fixture(tmp_path, monkeypatch)
    _patch(adjudication, mutate)
    _patch(gate, lambda g: g.update(adjudication_sha256=runner.digest(adjudication)))
    with pytest.raises(ValueError, match=message):
        runner.check_development_gate(gate, manifest, data)


def test_adjudication_file_drift_is_rejected(tmp_path, monkeypatch):
    """裁决文件被改过（哈希不符）⇒ 拒。"""
    data, gate, manifest, adjudication = _fixture(tmp_path, monkeypatch)
    _patch(adjudication, lambda a: a.update(decision="tampered"))
    with pytest.raises(ValueError, match="Adjudication evidence drift"):
        runner.check_development_gate(gate, manifest, data)


@pytest.mark.parametrize("field", ["unsupported_cognitive_claim", "severe_false_accusation",
                                   "invalid_displayed_reference",
                                   "serious_physics_contradiction", "ratable"])
def test_unresolved_gating_fields_block_the_gate(tmp_path, monkeypatch, field):
    """**门槛判据用到的字段一个都不许残留未裁决。**"""
    data, gate, manifest, _ = _fixture(tmp_path, monkeypatch)
    _patch(gate, lambda g: g["unresolved_disagreement_fields"].update({field: ["Q001"]}))
    with pytest.raises(ValueError, match="Unresolved disagreements include gating fields"):
        runner.check_development_gate(gate, manifest, data)


def test_unresolved_score_fields_do_not_block_the_gate(tmp_path, monkeypatch):
    """评分分歧**不进门槛判据** ⇒ 允许残留（但必须是 `*_1to5`）。"""
    data, gate, manifest, _ = _fixture(tmp_path, monkeypatch)
    _patch(gate, lambda g: g["unresolved_disagreement_fields"].update(
        {"targeting_1to5": ["Q002"], "actionability_1to5": ["Q003"]}))
    runner.check_development_gate(gate, manifest, data)


def test_a_review_not_covered_by_the_adjudication_is_rejected(tmp_path, monkeypatch):
    """初评必须是**裁决所针对的那一份** —— 换一份没登记过的评分文件就拒。"""
    data, gate, manifest, _ = _fixture(tmp_path, monkeypatch)
    stranger = tmp_path / "stranger.json"
    runner.write(stranger, {"reviewer_id": "ai_x", "items": []})
    # ⚠️ 改的是 reviewer_0.json，就要更新 review_reports[0]（我第一版改错了份，测试报的是 drift）
    report = tmp_path / "reviewer_0.json"
    _patch(report, lambda r: r.update(initial_ratings_sha256=runner.digest(stranger)))
    _patch(gate, lambda g: g["review_reports"][0].update(sha256=runner.digest(report)))
    with pytest.raises(ValueError, match="does not cover this review's locked ratings"):
        runner.check_development_gate(gate, manifest, data)
