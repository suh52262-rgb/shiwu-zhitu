"""Aggregate two locked AI initial reviews without adjudicating disagreements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t4_batch import ROOT, digest, evidence_digest, write

FLAGS = {
    "severe_false_accusation": "severe_false_accusations",
    "unsupported_cognitive_claim": "unsupported_cognitive_claims",
    "invalid_displayed_reference": "invalid_displayed_references",
    "serious_physics_contradiction": "serious_physics_contradictions",
}
SCORES = ("correctness_1to5", "targeting_1to5", "actionability_1to5", "answer_leakage_1to5")
BOOLEAN_FIELDS = (*FLAGS, "ratable", "material_has_definite_error", "explicit_error_detected")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def locked_review(path, bindings, expected_ids):
    review = read(path)
    return validate_locked_review(review, bindings, expected_ids)


def validate_locked_review(review, bindings, expected_ids):
    if (not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip()
            or review.get("provenance") != "independent_ai_review"
            or review.get("initial_ratings_locked") is not True):
        raise ValueError("Require named, locked independent AI initial ratings")
    if any(review.get(k) != v for k, v in bindings.items()):
        raise ValueError("Initial ratings do not bind unchanged batch evidence")
    items = review.get("items", [])
    if len(items) != 72 or {i.get("blind_item_id") for i in items} != expected_ids:
        raise ValueError("Initial ratings must contain each of 72 blind IDs exactly once")
    for item in items:
        if any(type(item.get(k)) is not bool for k in BOOLEAN_FIELDS):
            raise ValueError("Rating decisions must be explicit booleans")
        for key in SCORES:
            score = item.get(key)
            if item["ratable"]:
                if type(score) is not int or not 1 <= score <= 5:
                    raise ValueError("Ratable scores require integers 1 through 5")
            elif score is not None:
                raise ValueError("Unratable scores must be null, never zero-filled")
        if not item.get("evidence"):
            raise ValueError("Each rating requires an evidence explanation")
    return review


def metrics(items, mapping, records):
    result = {plural: sum(i[flag] for i in items) for flag, plural in FLAGS.items()}
    explicit = [i for i in items if mapping[i["blind_item_id"]]["category"] == "explicit_error"]
    result.update(explicit_error_total=len(explicit), explicit_error_recognized=sum(
        i["ratable"] and i["material_has_definite_error"] and i["explicit_error_detected"] for i in explicit))
    result["recorded"] = len(items)
    result["succeeded"] = sum(records[(mapping[i["blind_item_id"]]["case_id"],
                                      mapping[i["blind_item_id"]]["repeat"])]["status"] == "succeeded" for i in items)
    result["failed"] = len(items) - result["succeeded"]
    result["ratable"] = sum(i["ratable"] for i in items)
    result["unratable"] = len(items) - result["ratable"]
    result["mean_scores_ratable_only"] = {
        key: sum(i[key] for i in items if i["ratable"]) / result["ratable"]
        if result["ratable"] else None for key in SCORES}
    return result


def aggregate(batch, manifest_path, ratings_a, ratings_b, out):
    batch, manifest_path, out = map(lambda p: Path(p).resolve(), (batch, manifest_path, out))
    if out.exists():
        raise ValueError("Output exists; initial ratings and aggregates are append-only")
    if not batch.is_relative_to(ROOT) or not out.is_relative_to(ROOT):
        raise ValueError("Batch and report output must be within workspace")
    summary_path = batch / "summary.json"
    summary, manifest = read(summary_path), read(manifest_path)
    bindings = {"manifest_sha256": digest(manifest_path), "summary_sha256": digest(summary_path),
                "evidence_sha256": evidence_digest(batch)}
    if (summary.get("dry_run") is not False or summary.get("smoke") is not False
            or summary.get("records") != 72 or summary.get("evidence_sha256") != bindings["evidence_sha256"]):
        raise ValueError("Require complete actual development batch with unchanged raw evidence")
    expected_ids = {f"Q{i:03d}" for i in range(1, 73)}
    # Deliberately load both locked reviews before opening restricted identities.
    reviews = [locked_review(p, bindings, expected_ids) for p in (ratings_a, ratings_b)]
    if reviews[0]["reviewer_id"] == reviews[1]["reviewer_id"]:
        raise ValueError("Require two distinct independent reviewers")
    mapping = read(batch / "restricted_mapping.json")
    records_list = [read(p) for p in (batch / "runs").glob("*.json")]
    dataset_path = ROOT / manifest["dataset"]
    if digest(dataset_path) != manifest.get("files", {}).get(manifest["dataset"]):
        raise ValueError("Frozen dataset drift")
    dataset = read(dataset_path)
    categories = {c["id"]: c["category"] for p in dataset["problems"] if p["split"] == "development"
                  for c in p["cases"]}
    expected = {(case_id, repeat) for case_id in categories for repeat in range(1, 4)}
    records = {(r["case_id"], r["repeat"]): r for r in records_list}
    if (len(records_list) != 72 or set(records) != expected or set(mapping) != expected_ids
            or {(m["case_id"], m["repeat"]) for m in mapping.values()} != expected):
        raise ValueError("Require complete unique development records and blind mapping")
    for item in [*records_list, *mapping.values()]:
        if item["split"] != "development" or item["category"] != categories[item["case_id"]]:
            raise ValueError("Development identity/category mismatch")
    if any(r.get("manifest_sha256") != bindings["manifest_sha256"] or r.get("status") not in
           {"failed", "succeeded"} for r in records_list):
        raise ValueError("Record manifest/status mismatch")
    succeeded = sum(r["status"] == "succeeded" for r in records_list)
    if summary.get("succeeded") != succeeded or summary.get("failed") != 72 - succeeded:
        raise ValueError("Summary availability disagrees with raw records")
    for review in reviews:
        for item in review["items"]:
            identity = mapping[item["blind_item_id"]]
            record = records[(identity["case_id"], identity["repeat"])]
            if record["status"] == "failed" and item["ratable"]:
                raise ValueError("Failed workflows must be unratable and excluded from means")
    by_id = [{i["blind_item_id"]: i for i in r["items"]} for r in reviews]
    disagreements = []
    for blind_id in sorted(expected_ids):
        a, b = (r[blind_id] for r in by_id)
        differing = [k for k in BOOLEAN_FIELDS if a[k] != b[k]]
        differing += [k for k in SCORES if a["ratable"] and b["ratable"] and abs(a[k] - b[k]) >= 2]
        if differing:
            disagreements.append({"blind_item_id": blind_id, "fields": differing})
    # Validate and compute everything before creating the immutable output directory.
    reports = []
    for review in reviews:
        totals = metrics(review["items"], mapping, records)
        passed = (totals["succeeded"] / 72 >= .9 and totals["explicit_error_total"] == 12
                  and totals["explicit_error_recognized"] >= 10 and all(totals[k] == 0 for k in FLAGS.values()))
        reports.append({**bindings, "reviewer_id": review["reviewer_id"],
                        "provenance": "independent_ai_review", "initial_ratings_locked": True,
                        "blind_item_ids": sorted(expected_ids), "verdict": "accept" if passed else "reject",
                        "metrics": totals, "by_category": {
                            category: metrics([i for i in review["items"] if
                                mapping[i["blind_item_id"]]["category"] == category], mapping, records)
                            for category in sorted(set(categories.values()))}})
    if digest(summary_path) != bindings["summary_sha256"] or evidence_digest(batch) != bindings["evidence_sha256"]:
        raise ValueError("Batch changed during aggregation")
    out.mkdir(parents=True)
    entries = []
    for n, (source, report) in enumerate(zip((ratings_a, ratings_b), reports), 1):
        initial = out / f"ratings_{n}_initial.json"
        shutil.copyfile(source, initial)
        report["initial_ratings_sha256"] = digest(initial)
        report["initial_ratings_path"] = str(initial.relative_to(ROOT))
        target = out / f"reviewer_{n}_report.json"
        write(target, report)
        entries.append({"path": str(target.relative_to(ROOT)), "sha256": digest(target)})
    gate = {**bindings, "batch_dir": str(batch.relative_to(ROOT)), "review_reports": entries,
            "passed": all(r["verdict"] == "accept" for r in reports) and not disagreements,
            "adjudication_required": bool(disagreements), "disagreements": disagreements,
            "claim": "AI evaluation only; engineering gate, not teacher validation or population accuracy",
            "holdout_started": False}
    write(out / "development_gate.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("batch", "manifest", "ratings-a", "ratings-b", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    gate = aggregate(args.batch, args.manifest, args.ratings_a, args.ratings_b, args.out)
    print(json.dumps({"passed": gate["passed"], "adjudication_required": gate["adjudication_required"]}))


if __name__ == "__main__":
    main()
