"""Apply bounded, independently reviewed repairs to a NEW development version.

Inputs remain unchanged. This is data assembly, not retrieval or scoring. The
original seven cases retain their prior semantic reviews; narrow repairs are
identified separately rather than described as seven new blind reviews.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evaluation/datasets/annotation_recheck24_batch7_v5.json"
CANDIDATES = ROOT / "evaluation/datasets/rag_close_gap_candidates_20261011.json"
REVIEW = ROOT / "evaluation/datasets/rag_close_gap_independent_review_20261011.json"
REPAIR = ROOT / "evaluation/datasets/rag_close_primary_groups_independent_review_20261011.json"
OUT = ROOT / "evaluation/datasets/rag_closeout_v4_20261011.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt(record, quote=None):
    return {"ref": record["ref"], "block_id": str(record.get("block_id") or record["id"]),
            "sha256": record["sha256"], "excerpt_verbatim": quote or record["text"],
            "type": record["type"], "indexable": record["indexable"]}


def main():
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite a reviewed data version")
    old, candidates, review, repair = map(load, (BASE, CANDIDATES, REVIEW, REPAIR))
    if sha(BASE) != repair["source_sha256"] or sha(CANDIDATES) != review["candidate_sha256"]:
        raise ValueError("Review identity mismatch")
    records = {r["ref"]: r for r in repair["added_source_records"]}
    records.update({r["ref"]: r for r in review["supplementary_evidence"]})
    repairs = {r["id"]: r for r in repair["reviews"]}
    output = []
    for original in old["drafts"]:
        case = {k: deepcopy(original[k]) for k in ("id", "query", "cell_key", "premise_status",
                                                  "premise_class", "required_answer_points") if k in original}
        groups = deepcopy(original["acceptable_evidence_sets"])
        selected = groups if "two_lamp" in case["id"] else groups[:1]
        refs, ids, excerpts = [], [], []
        for group in selected:
            for ref, bid in zip(group["all_of"], group["all_of_block_ids"]):
                if ref not in refs:
                    refs.append(ref)
                    ids.append(bid)
            for e in group["excerpts"]:
                if e not in excerpts:
                    excerpts.append(e)
        for ref in repairs.get(case["id"], {}).get("required_primary_add", []):
            r = records[ref]
            refs.append(ref)
            ids.append(r["id"])
            excerpts.append(excerpt(r))
        if "kirchhoff" in case["id"]:
            case["required_answer_points"][4] = case["required_answer_points"][4].replace(
                "b32 不在证据组内、只作续块登记", "收尾 b32 已在 closeout-v4 纳入主证据组")
        case["acceptable_evidence_sets"] = [{"all_of": refs, "all_of_block_ids": ids,
            "supports": [f"要点{i}" for i in range(1, len(case["required_answer_points"]) + 1)],
            "excerpts": excerpts, "minimum_size": "not_proven",
            "why": "Prior reviewed main group plus explicit source repairs; partial support is never a full OR alternative."}]
        case["supplementary_evidence_sets"] = groups[1:] if "two_lamp" not in case["id"] else []
        case["review_provenance"] = {
            "original": "evaluation/datasets/annotation_recheck24_batch7_v4_reviewed.json",
            "narrow_repairs": str(REPAIR.relative_to(ROOT)).replace("\\", "/"),
            "scope": "prior full review retained; current independent review covers stated repairs only"}
        case["origin"] = {"file": str(BASE.relative_to(ROOT)), "sha256": sha(BASE), "id": case["id"]}
        if "fermat" in case["id"]:
            case["formula_scope"] = "Integral endpoints missing in b30 transcription are not scored or reconstructed. b29 sum and b30 integrand/path supported by original scan and b2/b8."
        output.append(case)
    for original in candidates["cases"]:
        ruling = next(r for r in review["reviews"] if r["id"] == original["id"])
        case = deepcopy(original)
        case["cell_key"] = case.pop("cell")
        for change in ruling["required_changes"]:
            field = change["field"]
            if field == "query":
                case["query"] = change["value"]
            elif field == "required_answer_points[0]":
                case["required_answer_points"][0] = change["value"]
            elif field == "acceptable_evidence_sets[0].all_of":
                case["acceptable_evidence_sets"][0]["all_of"] = change["value"]
                case["evidence"].append(deepcopy(records[change["value"][1]]))
                case["point_support"].append({"point_index": 1, "ref": change["value"][1],
                                              "quote": change["additional_quote"]})
            else:
                raise ValueError(f"Unimplemented review change: {field}")
        evidence = {r["ref"]: r for r in case["evidence"]}
        for group in case["acceptable_evidence_sets"]:
            group["all_of_block_ids"] = [str(evidence[r].get("block_id") or evidence[r]["id"])
                                         for r in group["all_of"]]
            group["supports"] = [f"要点{i}" for i in range(1, len(case["required_answer_points"]) + 1)]
            group["excerpts"] = [excerpt(evidence[s["ref"]], s["quote"]) for s in case["point_support"]]
            group["sufficiency_claim"] = "independent AI review accepted after exact stated revisions"
        case["status"] = "AI_REVIEWED_DEVELOPMENT"
        case["review_provenance"] = {"file": str(REVIEW.relative_to(ROOT)), "sha256": sha(REVIEW),
                                     "decision": ruling["decision"], "applied_exact_changes": ruling["required_changes"]}
        case["origin"] = {"file": str(CANDIDATES.relative_to(ROOT)), "sha256": sha(CANDIDATES)}
        output.append(case)
    data = {"status": "AI_REVIEWED_DEVELOPMENT", "annotation_version": "rag-closeout-v4-20261011",
            "drafts": output, "claim": "11 AI-reviewed development probes across eight cells; not 24 cases or a holdout.",
            "changes": ["four new source-reviewed probes", "partial OR groups moved to supplemental or merged",
                        "Kirchhoff zero-sum and Fermat optical-path evidence restored"],
            "old_scores_not_imported": True, "prior_frozen_files_unchanged": True,
            "held": ["recheck24_em_miscon_single_m"],
            "scope": "Data/tool closeout; no business/default changes; no student/teacher study"}
    from freeze_rag_closeout import validate
    ledger = validate(data)
    data["cell_ledger"] = ledger
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(ledger, ensure_ascii=False))


if __name__ == "__main__":
    main()
