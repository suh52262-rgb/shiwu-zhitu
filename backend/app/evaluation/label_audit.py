"""Offline label provenance checks; AI proposals are never reviewed gold."""
from datetime import datetime

from app.evaluation.path_metrics import labeled_node_recall
from app.services.graph_versions import graph_hash


def label_eligibility(case, graph, *, allow_proposed=False):
    annotation = case.get("annotation", {})
    if graph is None:
        return False, "missing_graph"
    if not annotation.get("graph_hash"):
        return False, "missing_graph_hash"
    if annotation.get("graph_hash") != graph_hash(graph):
        return False, "graph_hash_drift"
    if annotation.get("status") == "proposed" and allow_proposed:
        return True, "candidate_only"
    if annotation.get("status") != "reviewed":
        return False, "pending_review"
    if annotation.get("review_source") != "human":
        return False, "missing_human_review"
    reviewer = annotation.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        return False, "missing_reviewer"
    try:
        date = datetime.fromisoformat(annotation["reviewed_at"])
        if date.tzinfo is None:
            raise ValueError("timezone required")
    except (KeyError, TypeError, ValueError):
        return False, "invalid_review_date"
    return True, "reviewed"


def audit_dataset(dataset, load_graph):
    entries = dataset.get("cases") or [dataset]
    counts = {"total": len(entries), "reviewed": 0, "pending_review": 0,
              "unknown": 0, "hash_drift": 0, "candidate_node_labels": 0,
              "reviewed_node_labels": 0, "unresolved_node_labels": 0}
    rows = []
    for index, case in enumerate(entries):
        graph = load_graph(case.get("problem_id", dataset.get("problem_id")))
        eligible, reason = label_eligibility(case, graph)
        candidate, candidate_reason = label_eligibility(case, graph, allow_proposed=True)
        rate = labeled_node_recall(case, set(), graph, dataset.get("methods"))
        status = case.get("annotation", {}).get("status")
        counts["reviewed"] += eligible
        counts["pending_review"] += status == "proposed"
        counts["unknown"] += status not in {"proposed", "reviewed"}
        counts["hash_drift"] += reason == "graph_hash_drift"
        counts["candidate_node_labels"] += candidate and rate.computable
        counts["reviewed_node_labels"] += eligible and rate.computable
        counts["unresolved_node_labels"] += not rate.computable
        rows.append({"id": case.get("id", f"final:{index}"), "eligible": eligible,
                     "reason": reason, "candidate_eligible": candidate,
                     "candidate_reason": candidate_reason,
                     "node_label_valid": rate.computable,
                     "node_label_note": rate.note,
                     "annotation": case.get("annotation", {})})
    return {"counts": counts, "cases": rows}
