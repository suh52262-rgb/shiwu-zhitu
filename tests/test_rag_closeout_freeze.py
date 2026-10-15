"""Hard failures prevent malformed development cases from silently disappearing."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rag_close_freeze", Path(__file__).resolve().parents[1]
                                            / "scripts/freeze_rag_closeout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def data():
    return {"status": "AI_REVIEWED_DEVELOPMENT", "drafts": [{"id": "one", "query": "why",
            "cell_key": "em|normal|single", "required_answer_points": ["fact"],
            "review_provenance": "independent AI review", "acceptable_evidence_sets": [
                {"all_of": ["doc:p1:b2"], "all_of_block_ids": ["id"], "supports": ["要点1"]}]}]}


def test_ledger_reports_actual_gaps_without_claiming_24():
    result = module.validate(data())
    assert result["n_items"] == 1
    assert len(result["empty_cells"]) == 7
    assert sum(result["target_24_remaining"].values()) == 23


@pytest.mark.parametrize("mutation", [
    lambda d: d["drafts"][0].pop("query"),
    lambda d: d["drafts"].append(deepcopy(d["drafts"][0])),
    lambda d: d["drafts"][0]["acceptable_evidence_sets"][0]["all_of_block_ids"].clear(),
    lambda d: d["drafts"][0].update(cell_key="em|normal|comb"),
    lambda d: d["drafts"][0].pop("review_provenance"),
    lambda d: d.update(status="DRAFT_PENDING_REVIEW"),
    lambda d: d["drafts"][0]["required_answer_points"].append("unsupported second point"),
])
def test_invalid_inputs_fail(mutation):
    value = data()
    mutation(value)
    with pytest.raises(ValueError):
        module.validate(value)
