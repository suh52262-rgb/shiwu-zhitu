"""Known demo fixes preserve alternative routes without claiming human approval."""
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

from app.models.problem_graph import ProblemGraph
from app.services.branch_merge_checker import inspect_branch_merges
from app.services.graph_validator import validate_graph

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("problem_id", ["demo_03", "demo_04"])
def test_revised_merge_preserves_routes_and_does_not_force_method_specific_operation(problem_id):
    before = ProblemGraph(**json.loads((FIXTURE_ROOT / "branch_merges" /
                                       f"{problem_id}_before.json").read_text(encoding="utf-8")))
    after = ProblemGraph(**json.loads((PROBLEMS / problem_id /
                                      "graph.json").read_text(encoding="utf-8")))
    assert inspect_branch_merges(before)["findings"]
    reviewed = inspect_branch_merges(after)
    assert reviewed["findings"] == []
    assert reviewed["unknown_count"] == 1  # No automatic physics approval.
    assert after.valid_paths == before.valid_paths
    assert after.edges == before.edges
    assert after.metadata["needs_review"] is True
    assert after.metadata["revision_provenance"]["human_review_status"] == "pending"
    assert validate_graph(after, use_llm=False).checks["structure"] == "pass"
