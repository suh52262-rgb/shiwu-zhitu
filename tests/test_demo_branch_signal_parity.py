"""Alternative methods get the same key-node retry weight."""
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

from app.models.problem_graph import ProblemGraph
from app.models.student import StudentState
from app.services.stuck_detector import (
    StuckSignals, build_graph_signals, compute_graph_aware_stuck_score,
)


@pytest.mark.parametrize("pid,left,right", [
    ("demo_03", "S2a", "S2b"), ("demo_04", "S5a", "S5b"),
])
@pytest.mark.parametrize("attempts,errors", [(2, 0), (3, 2)])
def test_alternative_calculation_nodes_have_equal_retry_signals(pid, left, right, attempts, errors):
    path = PROBLEMS / pid / "graph.json"
    graph = ProblemGraph.model_validate_json(path.read_text(encoding="utf-8"))
    scores = []
    for node in (left, right):
        state = StudentState(student_id="parity", problem_id=pid, current_node=node,
                             node_attempts={node: attempts}, node_errors={node: errors})
        signals = build_graph_signals(graph, state)
        assert signals.is_key_node
        scores.append(compute_graph_aware_stuck_score(StuckSignals(), signals))
    assert scores[0] == scores[1]
