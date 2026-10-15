"""Pin the graph actually used by one attempt, without rewriting history."""
from app.models.problem_graph import ProblemGraph
from app.models.student import StudentState, TimelineEvent
from app.services.graph_versions import graph_hash, graph_content


def bind_graph(state: StudentState, candidate: ProblemGraph | None, release_status: str = "legacy_unreviewed") -> ProblemGraph | None:
    if state.graph_binding_status == "bound":
        if not state.graph_snapshot or graph_hash(state.graph_snapshot) != state.graph_version:
            raise ValueError("Attempt graph snapshot is missing or corrupt; restart explicitly")
        if state.graph_snapshot.get("problem_id") != state.problem_id:
            raise ValueError("Attempt graph snapshot belongs to another problem")
        return ProblemGraph(**state.graph_snapshot)
    if state.graph_binding_status == "unknown":
        # Existing state cannot establish which graph produced historical nodes.
        state.current_node = ""
        state.visited_nodes = []
        state.node_attempts = {}
        state.node_errors = {}
        state.last_step_match = {}
        state.last_hint_node = ""
        state.graph_mode = False
        return None
    if candidate is None:
        state.graph_binding_status = "graphless"
        state.graph_release_status = "graphless"
        return None
    if candidate.problem_id != state.problem_id:
        raise ValueError("Candidate graph belongs to another problem")
    previous = state.graph_binding_status
    state.graph_snapshot = graph_content(candidate)
    state.graph_version = graph_hash(candidate)
    state.graph_binding_status = "bound"
    state.graph_release_status = release_status
    if previous == "graphless":
        state.timeline.append(TimelineEvent(event_type="graph_mode_transition", content="graphless -> graph; hash=" + state.graph_version))
    return ProblemGraph(**state.graph_snapshot)
