"""Graph governance and version binding use temporary files and pure state only."""
import json
import pytest

from app.models.problem_graph import ProblemGraph
from app.models.student import StudentState
from app.services.graph_binding import bind_graph
from app.services.graph_versions import GraphVersions, graph_hash
from app.services.problem_repository import ProblemRepository
from app.services.attempt_store import AttemptStore


def graph(name="first"):
    return ProblemGraph(problem_id="p", nodes=[{"id": "S1", "name": name, "type": "result"}], edges=[])


def test_hash_is_canonical_and_review_is_separate():
    first = graph()
    changed = first.model_copy(deep=True)
    changed.metadata.update(needs_review=False, reviewed_by="human", reviewed_at="later")
    assert graph_hash(first) == graph_hash(changed)
    assert graph_hash(graph("different")) != graph_hash(first)


def test_draft_review_publish_and_update_keep_published_snapshot(tmp_path):
    repo = ProblemRepository(tmp_path)
    repo.save_graph("p", graph())
    assert repo.graph_release("p")["graph_status"] == "draft"
    assert repo.get_teaching_graph("p") is None
    with pytest.raises(ValueError, match="not been reviewed"):
        repo.publish_graph("p", published_by="teacher")
    with pytest.raises(ValueError, match="reviewer"):
        repo.mark_reviewed("p")
    repo.mark_reviewed("p", reviewed_by="teacher", note="Checked routes")
    assert repo.graph_release("p")["graph_status"] == "reviewed"
    first_hash = repo.publish_graph("p", published_by="teacher")
    assert repo.graph_release("p")["graph_version"] == first_hash
    repo.save_graph("p", graph("new draft"))
    assert repo.get_teaching_graph("p").nodes[0].name == "first"
    with pytest.raises(ValueError, match="not been reviewed"):
        repo.publish_graph("p", published_by="teacher")
    manifest = GraphVersions(repo.problem_dir("p")).manifest()
    assert manifest["reviews"][first_hash]["hash"] == first_hash
    assert manifest["reviews"][first_hash]["reviewer"] == "teacher"
    assert manifest["reviews"][first_hash]["date"]


def test_legacy_policy_is_explicit_and_does_not_write(tmp_path, monkeypatch):
    path = tmp_path / "p" / "graph.json"
    path.parent.mkdir()
    path.write_text(graph().model_dump_json(), encoding="utf-8")
    repo = ProblemRepository(tmp_path)
    monkeypatch.setenv("GRAPH_RELEASE_POLICY", "legacy-demo")
    assert repo.graph_release("p")["graph_status"] == "legacy_unreviewed"
    assert repo.get_teaching_graph("p")
    assert not (path.parent / ".graph_versions").exists()
    monkeypatch.setenv("GRAPH_RELEASE_POLICY", "published-only")
    assert repo.get_teaching_graph("p") is None
    assert repo.graph_release("p")["graph_status"] == "legacy_unreviewed"


def test_snapshot_tampering_is_detected(tmp_path):
    versions = GraphVersions(tmp_path / "p")
    version = versions.snapshot(graph())
    path = versions.root / (version + ".json")
    path.write_text(graph("tampered").model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        versions.load(version)
    with pytest.raises(ValueError, match="hash mismatch"):
        versions.snapshot(graph())


def test_attempt_stays_pinned_and_graphless_upgrade_has_event():
    state = StudentState(problem_id="p", graph_binding_status="unbound")
    assert bind_graph(state, None) is None
    bind_graph(state, graph())
    assert state.timeline[-1].event_type == "graph_mode_transition"
    first_hash = state.graph_version
    assert bind_graph(state, graph("new")).nodes[0].name == "first"
    assert state.graph_version == first_hash
    restored = StudentState.model_validate_json(state.model_dump_json())
    assert bind_graph(restored, None).nodes[0].name == "first"
    restored.graph_snapshot["nodes"][0]["name"] = "corrupt"
    with pytest.raises(ValueError, match="corrupt"):
        bind_graph(restored, None)


def test_old_attempt_does_not_fabricate_version_or_reuse_nodes():
    state = StudentState(problem_id="p", current_node="old", visited_nodes=["old"], node_errors={"old": 2})
    assert bind_graph(state, graph()) is None
    assert state.graph_binding_status == "unknown"
    assert not state.graph_version
    assert not state.current_node and not state.visited_nodes and not state.node_errors


def test_new_attempt_is_distinct_from_old_unknown_state():
    record = AttemptStore(persistent=False).create_attempt("s", "p")
    assert record.state.graph_binding_status == "unbound"
    assert StudentState().graph_binding_status == "unknown"


def test_valid_hash_cannot_bind_another_problem():
    state = StudentState(problem_id="other", graph_binding_status="unbound")
    with pytest.raises(ValueError, match="another problem"):
        bind_graph(state, graph())
    state.graph_binding_status = "bound"
    state.graph_snapshot = graph().model_dump()
    state.graph_version = graph_hash(graph())
    with pytest.raises(ValueError, match="another problem"):
        bind_graph(state, None)


def test_valid_hash_cannot_load_another_problem_snapshot(tmp_path):
    versions = GraphVersions(tmp_path / "other")
    versions.root.mkdir(parents=True)
    version = graph_hash(graph())
    (versions.root / (version + ".json")).write_text(graph().model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="problem_id"):
        versions.load(version)
    with pytest.raises(ValueError, match="problem_id"):
        versions.snapshot(graph())


def test_cli_diff_review_publish(tmp_path, capsys):
    from scripts.graph_release import main
    repo = ProblemRepository(tmp_path)
    repo.save_graph("p", graph())
    assert main(["p", "diff", "--root", str(tmp_path)]) == 0
    assert "unpublished" in capsys.readouterr().out
    assert main(["p", "review", "--root", str(tmp_path), "--reviewer", "human"]) == 0
    assert main(["p", "publish", "--root", str(tmp_path), "--reviewer", "human"]) == 0
    assert repo.graph_release("p")["graph_status"] == "published"


def test_parallel_release_records_are_not_lost(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    versions = GraphVersions(tmp_path / "p")
    graphs = [graph(str(index)) for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: versions.review(item, "human"), graphs))
    assert len(versions.manifest()["reviews"]) == len(graphs)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: versions.publish(item, "human"), graphs))
    assert len(versions.manifest()["releases"]) == len(graphs)


def test_corrupt_release_becomes_repository_data_error(tmp_path):
    from app.services.problem_repository import ProblemDataError
    repo = ProblemRepository(tmp_path)
    repo.save_graph("p", graph())
    versions = GraphVersions(repo.problem_dir("p"))
    versions.manifest_path.write_text("{", encoding="utf-8")
    with pytest.raises(ProblemDataError, match="release invalid"):
        repo.graph_release("p")


@pytest.mark.parametrize("corruption", ["unreviewed", "missing_release", "review_identity", "review_hash", "review_date", "release_identity"])
def test_published_policy_rejects_incomplete_governance(tmp_path, monkeypatch, corruption):
    from app.services.problem_repository import ProblemDataError
    from app.services.graph_versions import atomic_json
    repo = ProblemRepository(tmp_path)
    repo.save_graph("p", graph())
    repo.mark_reviewed("p", reviewed_by="human")
    version = repo.publish_graph("p", published_by="human")
    versions = GraphVersions(repo.problem_dir("p"))
    manifest = versions.manifest()
    if corruption == "unreviewed":
        manifest["reviews"] = {}
    elif corruption == "missing_release":
        manifest["releases"] = []
    elif corruption == "review_identity":
        manifest["reviews"][version]["reviewer"] = "unknown"
    elif corruption == "review_hash":
        manifest["reviews"][version]["hash"] = "different"
    elif corruption == "review_date":
        manifest["reviews"][version]["date"] = "not-a-date"
    elif corruption == "release_identity":
        manifest["releases"][0]["publisher"] = ""
    atomic_json(versions.manifest_path, manifest)
    monkeypatch.setenv("GRAPH_RELEASE_POLICY", "published-only")
    with pytest.raises(ProblemDataError, match="release invalid"):
        repo.graph_release("p")


def test_release_policy_can_come_from_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app import config
    monkeypatch.delenv("GRAPH_RELEASE_POLICY", raising=False)
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(graph_release_policy="published-only"))
    repo = ProblemRepository(tmp_path)
    path = tmp_path / "p" / "graph.json"
    path.parent.mkdir()
    path.write_text(graph().model_dump_json(), encoding="utf-8")
    assert repo.graph_release("p")["graph_policy"] == "published-only"
    assert repo.get_teaching_graph("p") is None


@pytest.fixture
def binding_api(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.api import attempts
    from app.db import persistence
    from app.main import app
    monkeypatch.setenv("PROBLEMS_DIR", str(tmp_path))
    monkeypatch.setenv("GRAPH_RELEASE_POLICY", "legacy-demo")
    directory = tmp_path / "p"
    directory.mkdir()
    (directory / "problem.json").write_text(json.dumps({"problem_id": "p", "title": "probe", "statement": "probe"}), encoding="utf-8")
    traces = []
    seen = []
    def invoke(state, config):
        seen.append(config["configurable"]["problem_graph"])
        return {**state, "current_step": "step", "finished": False, "hint": "", "hint_level": 0}
    monkeypatch.setattr(attempts, "tutor_graph", SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(persistence, "persist_trace", lambda *args, **kwargs: traces.append(kwargs) or True)
    for name in ("persist_events", "persist_hint", "persist_profile_weak_concepts", "resolve_pending_hints"):
        monkeypatch.setattr(persistence, name, lambda *args, **kwargs: False)
    return TestClient(app), directory, traces, seen


def test_api_pins_snapshot_and_trace_actual_version(binding_api, isolated_attempt_store):
    client, directory, traces, seen = binding_api
    (directory / "graph.json").write_text(graph().model_dump_json(), encoding="utf-8")
    payload = {"student_id": "s", "problem_id": "p", "answer": "probe"}
    first = client.post("/attempt/analyze", json=payload)
    assert first.status_code == 200, first.text
    first = first.json()
    assert first["graph_version"] == graph_hash(graph())
    assert first["graph_binding_status"] == "bound"
    assert first["graph_release_status"] == "legacy_unreviewed"
    (directory / "graph.json").write_text(graph("new").model_dump_json(), encoding="utf-8")
    second = client.post("/attempt/analyze", json={**payload, "attempt_id": first["attempt_id"]})
    assert second.status_code == 200, second.text
    assert second.json()["graph_version"] == first["graph_version"]
    assert seen[-1].nodes[0].name == "first"
    assert traces[-1]["payload"]["graph_version"] == first["graph_version"]
    assert isolated_attempt_store.load_attempt(first["attempt_id"]).state.graph_snapshot["nodes"][0]["name"] == "first"


def test_api_graphless_transition_survives_model_failure(binding_api, monkeypatch, isolated_attempt_store):
    from app.api import attempts
    client, directory, traces, seen = binding_api
    payload = {"student_id": "s", "problem_id": "p", "answer": "probe"}
    first = client.post("/attempt/analyze", json=payload).json()
    aid = first["attempt_id"]
    from datetime import datetime, timedelta, timezone
    isolated_attempt_store.update_attempt(aid, started_at=datetime.now(timezone.utc) - timedelta(seconds=120))
    (directory / "graph.json").write_text(graph().model_dump_json(), encoding="utf-8")
    original = attempts.tutor_graph.invoke
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic model failure")
    monkeypatch.setattr(attempts.tutor_graph, "invoke", fail)
    failed = client.post("/attempt/analyze", json={**payload, "attempt_id": aid})
    assert failed.status_code == 503
    state = isolated_attempt_store.load_attempt(aid).state
    assert state.graph_version == graph_hash(graph())
    assert sum(event.event_type == "graph_mode_transition" for event in state.timeline) == 1
    assert next(event.timestamp for event in state.timeline if event.event_type == "graph_mode_transition") >= 120
    assert traces[-1]["payload"]["graph_version"] == state.graph_version
    monkeypatch.setattr(attempts.tutor_graph, "invoke", original)
    retry = client.post("/attempt/analyze", json={**payload, "attempt_id": aid})
    assert retry.status_code == 200, retry.text
    state = isolated_attempt_store.load_attempt(aid).state
    assert sum(event.event_type == "graph_mode_transition" for event in state.timeline) == 1


def test_api_refuses_valid_hash_for_wrong_problem(binding_api, isolated_attempt_store):
    client, directory, traces, seen = binding_api
    (directory / "graph.json").write_text(graph().model_dump_json(), encoding="utf-8")
    payload = {"student_id": "s", "problem_id": "p", "answer": "probe"}
    first = client.post("/attempt/analyze", json=payload).json()
    state = isolated_attempt_store.load_attempt(first["attempt_id"]).state
    foreign = graph()
    foreign.problem_id = "foreign"
    state.graph_snapshot = foreign.model_dump()
    state.graph_version = graph_hash(foreign)
    isolated_attempt_store.update_attempt(first["attempt_id"], state=state.model_dump())
    response = client.post("/attempt/analyze", json={**payload, "attempt_id": first["attempt_id"]})
    assert response.status_code == 409
