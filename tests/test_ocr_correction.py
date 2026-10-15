"""Correction replaces the latest media projection while retaining evidence."""
import pytest

from app.api import attempts
from app.db import persistence
from app.services import ocr_correction
from test_media_lifecycle_integration import media_client, BASE


@pytest.fixture
def correction_client(media_client, monkeypatch, isolated_attempt_store):
    effects = []
    for name in ("persist_events", "persist_hint", "persist_trace", "supersede_media_hints"):
        monkeypatch.setattr(persistence, name, lambda *a, _name=name, **kw: effects.append((_name, a, kw)))
    monkeypatch.setattr(persistence, "persist_profile_weak_concepts",
                        lambda *a, **kw: effects.append(("profile", a, kw)) or True)

    def invoke(state, config):
        text = state["current_answer"]
        if text == "FAIL":
            raise RuntimeError("test graph failure")
        return {**state, "has_obvious_error": text != "CORRECT", "diagnosed_issue": "ocr-error" if text != "CORRECT" else "",
                "weak_concepts": ["wrong-ocr"] if text != "CORRECT" else [],
                "hint": "check text" if text != "CORRECT" else "", "hint_level": 1,
                "node_attempts": {"S1": state.get("node_attempts", {}).get("S1", 0) + 1},
                "finished": text == "DONE", "degraded_reasons": ["test"] if text == "DEGRADED" else []}
    monkeypatch.setattr(attempts.tutor_graph, "invoke", invoke)
    return media_client, isolated_attempt_store, effects


def upload(client, *, video=False):
    result = client.post("/video/analyze" if video else "/video/analyze_image", data=BASE,
                         files={"file": ("work.mp4" if video else "work.png", b"test")})
    assert result.status_code == 200, result.text
    data = result.json()
    return data, {**BASE, "attempt_id": data["attempt_id"],
                  "source_id": data["rounds"][-1]["source_id"], "base_revision": 0}


@pytest.mark.parametrize("video", [False, True])
def test_latest_media_replaces_counts_and_keeps_original(correction_client, video):
    client, store, effects = correction_client
    result, req = upload(client, video=video)
    before = store.load_attempt(req["attempt_id"])
    errors = before.error_count
    calls_before = len(effects)
    response = client.post("/video/correct_ocr", json={**req, "corrected_text": "CORRECT"})
    assert response.status_code == 200, response.text
    after = store.load_attempt(req["attempt_id"])
    assert after.error_count == errors - 1
    assert after.state.node_attempts == before.state.node_attempts
    assert after.state.current_answer == "CORRECT"
    source = after.state.media_sources[-1]
    assert source["original_text"] == "mgh = mv^2/2"
    assert source["revisions"][0]["analysis"]["issue"] == "ocr-error"
    assert source["revisions"][1]["supersedes"]["revision"] == 0
    trace = next(e[2] for e in reversed(effects) if e[0] == "persist_trace")
    assert trace["request_id"] and trace["problem_id"] == BASE["problem_id"]
    assert "calls" in trace["token_usage"] and trace["prompt_versions"]
    assert "graph_version" in trace["payload"]
    assert "media_sources" not in trace["before_state"]
    assert not any(e[0] == "profile" for e in effects[calls_before:])
    duplicate_calls = len(effects)
    response = client.post("/video/correct_ocr", json={**req, "corrected_text": "CORRECT"})
    assert response.status_code == 200 and response.json()["idempotent"]
    assert len(effects) == duplicate_calls
    response = client.post("/video/correct_ocr", json={**req, "base_revision": 1, "corrected_text": "wrong again"})
    assert response.status_code == 200, response.text
    assert store.load_attempt(req["attempt_id"]).error_count == errors
    assert len(store.load_attempt(req["attempt_id"]).state.media_sources[-1]["revisions"]) == 3
    assert client.post("/video/correct_ocr", json={**req, "corrected_text": "stale"}).status_code == 409


@pytest.mark.parametrize("text", ["FAIL", "DEGRADED"])
def test_failed_recompute_preserves_every_original_field(correction_client, text):
    client, store, effects = correction_client
    _, req = upload(client)
    before = store.load_attempt(req["attempt_id"]).model_dump(mode="json")
    count = len(effects)
    response = client.post("/video/correct_ocr", json={**req, "corrected_text": text})
    assert response.status_code == 503, response.text
    assert store.load_attempt(req["attempt_id"]).model_dump(mode="json") == before
    assert len(effects) == count + 1
    assert effects[-1][0] == "persist_trace"
    assert effects[-1][2]["payload"]["applied"] is False


def test_later_input_and_other_identity_cannot_rewrite(correction_client):
    client, store, effects = correction_client
    _, req = upload(client)
    assert client.post("/video/correct_ocr", json={**req, "student_id": "other", "corrected_text": "CORRECT"}).status_code == 403
    assert client.post("/attempt/analyze", json={**BASE, "attempt_id": req["attempt_id"], "answer": "later"}).status_code == 200
    before = store.load_attempt(req["attempt_id"]).model_dump(mode="json")
    assert client.post("/video/correct_ocr", json={**req, "corrected_text": "CORRECT"}).status_code == 409
    assert store.load_attempt(req["attempt_id"]).model_dump(mode="json") == before
    assert any(e[0] == "profile" for e in effects)


@pytest.mark.parametrize("action", ["finish", "reset", "complete"])
def test_terminal_correction_rejected_and_profile_finalized(correction_client, action):
    client, store, effects = correction_client
    _, req = upload(client)
    if action == "finish":
        assert client.post(f"/attempt/{req['attempt_id']}/finish", json={}).status_code == 200
    elif action == "reset":
        assert client.post("/attempt/reset", json=BASE).status_code == 200
    else:
        assert client.post("/video/correct_ocr", json={**req, "corrected_text": "DONE"}).status_code == 200
    assert client.post("/video/correct_ocr", json={**req, "corrected_text": "CORRECT"}).status_code == 409
    assert any(e[0] == "profile" for e in effects)


def test_request_hint_count_and_mistakes_are_replaced(correction_client):
    client, store, effects = correction_client
    response = client.post("/video/analyze_image", data={**BASE, "request_hint": "true"},
                           files={"file": ("work.png", b"test")})
    data = response.json()
    req = {**BASE, "attempt_id": data["attempt_id"], "source_id": data["rounds"][-1]["source_id"],
           "base_revision": 0, "corrected_text": "CORRECT"}
    assert store.load_attempt(req["attempt_id"]).hint_count == 1
    assert store.load_attempt(req["attempt_id"]).mistakes
    assert client.post("/video/correct_ocr", json=req).status_code == 200
    record = store.load_attempt(req["attempt_id"])
    assert record.hint_count == 1
    assert record.mistakes == []
    assert record.hint_history == []
    # Closing the window must publish the effective corrected evidence only.
    effects.clear()
    assert client.post(f"/attempt/{req['attempt_id']}/finish", json={}).status_code == 200
    assert [event[1][1] for event in effects if event[0] == "profile"] == [[]]


def test_closed_sources_drop_recompute_snapshots(correction_client, monkeypatch):
    from app.api import video
    # Two distinct teaching inputs close the previous correction window.
    # Repeated OCR now intentionally coalesces into one source.
    monkeypatch.setattr(video, "_snapshots_from", lambda *_: [
        {"ts": 0, "text": "mgh"}, {"ts": 20, "text": "mgh=mv^2/2"}])
    client, store, effects = correction_client
    data, req = upload(client, video=True)
    record = store.load_attempt(req["attempt_id"])
    assert len(record.state.media_sources) == 2
    assert "before_record" not in record.state.media_sources[0]
    assert record.state.media_sources[0]["correction_closed"]
    assert "media_sources" not in record.state.media_sources[-1]["before_record"]["state"]
    assert client.post("/video/correct_ocr", json={**req, "source_id": record.state.media_sources[0]["source_id"],
                                                  "corrected_text": "CORRECT"}).status_code == 409


def test_text_api_cannot_forge_media_provenance(correction_client):
    client, store, effects = correction_client
    response = client.post("/attempt/analyze", json={**BASE, "answer": "text", "media_type": "image"})
    assert response.status_code == 200
    assert response.json()["source_id"] == ""
    assert store.load_attempt(response.json()["attempt_id"]).state.media_sources == []


def test_failed_later_submission_does_not_close_correction_window(correction_client):
    client, store, effects = correction_client
    _, req = upload(client)
    effects.clear()
    response = client.post("/attempt/analyze", json={**BASE, "attempt_id": req["attempt_id"], "answer": "FAIL"})
    assert response.status_code == 503
    source = store.load_attempt(req["attempt_id"]).state.media_sources[-1]
    assert not source.get("correction_closed")
    assert "before_record" in source
    assert not any(item[0] == "profile" for item in effects)
    assert client.post("/video/correct_ocr", json={**req, "corrected_text": "CORRECT"}).status_code == 200


def test_correction_timestamp_is_actual_elapsed_time(correction_client, monkeypatch):
    client, store, effects = correction_client
    _, req = upload(client)
    monkeypatch.setattr(attempts, "_elapsed_seconds", lambda record: 123.4)
    response = client.post("/video/correct_ocr", json={**req, "corrected_text": "CORRECT"})
    assert response.status_code == 200
    assert response.json()["timeline"][-1]["timestamp"] == 123.4


@pytest.mark.parametrize("action", ["finish", "reset"])
def test_lifecycle_closure_flags_persist(correction_client, action):
    client, store, effects = correction_client
    _, req = upload(client)
    if action == "finish":
        response = client.post(f"/attempt/{req['attempt_id']}/finish", json={})
    else:
        response = client.post("/attempt/reset", json=BASE)
    assert response.status_code == 200
    source = store.load_attempt(req["attempt_id"]).state.media_sources[-1]
    assert source["correction_closed"] and source["profile_finalized"]
    assert "before_record" not in source
