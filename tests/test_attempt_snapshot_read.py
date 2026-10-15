"""A degraded session must remain readable when switching away and back."""
import pytest
from fastapi.testclient import TestClient

from app.db import persistence
from app.main import app
from app.models.student import TimelineEvent


@pytest.mark.parametrize("database_down", [True, False])
def test_read_memory_session_snapshot(monkeypatch, isolated_attempt_store, database_down):
    store = isolated_attempt_store
    record = store.create_attempt("snapshot_student", "phy_001")
    state = record.state
    state.current_answer = "work in progress"
    state.current_step = "modeling"
    state.timeline = [TimelineEvent(timestamp=1.5, event_type="submit",
                                    content=state.current_answer)]
    state.weak_concepts = ["work"]
    store.save_attempt(record, state)
    store.update_attempt(record.attempt_id,
                         hint_history=[{"hint_level": 1, "hint": "check work",
                                        "issue_type": "concept"}])

    def read_detail(aid):
        if database_down:
            raise persistence.PersistenceUnavailable("offline")
        return None

    monkeypatch.setattr(persistence, "get_attempt_detail", read_detail)
    response = TestClient(app).get(f"/attempt/{record.attempt_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["student_id"] == "snapshot_student"
    assert body["status"] == "in_progress"
    assert body["state"]["current_answer"] == "work in progress"
    assert body["events"][0] == {"timestamp": 1.5, "event_type": "submit",
                                 "content": "work in progress"}
    assert body["hints"][0]["hint"] == "check work"
    assert body["weak_concepts"] == ["work"]
    assert body["record_source"] == "state_snapshot"
    assert body["start_time"]
