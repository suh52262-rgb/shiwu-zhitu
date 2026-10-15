"""显式续做的归属、生命周期与变更锁契约（纯内存 + FakeLLM）。"""
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api import attempts
from app.db import persistence
from app.main import app
from fakes import GOOD_ANALYSIS, install_fake_llm

client = TestClient(app)
PAYLOAD = {"student_id": "owner", "problem_id": "phy_001", "answer": "机械能守恒"}


@pytest.mark.parametrize(
    "overrides,status",
    [({"student_id": "other"}, 403), ({"problem_id": "phy_002"}, 403),
     ({"attempt_id": "missing"}, 404)],
)
@pytest.mark.parametrize("request_hint", [False, True])
def test_invalid_explicit_attempt_has_no_side_effects(
    monkeypatch, isolated_attempt_store, overrides, status, request_hint,
):
    store = isolated_attempt_store
    record = store.create_attempt("owner", "phy_001")
    before = store.load_attempt(record.attempt_id).model_dump()
    fake = install_fake_llm(monkeypatch)
    writes = Mock()
    monkeypatch.setattr(persistence, "persist_events", writes)
    monkeypatch.setattr(persistence, "persist_trace", writes)
    monkeypatch.setattr(persistence, "persist_profile_weak_concepts", writes)
    response = client.post("/attempt/analyze", json={
        **PAYLOAD, "attempt_id": record.attempt_id, "request_hint": request_hint, **overrides,
    })
    assert response.status_code == status
    assert fake.calls == []
    writes.assert_not_called()
    assert store.load_attempt(record.attempt_id).model_dump() == before
    assert len(store._memory) == 1


@pytest.mark.parametrize("terminal", ["finished", "abandoned"])
@pytest.mark.parametrize("request_hint", [False, True])
def test_terminal_attempt_rejects_analyze_without_writes(
    monkeypatch, isolated_attempt_store, terminal, request_hint,
):
    store = isolated_attempt_store
    record = store.create_attempt("owner", "phy_001")
    store.finish_attempt(record.attempt_id, status=terminal)
    before = store.load_attempt(record.attempt_id).model_dump()
    fake = install_fake_llm(monkeypatch)
    response = client.post("/attempt/analyze", json={
        **PAYLOAD, "attempt_id": record.attempt_id, "request_hint": request_hint,
    })
    assert response.status_code == 409
    assert fake.calls == []
    assert store.load_attempt(record.attempt_id).model_dump() == before


def test_valid_explicit_attempt_resumes_hint_history(monkeypatch, isolated_attempt_store):
    fake = install_fake_llm(monkeypatch)
    first = client.post("/attempt/analyze", json=PAYLOAD).json()
    second = client.post("/attempt/analyze", json={
        **PAYLOAD, "attempt_id": first["attempt_id"], "request_hint": True,
    })
    assert second.status_code == 200
    assert second.json()["attempt_id"] == first["attempt_id"]
    record = isolated_attempt_store.load_attempt(first["attempt_id"])
    assert record.hint_count == 1
    assert [e.event_type for e in record.state.timeline] == ["submit", "submit", "hint_request"]
    assert fake.calls


def test_auto_finish_blocks_explicit_resume_and_empty_id_starts_new(monkeypatch):
    install_fake_llm(monkeypatch, analysis=GOOD_ANALYSIS)
    first = client.post("/attempt/analyze", json=PAYLOAD).json()
    assert first["status"] == "finished"
    rejected = client.post("/attempt/analyze", json={**PAYLOAD, "attempt_id": first["attempt_id"]})
    assert rejected.status_code == 409
    second = client.post("/attempt/analyze", json=PAYLOAD)
    assert second.status_code == 200
    assert second.json()["attempt_id"] != first["attempt_id"]


def test_preflight_empty_id_does_not_create_attempt(isolated_attempt_store):
    assert attempts.validate_explicit_attempt("", "owner", "phy_001") is None
    assert isolated_attempt_store._memory == {}


@pytest.mark.parametrize("endpoint", ["reset", "finish"])
def test_lifecycle_changes_wait_for_student_problem_lock(monkeypatch, isolated_attempt_store, endpoint):
    store = isolated_attempt_store
    record = store.create_attempt("owner", "phy_001")
    entered = threading.Event()
    lock = store.attempt_lock("owner", "phy_001")

    class ObservedLock:
        def acquire(self, **kwargs):
            entered.set()
            return lock.acquire(**kwargs)

        def release(self):
            lock.release()

    def observed_lock(student_id, problem_id):
        assert (student_id, problem_id) == ("owner", "phy_001")
        return ObservedLock()

    monkeypatch.setattr(store, "attempt_lock", observed_lock)

    def change_lifecycle():
        if endpoint == "reset":
            return client.post("/attempt/reset", json=PAYLOAD)
        return client.post(f"/attempt/{record.attempt_id}/finish")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with lock:
            future = executor.submit(change_lifecycle)
            assert entered.wait(timeout=2)
            # 锁仍由测试线程持有，变更不能越过锁提交。
            assert not future.done()
            assert store.load_attempt(record.attempt_id).status == "in_progress"
        assert future.result(timeout=2).status_code == 200
    assert store.load_attempt(record.attempt_id).status == (
        "abandoned" if endpoint == "reset" else "finished"
    )


def test_explicit_validation_runs_inside_attempt_lock(monkeypatch, isolated_attempt_store):
    store = isolated_attempt_store
    record = store.create_attempt("owner", "phy_001")
    install_fake_llm(monkeypatch)
    acquired = False
    underlying = store.attempt_lock("owner", "phy_001")

    class ObservedLock:
        def acquire(self, **kwargs):
            nonlocal acquired
            acquired = underlying.acquire(**kwargs)
            return acquired

        def release(self):
            nonlocal acquired
            acquired = False
            underlying.release()

    original = attempts.validate_explicit_attempt

    def validate(*args):
        assert acquired
        return original(*args)

    monkeypatch.setattr(store, "attempt_lock", lambda *args: ObservedLock())
    monkeypatch.setattr(attempts, "validate_explicit_attempt", validate)
    assert client.post("/attempt/analyze", json={
        **PAYLOAD, "attempt_id": record.attempt_id,
    }).status_code == 200
    assert not acquired


def test_finish_is_idempotent_and_does_not_revive_abandoned(isolated_attempt_store):
    store = isolated_attempt_store
    finished = store.create_attempt("owner", "phy_001")
    store.finish_attempt(finished.attempt_id, final_correct=True)
    response = client.post(f"/attempt/{finished.attempt_id}/finish", json={"final_correct": False})
    assert response.status_code == 200
    assert response.json()["final_correct"] is True
    abandoned = store.create_attempt("owner", "phy_001")
    store.abandon_attempt(abandoned.attempt_id)
    assert client.post(f"/attempt/{abandoned.attempt_id}/finish").status_code == 409
