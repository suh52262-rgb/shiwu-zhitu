"""All three input adapters must obey the same attempt lifecycle.

Exercise the real HTTP and LangGraph paths, replacing only external IO.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.agent import graph
from app.api import video
from app.db import persistence
from app.main import app
from app.services import review_builder
from app.vision import service
from fakes import GOOD_ANALYSIS, install_fake_llm


@pytest.fixture
def media_client(monkeypatch, tmp_path):
    monkeypatch.setattr(graph, "retrieve_knowledge_tool", lambda *a, **kw: [])
    for name in ("persist_events", "persist_hint", "persist_profile_weak_concepts",
                 "persist_trace"):
        monkeypatch.setattr(persistence, name, lambda *a, **kw: False)
    monkeypatch.setattr(persistence, "resolve_pending_hints", lambda *a, **kw: [])
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *a, **kw: None)
    monkeypatch.setattr(service, "call_vision", lambda *a, **kw: (
        json.dumps({"text": "mgh = mv^2/2", "has_content": True}), {}))
    monkeypatch.setattr(video, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(video, "RUNS_DIR", tmp_path / "video_runs")
    monkeypatch.setattr(video, "UPLOAD_DIR", tmp_path / "video_runs" / "uploads")

    def pipeline(path, out_dir, options, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps({"frames": [
            {"frame_type": "keyframe", "timestamp_sec": ts, "text": "mgh" if ts == 0 else "mgh = mv^2/2"}
            for ts in (0.0, 25.0)
        ]}), encoding="utf-8")
        return {"coverage": {"truncated": False, "video_total_sec": 30.0,
                             "covered_until_sec": 30.0}, "failed": 0}

    monkeypatch.setattr(service, "run_pipeline", pipeline)
    return TestClient(app)


BASE = {"student_id": "lifecycle_probe", "problem_id": "phy_001"}


def test_text_photo_video_continue_same_attempt_and_reset(media_client, monkeypatch,
                                                          isolated_attempt_store):
    install_fake_llm(monkeypatch)
    response = media_client.post("/attempt/analyze", json={**BASE, "answer": "mgh"})
    assert response.status_code == 200
    aid = response.json()["attempt_id"]
    for endpoint, filename in (("analyze_image", "work.png"), ("analyze", "work.mp4")):
        response = media_client.post(f"/video/{endpoint}",
            data={**BASE, "attempt_id": aid}, files={"file": (filename, b"mock-media")})
        assert response.status_code == 200, response.text
        assert response.json()["attempt_id"] == aid
    record = isolated_attempt_store.load_attempt(aid)
    assert record.is_open
    assert sum(e.event_type == "submit" for e in record.state.timeline) == 4
    assert media_client.post("/attempt/reset", json=BASE).status_code == 200
    assert isolated_attempt_store.load_attempt(aid).status == "abandoned"
    response = media_client.post("/attempt/analyze", json={**BASE, "answer": "mgh",
                                                           "attempt_id": aid})
    assert response.status_code == 409
    response = media_client.post("/attempt/analyze", json={**BASE, "answer": "mgh"})
    assert response.status_code == 200
    assert response.json()["attempt_id"] != aid


def test_video_completion_preserves_terminal_attempt(media_client, monkeypatch,
                                                     isolated_attempt_store):
    install_fake_llm(monkeypatch, analysis=GOOD_ANALYSIS)
    response = media_client.post("/video/analyze", data=BASE,
                                files={"file": ("work.mp4", b"mock-media")})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["submitted"] == 1
    assert result["remaining_snapshots"] == 1
    aid = result["attempt_id"]
    record = isolated_attempt_store.load_attempt(aid)
    assert record.status == "finished"
    assert sum(e.event_type == "submit" for e in record.state.timeline) == 1
    before = record.model_dump(mode="json")
    response = media_client.post("/video/analyze_image", data={**BASE, "attempt_id": aid},
                                files={"file": ("work.png", b"mock-media")})
    assert response.status_code == 409
    assert isolated_attempt_store.load_attempt(aid).model_dump(mode="json") == before
