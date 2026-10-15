"""GET /students/{id}/profile 接口测试。"""
import uuid

import pytest
from db_utils import db_available
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import persistence
from app.main import app

client = TestClient(app)


def test_profile_503_when_db_down(monkeypatch):
    def _unavailable(student_id):
        raise persistence.PersistenceUnavailable("connection refused")

    monkeypatch.setattr(persistence, "get_student_profile", _unavailable)
    assert client.get("/students/x/profile").status_code == 503


def test_profile_404_when_missing(monkeypatch):
    monkeypatch.setattr(persistence, "get_student_profile", lambda student_id: None)
    assert client.get("/students/ghost/profile").status_code == 404


def test_profile_200(monkeypatch):
    monkeypatch.setattr(
        persistence,
        "get_student_profile",
        lambda student_id: {"student_id": student_id, "weak_concepts": ["a"], "attempts": []},
    )
    resp = client.get("/students/s1/profile")
    assert resp.status_code == 200
    assert resp.json()["weak_concepts"] == ["a"]


@pytest.mark.skipif(not db_available(), reason="PostgreSQL 不可用（docker compose 未启动）")
@pytest.mark.real_db
def test_profile_integration_roundtrip():
    from app.models.problem import Problem as ProblemSchema

    run = uuid.uuid4().hex[:8]
    sid = f"stu_prof_{run}"
    problem = ProblemSchema(problem_id="phy_001", title="t", content="c")

    assert persistence.persist_attempt_start(sid, "phy_001", f"att{run}", problem=problem)
    assert persistence.persist_profile_weak_concepts(sid, ["机械能守恒条件"])

    resp = client.get(f"/students/{sid}/profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["student_id"] == sid
    assert body["weak_concepts"] == ["机械能守恒条件"]
    assert len(body["attempts"]) == 1
    assert body["attempts"][0]["problem_id"] == "phy_001"

    with __import__("app.db.postgres", fromlist=["get_engine"]).get_engine().begin() as conn:
        conn.execute(text("DELETE FROM attempts WHERE student_id=:s"), {"s": sid})
        conn.execute(text("DELETE FROM student_profiles WHERE student_id=:s"), {"s": sid})
        conn.execute(text("DELETE FROM students WHERE id=:s"), {"s": sid})
