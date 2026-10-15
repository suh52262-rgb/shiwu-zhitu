"""FastAPI 基础测试：不依赖数据库 / LLM 等外部服务。"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.4.0"


def test_demo_problem_endpoint():
    resp = client.get("/problems/demo")
    assert resp.status_code == 200
    assert resp.json()["problem_id"] == "phy_001"


def test_attempt_reset_endpoint():
    resp = client.post(
        "/attempt/reset", json={"student_id": "s", "problem_id": "p"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"
