"""GET /knowledge/search 接口测试（假 Tool，不依赖向量库）。"""
from fastapi.testclient import TestClient

from app.api import knowledge as knowledge_module
from app.main import app

client = TestClient(app)


def test_search_returns_formatted_items(monkeypatch):
    captured = {}

    def _fake_tool(query, knowledge_points=None, top_k=5):
        captured.update(query=query, knowledge_points=knowledge_points, top_k=top_k)
        return [{"text": "机械能守恒的条件…", "source": "教材.pdf", "page": 85, "score": 0.77}]

    monkeypatch.setattr(knowledge_module, "retrieve_knowledge", _fake_tool)
    resp = client.get(
        "/knowledge/search",
        params={"query": "机械能守恒", "top_k": 1, "knowledge_points": "动能定理, 摩擦力做功"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "机械能守恒"
    assert body["knowledge_points"] == ["动能定理", "摩擦力做功"]
    assert body["items"][0]["page"] == 85
    assert captured["knowledge_points"] == ["动能定理", "摩擦力做功"]


def test_search_query_required():
    assert client.get("/knowledge/search").status_code == 422
