"""杂项边界测试：chat 占位、配置、embedding 工厂。"""
import pytest

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.rag import embedding as embedding_module

client = TestClient(app)


def test_chat_placeholder():
    resp = client.post("/chat", json={"message": "你好"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_implemented"


def test_postgres_dsn_contains_components():
    dsn = get_settings().postgres_dsn
    assert dsn.startswith("postgresql+psycopg://")
    assert "@localhost:5432/" in dsn


def test_embedding_factory_unknown_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "word2vec")
    get_settings.cache_clear()
    embedding_module.clear_cache()      # get_embedding_client 现在有进程内缓存
    try:
        with pytest.raises(ValueError, match="未知 EMBEDDING_PROVIDER"):
            embedding_module.get_embedding_client()
    finally:
        get_settings.cache_clear()
        embedding_module.clear_cache()


def test_embedding_qwen_requires_api_key(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "")
    get_settings.cache_clear()
    embedding_module.clear_cache()
    try:
        with pytest.raises(ValueError, match="QWEN_API_KEY"):
            embedding_module.get_embedding_client()
    finally:
        get_settings.cache_clear()
        embedding_module.clear_cache()


def test_embedding_client_is_cached(monkeypatch):
    """回归：BGE 客户端必须缓存。

    它的构造会加载整个模型（数百 ms~数秒、上百 MB），而 retriever 每次检索都会
    调用工厂；不缓存就等于每轮 RAG 都重新加载一遍模型。
    这里用假实现验证"同一配置只构造一次"，不真的加载模型。
    """
    constructed: list[str] = []

    class FakeClient:
        def __init__(self, model_name: str) -> None:
            constructed.append(model_name)

        def embed(self, texts):  # pragma: no cover - 本用例不调用
            return []

    monkeypatch.setattr(embedding_module, "LocalBgeEmbedding", FakeClient)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bge")
    get_settings.cache_clear()
    embedding_module.clear_cache()
    try:
        first = embedding_module.get_embedding_client()
        second = embedding_module.get_embedding_client()
        assert first is second
        assert len(constructed) == 1
    finally:
        get_settings.cache_clear()
        embedding_module.clear_cache()
