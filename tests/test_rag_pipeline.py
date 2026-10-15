"""RAG 全链路测试：生成小 PDF → 入库（假 embedding）→ 检索。

不联网、不加载真实模型；Qdrant 使用本地模式 + pytest 临时目录。
"""
import fitz
import pytest
from qdrant_client import QdrantClient

from app.rag import ingest as ingest_module
from app.rag import retriever as retriever_module
from app.db import qdrant as qdrant_module


class FakeEmbedding:
    """确定性假 embedding：字符频率 → 归一化 16 维向量。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * 16
            for ch in text:
                v[ord(ch) % 16] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors


PDF_TEXT = (
    "Chapter 1 Work and Kinetic Energy. The net work done on an object equals the "
    "change of its kinetic energy. This is called the work energy theorem. When the "
    "rough incline exerts friction, the friction force does negative work.\n\n"
    "Chapter 2 Conservation of Mechanical Energy. Mechanical energy is conserved only "
    "when gravity is the only force doing work. Friction converts mechanical energy "
    "into internal energy, so mechanical energy decreases on a rough incline."
)


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    """把 ingest / retriever 的依赖替换为：临时本地 Qdrant + FakeEmbedding。"""
    client = QdrantClient(path=str(tmp_path / "qdb"))
    fake = FakeEmbedding()
    monkeypatch.setattr(ingest_module, "get_qdrant", lambda: client)
    monkeypatch.setattr(ingest_module, "get_embedding_client", lambda: fake)
    monkeypatch.setattr(retriever_module, "get_qdrant", lambda: client)
    monkeypatch.setattr(retriever_module, "get_embedding_client", lambda: fake)
    yield client
    client.close()


@pytest.fixture
def tiny_pdf(tmp_path):
    pdf_path = tmp_path / "demo_book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 780), PDF_TEXT, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_retrieve_before_ingest_returns_empty(rag_env):
    assert retriever_module.retrieve("friction on incline") == []


def test_ingest_counts_chunks(rag_env, tiny_pdf):
    count = ingest_module.ingest_pdf(tiny_pdf)
    assert count >= 1
    # 幂等：重复入库数量不变（确定性 ID 覆盖）
    assert ingest_module.ingest_pdf(tiny_pdf) == count


def test_retrieve_roundtrip(rag_env, tiny_pdf):
    ingest_module.ingest_pdf(tiny_pdf)
    results = retriever_module.retrieve("friction does negative work on incline", top_k=3)

    assert 1 <= len(results) <= 3
    for item in results:
        assert set(item) >= {"text", "source", "page", "score"}
        assert item["source"] == "demo_book.pdf"
        assert item["page"] == 1
    # 语义命中的 chunk 应包含关键词
    assert any("friction" in r["text"].lower() for r in results)


def test_retrieve_empty_query(rag_env, tiny_pdf):
    ingest_module.ingest_pdf(tiny_pdf)
    assert retriever_module.retrieve("   ") == []
