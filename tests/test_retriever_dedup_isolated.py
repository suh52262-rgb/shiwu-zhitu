"""Exercise result collection without a model, corpus, database or network."""
from types import SimpleNamespace

from app.rag import block_store, retriever


def install_pool(monkeypatch, payloads):
    points = [SimpleNamespace(payload=p, score=1 - i * 0.01)
              for i, p in enumerate(payloads)]
    client = SimpleNamespace(collection_exists=lambda _: True,
                             query_points=lambda **_: SimpleNamespace(points=points))
    monkeypatch.setattr(retriever, "get_qdrant", lambda: client)
    monkeypatch.setattr(retriever, "get_embedding_client",
                        lambda: SimpleNamespace(embed=lambda _: [[1., 0.]]))
    monkeypatch.setattr(block_store, "doc_index_status", lambda _: "indexed")
    monkeypatch.setattr(block_store, "is_index_visible", lambda _: True)


def test_duplicate_at_front_does_not_consume_result_slot(monkeypatch):
    first = {"source": "doc", "page": 1, "text": "a substantive caption"}
    pool = [first, {**first, "type": "figure_caption"}]
    pool.extend({"source": "doc", "page": i + 2, "text": f"substantive result {i}"}
                for i in range(8))
    install_pool(monkeypatch, pool)
    hits = retriever.retrieve("query", top_k=8, expand_context=False)
    assert len(hits) == 8
    keys = [(h["source"], h["page"], h["text"].strip()) for h in hits]
    assert len(set(keys)) == 8


def test_small_pool_retains_same_text_from_different_provenance(monkeypatch):
    first = {"source": "doc", "page": 1, "text": "the same substantive text"}
    install_pool(monkeypatch, [first, {**first, "text": " " + first["text"] + " "},
                              {**first, "page": 2}, {**first, "source": "other"}])
    hits = retriever.retrieve("query", top_k=8, expand_context=False)
    assert [(h["source"], h["page"]) for h in hits] == [("doc", 1), ("doc", 2), ("other", 1)]
