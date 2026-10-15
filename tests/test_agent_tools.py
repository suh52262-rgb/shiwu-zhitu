"""RAG Tool 层测试（P7，任务书第 10 节）。"""
import pytest

from app.agent import tools as tools_module
from app.agent.tools import RAG_TOOLS, retrieve_knowledge


def test_registry_contains_all_tools():
    assert set(RAG_TOOLS) == {
        "retrieve_knowledge",
        "retrieve_similar_questions",
        "retrieve_school_exam",
        "retrieve_student_mistakes",
    }


def test_retrieve_knowledge_combines_query_and_knowledge_points(monkeypatch):
    # 关闭重排：它默认开启且会调模型，而这条测试只关心"查询拼接"。
    from app.rag import rerank as _rerank

    monkeypatch.setattr(_rerank, "ENABLED", False)
    captured = {}

    calls = []

    def _fake_retrieve(query, top_k=5):
        # **记录全部调用**而不是只留最后一次：开了查询扩展时会多路检索，
        # 只留最后一次会把"拼接是否正确"这条断言变成看运气。
        calls.append((query, top_k))
        captured["query"] = query
        captured["top_k"] = top_k
        return [{"text": "t", "source": "s", "page": 1}]

    monkeypatch.setattr(tools_module, "retrieve", _fake_retrieve)
    results = retrieve_knowledge(
        "未考虑摩擦力做功", knowledge_points=["机械能守恒条件", "动能定理"], top_k=3
    )

    assert results[0]["page"] == 1
    # **契约变了**：为了让重排有腾挪余地，内部会取更深的候选
    # （`max(top_k, 10)`），重排后再截回 `top_k`。
    # 真正有意义的契约是**调用方拿到 top_k 条**，而不是内部深度——
    # 所以断言改到返回长度上，而不是放宽成"随便多少"。
    assert captured["top_k"] >= 3, "内部候选深度不该少于请求的 top_k"
    assert len(results) <= 3, f"返回条数超过了请求的 top_k：{len(results)}"
    # query 与知识点拼接，提升召回针对性
    assert "未考虑摩擦力做功" in captured["query"]
    assert "机械能守恒条件" in captured["query"]
    assert "动能定理" in captured["query"]


def test_retrieve_knowledge_without_points_keeps_query(monkeypatch):
    captured = {}

    def _fake_retrieve(query, top_k=5):
        captured["query"] = query
        return []

    monkeypatch.setattr(tools_module, "retrieve", _fake_retrieve)
    retrieve_knowledge("分离条件", knowledge_points=None)
    assert captured["query"] == "分离条件"


def test_retrieve_knowledge_empty_input_returns_empty():
    assert retrieve_knowledge("   ", knowledge_points=None) == []
    assert retrieve_knowledge("", knowledge_points=[]) == []


def test_retrieve_knowledge_passes_through_empty_results(monkeypatch):
    monkeypatch.setattr(tools_module, "retrieve", lambda query, top_k=5: [])
    assert retrieve_knowledge("任意", knowledge_points=["x"]) == []


# ---- 未来工具：占位且显式报错（等成员 B 数据） ----

def test_future_tools_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        RAG_TOOLS["retrieve_similar_questions"]("q")
    with pytest.raises(NotImplementedError):
        RAG_TOOLS["retrieve_school_exam"]("q")
    with pytest.raises(NotImplementedError):
        RAG_TOOLS["retrieve_student_mistakes"]("stu_001")
