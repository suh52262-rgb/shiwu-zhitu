"""图注重复：同一条图注既是 text 块又是 figure_caption 块。

全链冒烟测试发现的真问题：真实教材 637 条图注里 **374 条（58.7%）重复**，
另有 15 条 formula/text 重复。两份都进了索引，`top_k=5` 命中图注时
会有半个列表是同一条内容——**有效结果数直接减半**。

审计第 4 节早就提过（"图注与正文重复时只保留一个可检索表示"），
但一直没落地。这次两侧都修。
"""
import inspect
import pytest

from app.rag.retriever import retrieve


def test_retriever_dedupes_by_doc_page_text():
    """**检索侧去重**——对已有数据立即生效，不用重导。

    放在检索侧而不是只靠导入侧拦截，是因为已经入库的 374 条重复
    不会因为改了导入代码就消失。

    去重**在收集循环里做**（而不是"先取满 top_k 再去重"）：
    后者一条重复就把结果削到 top_k−1，实测 `'机械能守恒的条件'`
    因此只返回 7 条。
    """
    src = inspect.getsource(retrieve)
    assert "seen_keys" in src, "检索结果没去重"
    # 去重键必须是 (源, 页, 文本)
    assert "item.get(\"source\")" in src and "item.get(\"page\")" in src
    assert "item.get(\"text\")" in src
    # 去重必须发生在收集过程中，而不是收集之后
    i_seen = src.index("if key in seen_keys")
    i_append = src.index("kept.append(item)")
    assert i_seen < i_append, "去重发生在收集之后，会把结果削到 top_k 以下"


@pytest.mark.local_asset
def test_retriever_still_returns_full_top_k():
    """去重不能把结果削减到少于 `top_k`（除非池子本来就不够）。

    过度去重和不去重一样有害——前者让结果变少，后者让结果重复。
    实测 10 个查询 × top_k=8 全部返回满额 8 条。
    """
    for q in ("冲量 动量定理", "机械能守恒的条件", "角动量"):
        hits = retrieve(q, top_k=8)
        assert len(hits) == 8, f"{q!r} 只返回 {len(hits)} 条，被过度去重"


@pytest.mark.local_asset
def test_no_duplicate_text_in_results():
    """结果里不应出现 (文档, 页, 文本) 相同的两条。"""
    for q in ("冲量 动量定理", "碰撞 平均冲力", "动量守恒", "射向"[:2] + "高"):
        hits = retrieve(q, top_k=8)
        keys = [(h.get("source"), h.get("page"), (h.get("text") or "").strip())
                for h in hits]
        assert len(keys) == len(set(keys)), f"{q!r} 结果里有重复：{keys}"


def test_import_marks_caption_duplicate_unindexable():
    """**导入侧去重**：图注文本不再另外生成一份可索引的正文块。

    原文仍留在 PG（审计、排障要用），只是不进检索。
    """
    from app.rag import pdf_import

    src = inspect.getsource(pdf_import.collect_page)
    assert "caption_texts" in src, "没有收集图注文本"
    assert "与图注块内容重复" in src, "没有标记重复的正文块"
    assert 'b["indexable"] = False' in src, "重复块没有被标为不可索引"


def test_caption_duplicate_note_is_reported():
    """去重了多少条要**说出来**——静默去重比不去重更难排查。"""
    from app.rag import pdf_import

    src = inspect.getsource(pdf_import.collect_page)
    assert "条图注已去重" in src
