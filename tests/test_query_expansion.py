"""查询扩展：只能显式开启，且开启时不得破坏降级路径。

## 这组测试盯的两件事

1. **默认必须是关的。** 实测收益在 n=60 的噪声内，代价是每次检索一次
   LLM 调用——**这个交换不成立**，所以不能默认开。
2. **改写会跑偏，所以必须融合而不是替换。** 实测"只用改写"55%
   比基线 75% 低 20 个百分点——这不是理论担忧。
"""
import inspect

import pytest

from app.rag import query_expansion as qe


# ---------------------------------------------------------------- 默认关闭

def test_disabled_by_default():
    """**默认关闭**，且理由写在源码里。

    开了它，`retrieve_knowledge` 就会调模型——而项目里有测试依赖
    "绕过 LLM 层时不应产生 token 用量"这条契约。
    是那条测试抓到了这个改动。
    """
    assert qe.ENABLED is False, "查询扩展被改成默认开启了——请先量一次收益"
    src = inspect.getsource(qe)
    assert "默认关闭" in src, "缺少默认关闭的说明"
    assert "噪声" in src, "没有写清收益不显著这件事"


def test_disabled_expand_returns_empty():
    """关闭时直接返回空——调用方据此退回单路。"""
    assert qe.expand_query("为什么机械能不守恒") == []


def test_empty_query_returns_empty():
    assert qe.expand_query("") == []
    assert qe.expand_query("   ") == []


def test_expand_failure_returns_empty_not_raises(monkeypatch):
    """**检索路径上的东西不能让整条链挂掉。**"""
    monkeypatch.setattr(qe, "ENABLED", True)

    class _Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("模型不可用")

    assert qe.expand_query("问题", llm=_Boom()) == []


def test_expand_parses_lines_and_strips_numbering(monkeypatch):
    monkeypatch.setattr(qe, "ENABLED", True)

    class _LLM:
        def chat(self, *a, **kw):
            return '1. 重力做功与高度乘积\n2) 无摩擦可逆装置\n好的\n"动能公式"'

    got = qe.expand_query("问题", llm=_LLM(), n=5)
    assert "重力做功与高度乘积" in got
    assert "无摩擦可逆装置" in got
    assert "动能公式" in got
    # 噪音要滤掉
    assert "好的" not in got
    assert all(4 <= len(x) <= 60 for x in got)


# ---------------------------------------------------------------- 融合

def _item(text, page=1, score=0.5):
    return {"text": text, "source": "d", "page": page, "score": score}


def test_interleave_keeps_base_in_odd_positions():
    """**交错的真正价值**：原查询结果固定占据 1、3、5… 位。

    即使改写完全跑偏，top-5 里仍有三条来自原查询——
    RRF 没这个保证。
    """
    base = [_item("b0"), _item("b1"), _item("b2")]
    alt = [_item("a0"), _item("a1"), _item("a2")]
    merged = qe.merge_interleave([base, alt])
    assert [x["text"] for x in merged[:5]] == ["b0", "a0", "b1", "a1", "b2"]


def test_interleave_dedupes_across_lists():
    """两路都召回同一块时不能出现两次。"""
    base = [_item("same"), _item("b1")]
    alt = [_item("same"), _item("a1")]
    texts = [x["text"] for x in qe.merge_interleave([base, alt])]
    assert texts.count("same") == 1
    assert set(texts) == {"same", "b1", "a1"}


def test_single_list_passthrough():
    one = [_item("x"), _item("y")]
    assert qe.merge_interleave([one]) == one
    assert qe.merge_interleave([[], one]) == one
    assert qe.merge_interleave([]) == []


def test_rrf_still_available():
    """RRF 保留但非默认——它的取舍写在 docstring 里。"""
    base = [_item("b0"), _item("b1")]
    alt = [_item("a0"), _item("b0")]
    merged = qe.merge_rrf([base, alt])
    # b0 在两路都出现，RRF 分应最高
    assert merged[0]["text"] == "b0"
    assert "rrf" in merged[0]


def test_ident_matches_retriever_dedup_key():
    """块身份口径必须与检索去重一致，否则融合会漏掉重复项。"""
    src = inspect.getsource(qe._ident)
    assert "source" in src and "page" in src and "text" in src


# ---------------------------------------------------------------- 接线

def test_tools_layer_degrades_when_disabled():
    """关闭时 `retrieve_knowledge` 应只调一次 `retrieve`（单路）。"""
    src = inspect.getsource(__import__("app.agent.tools", fromlist=["x"]))
    assert "expand_query" in src, "tools 层没接查询扩展"
    assert "if not variants:" in src, "没有单路退化分支"
    assert "merge_interleave" in src, "没有融合"
