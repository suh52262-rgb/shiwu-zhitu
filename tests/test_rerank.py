"""LLM 重排：三轮失败之后唯一被实测证明有效的检索改动。

## 为什么这组测试值得单独写

前三轮改动（查询扩展 +2.5% p=0.18、长度惩罚 +0.7%、标题降权 **+0.0%**）
都是**在同一套向量分数上做缩放**，全部无效。

重排有效是因为它引入**新信息**（query-passage 交互），改的是 top-5 的**成员**。

这组测试盯的是：**它必须真的开启**（默认开，与查询扩展相反），
以及**任何失败都要退回原序**（重排不能成为检索的单点故障）。
"""
import inspect

import pytest

from app.rag import rerank as rr


# ---------------------------------------------------------------- 默认开启

def test_enabled_by_default():
    """**默认开启**——与查询扩展相反。

    查询扩展的收益在噪声内（p=0.18），代价是每次检索一次模型调用，
    不值得默认开。这个 p=0.0009，是三轮里唯一显著的。
    """
    assert rr.ENABLED is True, "重排被改成默认关闭了——请先用 golden set 量一次"
    src = inspect.getsource(rr)
    assert "0.0009" in src, "缺少实测依据"


def test_disabled_returns_original(monkeypatch):
    monkeypatch.setattr(rr, "ENABLED", False)
    items = [{"text": "a"}, {"text": "b"}]
    assert rr.rerank("q", items) == items


def test_empty_inputs_are_safe():
    assert rr.rerank("q", []) == []
    assert rr.rerank("", [{"text": "a"}]) == [{"text": "a"}]
    one = [{"text": "only"}]
    assert rr.rerank("q", one) == one


# ---------------------------------------------------------------- 降级

def test_llm_failure_returns_original(monkeypatch):
    """**任何失败都退回原序**——检索不能因重排而失败。"""
    class _Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("模型不可用")

    items = [{"text": f"p{i}"} for i in range(5)]
    assert rr.rerank("q", items, llm=_Boom()) == items


def test_unparseable_reply_keeps_results(monkeypatch):
    """模型输出无法解析时**保持原序，而不是清空结果**。

    一个不可靠的判断不该否决向量检索的全部产出。
    """
    class _Garbage:
        def chat(self, *a, **kw):
            return "我不确定"

    items = [{"text": f"p{i}"} for i in range(5)]
    assert rr.rerank("q", items, llm=_Garbage()) == items


def test_model_says_none_refuses_instead_of_keeping_results():
    """🔴 **模型明确说"都不相关" ⇒ 拒答（返回空），不是保持原序**（2026-10-10 口径变更）。

    ⚠️ **这条测试原先断言的是相反的行为**（"保持原序"），理由是
    "它可能只是判断保守，不能据此把结果清空"。
    而 RAG 核查查出了那个理由的代价：
    ```
    酒店 query 的重排模型**明确输出「无」**，代码却回退原序
    ⇒ **无关问题照样被塞进三条物理教材结果**。
    ```
    ⇒ 用户定的新口径：**明确「无」＝ 拒答**；
      **超时 / 异常 / 解析不了**才回退原序 —— 两者**必须分开**（见下一条测试）。
    """
    class _None:
        def chat(self, *a, **kw):
            return "无"

    items = [{"text": f"p{i}"} for i in range(5)]
    st: dict = {}
    assert rr.rerank("q", items, llm=_None(), status=st) == []
    assert st["rerank_status"] == "none", st
    assert "没有相关材料" in st["rerank_reason"]


def test_unparseable_output_falls_back_but_says_so():
    """**解析不了 ⇒ 保守回退原序，但如实记状态** —— 不许伪装成"重排成功"或"无结果"。

    ⚠️ 判据要窄：只有"整条回复基本就是那个否定词"才算「无」。
    否则一句「第 2 条不相关」会被误读成"全都无关" ⇒ **把好结果也拒了**。
    """
    class _Junk:
        def chat(self, *a, **kw):
            return "???"

    class _Partial:
        def chat(self, *a, **kw):
            return "第 2 条不相关"

    items = [{"text": f"p{i}"} for i in range(5)]

    st: dict = {}
    assert rr.rerank("q", items, llm=_Junk(), status=st) == items
    assert st["rerank_status"] == "unparsed", st

    st2: dict = {}
    out = rr.rerank("q", items, llm=_Partial(), status=st2)
    assert st2["rerank_status"] == "ok", st2        # ← **不许当成"全都无关"**
    assert len(out) == len(items)


def test_a_rerank_failure_is_recorded_not_hidden():
    """超时/异常 ⇒ 回退原序 **且记 `failed` + 原因**（不能伪装成成功或无结果）。"""
    class _Boom:
        def chat(self, *a, **kw):
            raise TimeoutError("rerank timeout")

    items = [{"text": f"p{i}"} for i in range(5)]
    st: dict = {}
    assert rr.rerank("q", items, llm=_Boom(), status=st) == items
    assert st["rerank_status"] == "failed", st
    assert "TimeoutError" in st["rerank_reason"]


# ---------------------------------------------------------------- 重排行为

def test_reorders_without_losing_items():
    """**只换序，不增删**——长度必须不变。"""
    class _Pick:
        def chat(self, *a, **kw):
            return "3,1"

    items = [{"text": f"p{i}"} for i in range(1, 6)]
    out = rr.rerank("q", items, llm=_Pick())
    assert len(out) == len(items)
    assert [x["text"] for x in out[:2]] == ["p3", "p1"]
    # 未被选中的保持原相对顺序
    assert [x["text"] for x in out[2:]] == ["p2", "p4", "p5"]


def test_parse_picks_tolerates_formats():
    """模型输出格式多变：`3,7,1` / `3、7` / `第3段` / 带解释。"""
    assert rr.parse_picks("3,7,1", 10) == [3, 7, 1]
    assert rr.parse_picks("3、7", 10) == [3, 7]
    assert rr.parse_picks("第 3 段最相关", 10) == [3]
    # 越界与重复要去掉
    assert rr.parse_picks("99,2,2,1", 5) == [2, 1]
    # 取不到就返回空 → 调用方退回原序
    assert rr.parse_picks("无", 5) == []
    assert rr.parse_picks("", 5) == []


def test_picks_beyond_depth_ignored():
    """候选池深度限制了可选编号——越界的不能选。"""
    assert rr.parse_picks("8", 3) == []


def test_only_head_is_reranked():
    """只重排前 `depth` 条，更深的保持不动（它们本来也不会进 top_k）。"""
    class _Pick:
        def chat(self, *a, **kw):
            return "2,1"

    items = [{"text": f"p{i}"} for i in range(1, 13)]
    out = rr.rerank("q", items, llm=_Pick(), depth=4)
    assert [x["text"] for x in out[:2]] == ["p2", "p1"]
    assert [x["text"] for x in out[4:]] == ["p5", "p6", "p7", "p8",
                                           "p9", "p10", "p11", "p12"]


# ---------------------------------------------------------------- 接线

def test_tools_takes_deeper_candidates_and_truncates_back():
    """**防"写了没接"**，而且要接对两处：

    · 候选要**取深**（只取 top_k 的话重排没有腾挪余地，退化成 no-op）
    · 重排后要**截回 top_k**（不截会让提示词里多塞一倍内容）
    """
    from app.agent import tools

    src = inspect.getsource(tools.retrieve_knowledge)
    assert "rerank" in src, "tools 层没接重排"
    assert "max(top_k, 10)" in src, "候选没取深，重排会退化成 no-op"
    assert "results[:top_k]" in src, "重排后没有截回 top_k"
    # 截断必须在重排之后
    assert src.index("rerank(") < src.index("results[:top_k]")
