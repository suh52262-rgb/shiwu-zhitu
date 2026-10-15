"""检索的上下文扩展（解决正文块碎片化）。

## 背景

全量导入后正文块长度**中位数只有 12 字、43% 不足 10 字**
（`'动量守恒'`、`'1. 角动量'`）。这种块命中得再准，也**给 Tutor
提供不了上下文**——它没法据此讲清一句话。

## 为什么扩「返回的文本」而不是「被向量化的文本」

先试过在**索引时**把同页邻块拼进 `embed_text`：正文检索变好了，
但**破坏了公式检索**——几个邻块拼进同一个向量会让它代表"一堆话题的
混合"，具体查询的信号被稀释（"机械能守恒的条件"变成命中 `l > l_*`）。

所以：`embed_text` 保持干净（向量质量不受影响），
返回时再从 PostgreSQL 取同页邻块。这正是两层设计该发挥作用的地方——
**向量库负责找得准，事实库负责给得全**。
"""
import pytest

from app.rag.retriever import (
    HEADING_CONTEXT,
    _expand_context,
    _looks_like_content,
)


# ---------------------------------------------------------------- 噪声过滤

def test_toc_entries_are_filtered_out():
    """目录条目不能拼进上下文。

    实测 `'机械能守恒'`（p14）的邻块是
    `'91 习题 93 第三章 机械能守恒 98 81.功和能· 98 1.1历史性的评述'`
    ——那是目录，拼进去只会误导 Tutor。
    """
    assert not _looks_like_content("91 习题 93 第三章 98")
    assert not _looks_like_content("151 81.角动量守恒 151")
    assert not _looks_like_content("12")


def test_real_content_is_kept():
    assert _looks_like_content("在只有重力或弹力做功的情形下，机械能保持不变。")
    assert _looks_like_content("图1-17抛体运动的轨迹")
    # 长句即使数字多也算内容
    assert _looks_like_content("质量 m=1200 kg 的汽车以 v0=15.0 m/s 撞向固定壁。")


def test_pure_digits_and_short_junk_rejected():
    assert not _looks_like_content("")
    assert not _looks_like_content("   ")
    assert not _looks_like_content("26")
    assert not _looks_like_content("(1.15)")


# ---------------------------------------------------------------- 扩展行为

def _hit(text, page=1):
    return {"text": text, "page": page, "source": "doc", "context": text}


def test_context_falls_back_to_text_when_no_facts(monkeypatch):
    """事实库取不到内容时，`context` 退化为 `text`——**不能因此让检索失败**。"""
    import app.rag.block_store as bs

    def boom(*a, **kw):
        raise RuntimeError("数据库不可用")

    monkeypatch.setattr(bs, "load_page_blocks", boom)
    hits = [_hit("动量守恒")]
    _expand_context(hits)
    assert hits[0]["context"] == "动量守恒"


def test_text_is_never_modified(monkeypatch):
    """`text` 必须保持原样——引用与展示都用它，只有喂 LLM 的上下文变长。"""
    import app.rag.block_store as bs

    monkeypatch.setattr(bs, "load_page_blocks", lambda doc, pages: {
        ("doc", 1): [
            {"block_index": 0, "type": "text", "text": "前面的一段完整正文内容"},
            {"block_index": 1, "type": "text", "text": "动量守恒"},
            {"block_index": 2, "type": "text", "text": "后面的一段完整正文内容"},
        ]
    })
    hits = [_hit("动量守恒")]
    _expand_context(hits)
    assert hits[0]["text"] == "动量守恒", "text 被改动了"
    assert "前面的一段" in hits[0]["context"]
    assert "后面的一段" in hits[0]["context"]


def test_headings_take_more_context_after_than_before(monkeypatch):
    """标题的邻块要**少取前、多取后**。

    实测 `'1. 角动量'`（p217）取前后对称邻块时，context 变成了上一节的
    陀螺章动——完全跑题。中文教材里标题后面跟的是本节内容。
    """
    import app.rag.block_store as bs

    blocks = [{"block_index": i, "type": "text",
               "text": f"前面第{i}段足够长的正文内容用来当邻块"} for i in range(6)]
    blocks.append({"block_index": 6, "type": "text", "text": "1. 角动量"})
    blocks += [{"block_index": 7 + i, "type": "text",
                "text": f"后面第{i}段足够长的正文内容用来当邻块"} for i in range(8)]
    monkeypatch.setattr(bs, "load_page_blocks", lambda doc, pages: {("doc", 1): blocks})

    hits = [_hit("1. 角动量")]
    _expand_context(hits)
    ctx = hits[0]["context"]
    n_before = sum(1 for i in range(6) if f"前面第{i}段" in ctx)
    n_after = sum(1 for i in range(8) if f"后面第{i}段" in ctx)

    assert n_after > n_before, f"标题没有偏向取后面的内容：前 {n_before} 后 {n_after}"
    assert n_before <= HEADING_CONTEXT[0]
    assert n_after >= 5


def test_long_hits_take_symmetric_context(monkeypatch):
    """正文命中用对称邻块——它本身已经是一段完整的话。"""
    import app.rag.block_store as bs

    blocks = [{"block_index": i, "type": "text",
               "text": f"前面第{i}段足够长的正文内容用来当邻块"} for i in range(5)]
    body = "这是一段足够长的正文命中块，本身已经是完整的话了。"
    blocks.append({"block_index": 5, "type": "text", "text": body})
    blocks += [{"block_index": 6 + i, "type": "text",
                "text": f"后面第{i}段足够长的正文内容用来当邻块"} for i in range(5)]
    monkeypatch.setattr(bs, "load_page_blocks", lambda doc, pages: {("doc", 1): blocks})

    hits = [_hit(body)]
    _expand_context(hits)
    ctx = hits[0]["context"]
    # 对称：前后各取 CONTEXT_NEIGHBORS 个（索引 1~4 与 6~9；0 与 10 落在窗口外）
    n_before = sum(1 for i in range(5) if f"前面第{i}段" in ctx)
    n_after = sum(1 for i in range(5) if f"后面第{i}段" in ctx)
    assert n_before == n_after == 4, f"不对称：前 {n_before} 后 {n_after}"


def test_context_is_length_capped(monkeypatch):
    """上下文要有上限——否则一整页都能灌进提示词。"""
    import app.rag.block_store as bs

    blocks = [{"block_index": i, "type": "text", "text": "长" * 200} for i in range(20)]
    blocks[10]["text"] = "动量守恒"
    monkeypatch.setattr(bs, "load_page_blocks", lambda doc, pages: {("doc", 1): blocks})

    hits = [_hit("动量守恒")]
    _expand_context(hits)
    from app.rag.retriever import CONTEXT_MAX_CHARS

    assert len(hits[0]["context"]) <= CONTEXT_MAX_CHARS


def test_next_page_is_loaded(monkeypatch):
    """标题落在页面底部时，本节内容在下一页——所以下一页也要载入。"""
    import app.rag.block_store as bs

    seen: list[list[int]] = []

    def fake(doc, pages):
        seen.append(list(pages))
        return {}

    monkeypatch.setattr(bs, "load_page_blocks", fake)
    _expand_context([_hit("1. 角动量", page=217)])
    assert seen, "没有请求任何页"
    assert 218 in seen[0], f"没有载入下一页：{seen[0]}"
