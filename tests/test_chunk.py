"""分块策略单元测试。"""
from app.rag.chunk import chunk_text


def test_short_text_returns_single_chunk():
    text = "动能定理：合外力做的功等于动能的变化。"
    assert chunk_text(text) == [text]


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_long_text_split_within_max_chars():
    text = "这是一段测试文本。" * 100  # 900 字
    chunks = chunk_text(text, max_chars=400, overlap=60)
    assert len(chunks) >= 2
    assert all(len(c) <= 400 for c in chunks)
    assert "".join(chunks).find("测试") != -1


def test_adjacent_chunks_overlap():
    text = "第一句讲受力分析。第二句讲动能定理。第三句讲机械能守恒。第四句讲摩擦力做功。" * 20
    chunks = chunk_text(text, max_chars=200, overlap=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        # 下一块的开头应与上一块的结尾有重叠
        assert nxt.startswith(prev[-40:]) or prev.endswith(nxt[:40]) or len(prev) < 40


def test_hard_split_for_unbroken_text():
    text = "a" * 1000  # 无句读边界
    chunks = chunk_text(text, max_chars=400, overlap=60)
    assert len(chunks) >= 3
    assert all(len(c) <= 400 for c in chunks)
