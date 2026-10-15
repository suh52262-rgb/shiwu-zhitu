"""证据块落点分类的测试 —— **身份缺失必须报「不可验证」，不能判「未投递」**。

## 为什么这组测试必须存在
```
`delivered_rag_sources()` 曾**不带 `id`** ⇒ 评测侧 `delivered` 集合恒空 ⇒
**每个块都被判成"未投递"** ⇒「线2 满足任一充分证据集 = 0」等结论**全是假的**，
而我把它当真实发现报了两次 ✗
⇒ 这组测试就是把那条教训**钉住**：三类落点必须分得开，
  且**身份缺失是一种独立结论**（不可验证），**不是"未投递"、也不是"未召回"**。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evidence_coverage import (  # noqa: E402
    DELIVERED, EXPANSION_MISSED, NOT_RECALLED, RANK_EXCLUDED, TRUNCATED,
    UNVERIFIABLE, classify_block, summarize,
)

TUTOR_HITS = 3


def _ctx(*ranks_to_ids):
    return {r: set(ids) for r, ids in ranks_to_ids}


def test_delivered_is_not_reported_as_truncated():
    """**已投递**必须判 delivered —— 这条就是当初那个假发现的回归测试。"""
    got = classify_block("A", delivered={"A"}, top_ctx={"A"},
                         ctx_by_rank=_ctx((1, ["A"])), order=["A"], tutor_hits=TUTOR_HITS)
    assert got == DELIVERED


def test_real_truncation_is_distinguishable():
    """**真实截断**：进了前 N 条目录，但不在投递片段里。"""
    got = classify_block("B", delivered={"A"}, top_ctx={"A", "B"},
                         ctx_by_rank=_ctx((1, ["A", "B"])), order=["A"], tutor_hits=TUTOR_HITS)
    assert got == TRUNCATED


def test_rank_excluded_is_not_truncation():
    """**名次被排除**：召回到了，但只在第 N+1 条的目录里 ⇒ 不是截断。"""
    got = classify_block("C", delivered={"A"}, top_ctx={"A"},
                         ctx_by_rank=_ctx((1, ["A"]), (4, ["C"])), order=["A", "X", "Y", "C"],
                         tutor_hits=TUTOR_HITS)
    assert got == RANK_EXCLUDED
    assert got != TRUNCATED


def test_expansion_missed_vs_not_recalled():
    """命中在列但**它自己的目录里没有它** ⇒ expansion_missed；否则 not_recalled。"""
    missed = classify_block("D", delivered={"A"}, top_ctx={"A"},
                            ctx_by_rank=_ctx((1, ["A"])), order=["A", "D"], tutor_hits=TUTOR_HITS)
    assert missed == EXPANSION_MISSED
    never = classify_block("E", delivered={"A"}, top_ctx={"A"},
                           ctx_by_rank=_ctx((1, ["A"])), order=["A"], tutor_hits=TUTOR_HITS)
    assert never == NOT_RECALLED


def test_missing_identity_is_unverifiable_not_a_miss():
    """🔴 **身份缺失 ⇒ 不可验证**（用户明确要求），**不得记成"未投递"**。"""
    for bad in (None, "", 0):
        got = classify_block(bad, delivered={"A"}, top_ctx={"A"},
                             ctx_by_rank=_ctx((1, ["A"])), order=["A"], tutor_hits=TUTOR_HITS)
        assert got == UNVERIFIABLE, f"block_id={bad!r} 应判不可验证"
        assert got not in (DELIVERED, TRUNCATED, NOT_RECALLED, RANK_EXCLUDED, EXPANSION_MISSED)


def test_summarize_keeps_categories_separate():
    """汇总时四类**不许合并**（合并就会把"不可验证"混进"未命中"）。"""
    s = summarize([DELIVERED, DELIVERED, TRUNCATED, UNVERIFIABLE, NOT_RECALLED])
    assert s == {DELIVERED: 2, TRUNCATED: 1, UNVERIFIABLE: 1, NOT_RECALLED: 1}
    assert s[UNVERIFIABLE] == 1 and s[NOT_RECALLED] == 1
