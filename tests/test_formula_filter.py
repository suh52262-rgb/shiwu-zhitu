"""扫描件文字层里的坏公式隔离。

## 这一组测的是「不会把乱码当教材原文返回」

这版教材的文字层来自 OCR，公式基本不可用。实测前 200 页有 96 行这种内容
被当作正文入库：`'s = (vo +v)t =( ++at)'`（应为 `s = v₀t + ½at²`）。
它们和 VL 生成的正规 LaTeX 一起进了向量库，检索时会被当作「教材原文」。

处理方式：**原文保留在 PG，只标 `indexable=False`**。
"""
import pytest

from app.rag.formula_filter import classify_text_line, looks_like_broken_formula


# ---------------------------------------------------------------- 应当拦下

@pytest.mark.parametrize("line", [
    "s = (vo +v)t =( ++at)",              # 应为 s = v₀t + ½at²
    "11.y.=9.460530×10l*m~10l°m.1010",
    "1AU = 1.495 978 92 ×10m,",
    "= 2. 062648 × 10°AU",
    "787× 960",
    "a=",
    "α - lα|= lim",
    "3. 99 × 10 -2",
    "1* = 元/180×60 ×60",
])
def test_broken_formulas_are_detected(line):
    assert looks_like_broken_formula(line), line


# ---------------------------------------------------------------- 不能误伤

@pytest.mark.parametrize("line", [
    "这是抛物线方程（见图1－17中的黑线轨迹，而灰线轨迹是考虑到空气阻力",
    "根据牛顿第二定律：",
    "牛顿第二定律",
    "图1-17抛体运动的轨迹",
    # ↓ 下面这几条是**实测误伤过的**，加了"汉字绝对个数"规则才分开
    "令(1.14)和(1.15)式中的g=0.,则有",
    "R=6370km，R。=R-2900km=3470km.将已知数据代入,可解出",
    "1－10.在一段时间间隔△t内（1）Ar=0，（2）△r=0，（3）4s=0,",
    "在两极处，±=±π/2，cos±=0，W=W,视重最大；",
])
def test_prose_is_not_blocked(line):
    """**误伤就是内容丢失**。

    `'令(1.14)和(1.15)式中的g=0.,则有'` 的汉字占比是 7/24 = 0.29，
    刚好压在 0.30 的占比阈值下——只靠占比必然误伤。
    真正的坏公式汉字数是 0~2，用**绝对个数**才分得开。
    """
    assert not looks_like_broken_formula(line), line


def test_formula_numbers_are_kept():
    """`(1.15)` 这类纯编号是**有用**的检索锚点（正文会写"由式(1.15)"），
    而且它不"错"，所以要放行。"""
    assert not looks_like_broken_formula("(1.15)")
    assert not looks_like_broken_formula("（1.16）")


def test_long_paragraphs_are_never_flagged():
    """长段落不判——公式碎片都是短行。"""
    long_line = "在只有重力或弹力做功的情形下，" * 10 + "E=常量"
    assert not looks_like_broken_formula(long_line)


# ---------------------------------------------------------------- 门控

def test_filter_only_applies_to_scanned_pages():
    """**原生文字页不拦**。

    原生页的公式是排版工具正常导出的（虽然也可能不准，但不是 OCR 的锅），
    拦掉会误伤。这个函数只在扫描件上生效。
    """
    line = "s = (vo +v)t =( ++at)"
    assert classify_text_line(line, is_scanned=False) == (True, "")
    ok, reason = classify_text_line(line, is_scanned=True)
    assert ok is False
    assert reason, "拦掉时必须给出原因——静默的过滤器比不过滤更危险"


def test_classify_returns_reason_for_traceability():
    """`note` 要能说明"为什么没索引"，否则事后无法统计拦了多少、都是什么。"""
    ok, reason = classify_text_line("787× 960", is_scanned=True)
    assert ok is False
    assert "坏公式" in reason or "隔离" in reason


def test_empty_input_is_safe():
    assert not looks_like_broken_formula("")
    assert not looks_like_broken_formula("   ")
    assert classify_text_line("", is_scanned=True) == (True, "")
