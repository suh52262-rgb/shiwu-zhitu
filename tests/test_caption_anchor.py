"""图注锚点插图定位（不依赖模型）。

这一路的动机来自 40 页抽样的结论：**bbox 定位是整个流程最不可靠的一环**
（两家模型的两两一致率平均只有 IoU 0.563），而图注位置是从文字层
**精确**拿到的。

判定指标用**包含率**而不是 IoU：第 47 页上 qwen 的 IoU 最高（0.91）
但包含率只有 91%（切掉了 9% 的图）；这一路 IoU 0.72 但包含率 100%。
裁图是给学生看的——**切掉图比留白严重得多**。
"""
from pathlib import Path
import pymupdf
import pytest

from app.rag.pdf_layout import find_captions, locate_figure_by_caption, text_lines

PDF = sorted(__import__("pathlib").Path(str(Path(__file__).resolve().parents[1] / 'data/textbooks'))
             .glob("教材-3-*.pdf"))
pytestmark = [pytest.mark.local_asset, pytest.mark.skipif(not PDF, reason="教材 PDF 不在本机")]


@pytest.fixture(scope="module")
def doc():
    with pymupdf.open(str(PDF[0])) as d:
        yield d


def _truth_p47(page) -> list[float]:
    """第 47 页图1-17 的**人工核对过**的正解（归一化 0~1000 → 页坐标）。"""
    W, H = page.rect.width, page.rect.height
    return [495 / 1000 * W, 293 / 1000 * H, 842 / 1000 * W, 392 / 1000 * H]


def _contains(outer, inner) -> float:
    x0, y0 = max(outer[0], inner[0]), max(outer[1], inner[1])
    x1, y1 = min(outer[2], inner[2]), min(outer[3], inner[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / max(1e-6, (inner[2]-inner[0]) * (inner[3]-inner[1]))


def test_never_clips_the_figure_on_the_calibration_page(doc):
    """**最关键的一条**：推出的区域必须完整包住图。

    这正是模型做不到的——qwen 的框 IoU 最高却切掉了 9% 的图。
    """
    page = doc[46]
    cap = next(c for c in find_captions(page) if "1-17" in c["text"] or "1－17" in c["text"])
    box = locate_figure_by_caption(page, cap["bbox"])
    assert box is not None
    assert _contains(box, _truth_p47(page)) == pytest.approx(1.0), "切掉了图的一部分"


def test_top_edge_lands_on_the_crossing_line(doc):
    """上边界应当落在"正文重新横穿"的那一行的下沿——实测能对到 1pt 以内。"""
    page = doc[46]
    cap = next(c for c in find_captions(page) if "1-17" in c["text"] or "1－17" in c["text"])
    box = locate_figure_by_caption(page, cap["bbox"])
    assert abs(box[1] - _truth_p47(page)[1]) < 3.0, "上边界没对准横穿行"


def test_bottom_edge_is_the_caption_top(doc):
    page = doc[46]
    cap = next(c for c in find_captions(page) if "1-17" in c["text"] or "1－17" in c["text"])
    box = locate_figure_by_caption(page, cap["bbox"])
    assert box[3] == pytest.approx(cap["bbox"][1], abs=0.5)


def test_covers_every_real_caption_in_the_sample(doc):
    """12 页抽样里 16 条图注**全部**能推出区域（含此前失败的三种情况）。

    三种曾经的失败原因都已处理：图太大（改用页面比例设限）、
    图太小（下限降到 20pt）、图注其实是正文引用（提前过滤）。
    """
    pages = [47, 146, 152, 323, 404, 200, 260, 401]
    total = ok = 0
    for pno in pages:
        page = doc[pno - 1]
        for cap in find_captions(page):
            total += 1
            if locate_figure_by_caption(page, cap["bbox"]) is not None:
                ok += 1
    assert total >= 12, f"样本太少（{total} 条），断言没有意义"
    assert ok == total, f"{total - ok}/{total} 条图注推不出区域"


def test_junk_captions_from_prose_are_filtered(doc):
    """正文里的引用（如 "图6-70)"）不该被当成图注。

    它推出来的"图"只有 5pt 高——不排掉就会产出垃圾裁图。
    """
    page = doc[322]                     # 第 323 页
    texts = [c["text"] for c in find_captions(page)]
    assert not any(t.rstrip().endswith(")") for t in texts), texts


def test_returns_none_when_there_is_no_crossing_line():
    """推不出时必须返回 None（交给模型兜底），而不是硬给一个错框。"""
    class FakePage:
        rect = pymupdf.Rect(0, 0, 1000, 1360)

    # 页面上一条横穿行都没有
    import app.rag.pdf_layout as mod
    original = mod.text_lines
    mod.text_lines = lambda page: []
    try:
        assert locate_figure_by_caption(FakePage(), [500, 800, 800, 820]) is None
    finally:
        mod.text_lines = original


def test_lines_are_line_level_not_block_level(doc):
    """必须用行级 bbox——块级会把公式行与正文合并，实测过。"""
    page = doc[46]
    lines = text_lines(page)
    assert lines
    # 行高普遍在 20~40pt；块级会出现 70pt 以上的合并块
    heights = [l["bbox"][3] - l["bbox"][1] for l in lines]
    assert max(heights) < 70, f"出现了疑似块级的合并行：{max(heights):.0f}pt"
