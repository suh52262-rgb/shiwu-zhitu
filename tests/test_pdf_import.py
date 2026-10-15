"""PDF 导入链路的测试。

分三层：
  1. `formula_text` —— 纯函数，最容易测也最容易写错（替换顺序踩过坑）
  2. `pdf_layout`   —— 另见 test_caption_anchor.py
  3. `pdf_import`   —— 块的组装与幂等键
"""
import sys
import types
from pathlib import Path

import pytest

from app.rag import pdf_import
from app.rag.formula_text import formula_embed_text, latex_to_readable


# ---------------------------------------------------------------- 公式可读化

def test_fraction_is_converted_through_wrapping_commands():
    """`\\boldsymbol{}` 之类的零宽包装必须先去掉，否则 `\\frac` 匹配不到。

    实测踩过：`\\frac{d\\boldsymbol{P}}{dt}` 输出成 `\\frac d P dt`
    ——外层花括号含嵌套，`\\{([^{}]*)\\}` 匹配不上。
    """
    out = latex_to_readable(r"\frac{d\boldsymbol{P}}{dt}")
    assert "\\frac" not in out, out
    assert "dP" in out and "dt" in out, out


def test_function_names_get_spaces():
    """`x\\tan\\alpha` 不能粘成 `xtanα`——那样检索时是噪声。"""
    out = latex_to_readable(r"x\tan\alpha")
    assert "tan" in out and "α" in out
    assert "xtan" not in out, out


def test_subscripts_and_greek():
    out = latex_to_readable(r"v_0^2\cos^2\alpha")
    assert "v_0^2" in out
    assert "cos" in out and "α" in out


def test_environments_are_stripped():
    out = latex_to_readable(
        r"\left\{\begin{aligned}x&=(v_0\cos\alpha)t,\\y&=(v_0\sin\alpha)t.\end{aligned}\right."
    )
    assert "begin" not in out and "aligned" not in out and "&" not in out, out
    assert "cos" in out and "sin" in out


def test_no_leftover_backslashes():
    """收尾必须清干净，否则检索串里全是 `\\` 噪声。"""
    for src in (r"\left\{x\right.", r"\sqrt{S}", r"\frac{a}{b}", r"\propto"):
        assert "\\" not in latex_to_readable(src), src


def test_empty_input_is_safe():
    assert latex_to_readable("") == ""
    assert latex_to_readable("$$$$") == ""


def test_embed_text_carries_page_chapter_and_number():
    """编号是公式与正文互相引用的锚点（正文会写"由式(1.15)"），必须带上。"""
    out = formula_embed_text(r"y=x\tan\alpha", page=47,
                             chapter="第一章质点运动学", number="(1.15)")
    assert "公式(1.15)" in out
    assert "第47页" in out
    assert "第一章质点运动学" in out


def test_embed_text_without_number_still_has_context():
    out = formula_embed_text(r"y=x", page=47)
    assert "公式" in out and "第47页" in out


# ---------------------------------------------------------------- 幂等键

def test_block_id_is_stable_and_unique():
    """uuid5 保证重导覆盖而不是累积——这是幂等入库的前提。"""
    a = pdf_import.block_id("doc", 1, 0)
    assert a == pdf_import.block_id("doc", 1, 0)
    assert a != pdf_import.block_id("doc", 1, 1)
    assert a != pdf_import.block_id("doc", 2, 0)
    assert a != pdf_import.block_id("other", 1, 0)


# ---------------------------------------------------------------- 块组装

class _FakePage:
    """够用的假页面：两行正文 + 一条图注，不含插图。"""

    rect = types.SimpleNamespace(width=1000.0, height=1360.0)

    def __init__(self, lines):
        self._lines = lines

    def get_text(self, kind="dict"):
        return {"blocks": [{"type": 0, "lines": [
            {"bbox": b, "spans": [{"text": t}]} for t, b in self._lines]}]}

    def get_images(self, full=True):
        return []

    def get_pixmap(self, clip=None, dpi=150):
        raise AssertionError("本用例不该裁图")


def test_collect_page_marks_source_kind(monkeypatch):
    """`source_kind` 是排障的关键：一眼看出这条内容哪来的、可信度如何。"""
    page = _FakePage([
        ("图1-17抛体运动的轨迹", [500, 500, 800, 520]),
        ("这是正文", [100, 100, 400, 120]),
    ])
    monkeypatch.setattr(pdf_import, "page_kind", lambda p: "scanned_with_ocr")
    pr = pdf_import.collect_page(page, 47, doc_id="d", use_vl=False)
    kinds = {b["source_kind"] for b in pr.blocks}
    assert kinds <= {"text_layer", "caption_anchor"}, kinds


def test_collect_page_never_sets_is_generated_for_text_layer(monkeypatch):
    """文字层来的内容不是机器生成——这个标记错了会让 Tutor 引假依据。"""
    page = _FakePage([("这是正文", [100, 100, 400, 120])])
    monkeypatch.setattr(pdf_import, "page_kind", lambda p: "scanned_with_ocr")
    pr = pdf_import.collect_page(page, 1, doc_id="d", use_vl=False)
    assert all(b.get("is_generated", False) is False for b in pr.blocks)


def test_every_block_carries_doc_id(monkeypatch):
    """**块本身必须带 doc_id**。

    Qdrant 的按文档过滤与按文档删除都读 payload.doc_id；
    只放在结果信封里的话，按 doc_id 查不到任何点（踩过这个坑）。
    """
    page = _FakePage([("这是正文", [100, 100, 400, 120])])
    monkeypatch.setattr(pdf_import, "page_kind", lambda p: "scanned_with_ocr")
    pr = pdf_import.collect_page(page, 1, doc_id="my-doc", use_vl=False)
    assert pr.blocks
    assert all(b["doc_id"] == "my-doc" for b in pr.blocks)


def test_figure_number_extraction():
    """配对要用图号，不能拿整串比——VL 给 `图1-17`，图注是 `图1-17抛体运动的轨迹`。"""
    assert pdf_import._figure_number("图1-17抛体运动的轨迹") == "1-17"
    assert pdf_import._figure_number("图 3－28") == "3-28"
    assert pdf_import._figure_number("没有图号") == ""


def test_content_hash_changes_with_content():
    a = [{"text": "甲", "embed_text": "甲"}]
    b = [{"text": "乙", "embed_text": "乙"}]
    assert pdf_import.content_hash(a) != pdf_import.content_hash(b)
    assert pdf_import.content_hash(a) == pdf_import.content_hash(a)


def test_page_kind_distinguishes_native_and_scanned():
    class P:
        rect = types.SimpleNamespace(width=100, height=100)

        def __init__(self, images, text):
            self._img, self._text = images, text

        def get_images(self, full=True):
            return self._img

        def get_image_rects(self, xref):
            return [types.SimpleNamespace(width=100, height=100)]

        def get_text(self, kind="text"):
            return self._text

    assert pdf_import.page_kind(P([(1,)], "很长的文字" * 60)) == "scanned_with_ocr"
    assert pdf_import.page_kind(P([(1,)], "")) == "scanned_no_text"
    assert pdf_import.page_kind(P([], "文字" * 10)) == "native_text"
    assert pdf_import.page_kind(P([], "")) == "empty"
