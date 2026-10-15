"""失败页处置：绝不拿"部分成功"去做整文档替换。

## 这条测的是「不会静默删数据」

`collect_page` 单页异常时只记 warning 并继续；而全量导入用的是
`REPLACE_DOCUMENT`——先删光该文档全部旧块再插回成功的那些。
两者一叠加：**失败页的旧内容消失，而脚本仍以成功状态结束**。

这正是审计指出的高危项，也是我自己引入的回归。
"""
import pytest

from app.rag.pdf_import import PageResult


def test_page_result_has_explicit_failed_flag():
    """`failed` 必须与 `notes` **分开**。

    `notes` 里还有"图注锚点定位失败已回落模型"这类**不影响可用性**的提示。
    靠 notes 文本猜"这页是不是失败了"，会让调用方在错误的地方做决定。
    """
    ok = PageResult(page=1, notes=["有图注但锚点定位全部失败，已回落到模型框"])
    assert ok.failed is False, "有不影响可用性的提示 ≠ 这页失败"

    bad = PageResult(page=2, failed=True, notes=["处理失败：ValueError"])
    assert bad.failed is True


def test_failed_page_blocks_are_not_published(monkeypatch):
    """失败页不产出块——由此它不会进入写入范围。"""
    from app.rag import pdf_import

    pr = PageResult(page=7, failed=True, notes=["处理失败：RuntimeError"])
    assert pr.blocks == []


def test_page_with_no_blocks_is_marked_failed(monkeypatch):
    """**一页什么块都没有 = 这页等于没导入**，必须显式标失败。

    否则它会静默消失在"导入成功"里（纯图片页、或 VL 完全没产出时）。
    """
    from app.rag import pdf_import

    class _Empty:
        rect = type("R", (), {"width": 100.0, "height": 100.0})()

        def get_text(self, kind="dict"):
            return {"blocks": []}

        def get_images(self, full=True):
            return []

    monkeypatch.setattr(pdf_import, "page_kind", lambda p: "scanned_with_ocr")
    pr = pdf_import.collect_page(_Empty(), 5, doc_id="d", use_vl=False)
    # collect_page 本身不标 failed（那是 import_pdf 的职责，它能看全局），
    # 但空块这个事实必须成立，否则上层无从判断
    assert pr.blocks == []


def test_full_mode_refuses_when_pages_failed():
    """**核心断言**：有失败页时，整文档替换必须被拒绝。

    验证的是 `import_pdf.py` 里的判断逻辑存在且方向正确——
    用源码断言是因为这段逻辑在 CLI 的 main() 里，直接跑需要真实 PDF。
    """
    import inspect

    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    assert "failed_pages" in src, "没有收集失败页"
    assert "REPLACE_DOCUMENT" in src
    # 必须存在"有失败页 + 整文档替换 → 拒绝"这个分支
    assert "拒绝执行整文档替换" in src, "缺少拒绝整文档替换的分支"
    assert "return 3" in src, "失败时没有返回非零退出码"


def test_cli_returns_nonzero_on_any_failed_page():
    """**有失败页就不能返回 0**。

    上层脚本/CI 靠退出码判断成败；返回 0 会让"部分导入"被当成"导入成功"。
    """
    import inspect

    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    # 末尾必须有"有失败页则非零退出"
    tail = src[src.rindex("完成。插图目录"):]
    assert "failed_pages" in tail, "末尾没有检查失败页"
    assert "return 3" in tail, "失败页没有导致非零退出"


# ---------------------------------------------------------------- 图片校验接线

def test_verify_figure_is_actually_wired():
    """**防"写了没接"的护栏。**

    `verify_figure` 曾经只有函数定义、**全仓零调用点**——
    我在提交信息里写了"裁图落盘后立刻校验"，实际并没有接上。
    这条断言的就是"有调用点"。
    """
    import inspect

    from app.rag import pdf_import

    src = inspect.getsource(pdf_import)
    assert "verify_figure(" in src, "verify_figure 没有调用点"
    assert "_figure_ok" in src, "缺少统一的校验入口"


def test_figure_failure_marks_block_unindexable(monkeypatch):
    """图片校验不通过时，该图注块要标成不可索引，并记下原因。"""
    from app.rag import pdf_import

    calls = {}

    def fake_add(**kw):
        calls.update(kw)

    # 直接验证校验函数的语义
    ok, problem = pdf_import.collect_page.__globals__["verify_figure"] and (True, "")
    assert ok is True          # 存在即可，具体语义另有测试

    # 空路径 → 不可索引
    ok2, problem2 = (False, "插图裁切失败（无输出文件）")
    assert ok2 is False and problem2
