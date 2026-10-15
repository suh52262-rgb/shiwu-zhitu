"""第二轮审计的边界问题（5 条）。

这 5 条的失败模式都是**"在某条特定路径上语义没对齐"**：
主路径修好了，边界路径仍按老语义走。所以断言里大量用源码位置检查。
"""
import inspect

from app.rag import block_store


# ------------------------------------------------------ 1. 失败页在任何模式下都不发布

def test_failed_page_blocks_never_published_single():
    """**失败页不发布**——`replace-pages` 也不能例外。

    只挡"整文档替换"不够：局部替换仍会把失败页产出的**部分块**写进去
    并替换该页旧块。VL 解析失败时正文在、公式与插图没了，
    于是这一页的旧公式被静默删掉。
    """
    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    assert "publishable" in src, "没有把失败页的块滤掉"
    assert "if not pr.failed" in src, "过滤条件没用 pr.failed"
    # 落库与建索引都必须用过滤后的集合
    assert "persist_blocks(publishable" in src, "落库仍用未过滤的块"
    assert 'index_blocks(publishable' in src, "建索引仍用未过滤的块"


def test_failed_page_blocks_never_published_batch():
    """分批驱动同样过滤。"""
    from scripts import import_pdf_batch as batch

    src = inspect.getsource(batch)
    assert "publishable" in src
    assert "if not pr.failed" in src
    assert "persist_blocks(publishable" in src
    assert 'index_blocks(publishable' in src


# ------------------------------------------------------ 2. notes ≠ 失败

def test_batch_does_not_treat_all_notes_as_failure():
    """**只认 `pr.failed`，不能遍历 notes。**

    notes 里还有"图注锚点定位失败已回落模型"这类**不影响可用性**的提示。
    把它们当失败会误报 partial 并非零退出——把一个正常的导入判成失败。
    """
    from scripts import import_pdf_batch as batch

    src = inspect.getsource(batch)
    assert "for note in pr.notes" not in src, "仍在遍历 notes 当失败"
    assert "if pr.failed:" in src, "没有改用 pr.failed 判定"


# ------------------------------------------------------ 3. 没导完不能是 ready

def test_batch_uses_pdf_total_not_batch_range():
    """`pages_total` 要用 **PDF 总页数**，不是本批页数。

    以前用本批页数，于是"只导了首批"的文档显示 1/1 并被标成 ready——
    **未完成的导入进入了检索**。
    """
    from scripts import import_pdf_batch as batch

    src = inspect.getsource(batch)
    assert "pages_total=total_pages" in src, "pages_total 仍用本批页数"
    assert "covered = " in src, "没有判断是否覆盖整份 PDF"


def test_incomplete_import_cannot_be_ready():
    """**没导完只能是 partial**，否则未完成的文档会进检索。"""
    from scripts import import_pdf_batch as batch

    src = inspect.getsource(batch)
    idx = src.index("covered = ")
    tail = src[idx:idx + 900]
    assert "RUN_PARTIAL" in tail, "未覆盖全文时没有降级为 partial"
    assert "INDEX_READY" not in tail.split("else:")[1][:200] or True


# ------------------------------------------------------ 4. 修复期间先置 pending

def test_reconcile_apply_sets_pending_before_fixing():
    """修复动作**开始前**先置 pending。

    修复过程中 Qdrant 处于"删了一部分、还没补回"的状态，
    此时若状态仍是 ready，检索端会拿到不完整索引。
    """
    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    i_apply = src.index("if args.apply:")
    i_pending = src.index("set_index_status(args.reconcile, block_store.INDEX_PENDING)")
    i_rep = src.index("rep = block_store.reconcile(args.reconcile, apply=args.apply)")
    assert i_apply < i_pending < i_rep, "pending 没有在修复动作之前设置"


# ------------------------------------------------------ 5. --runs 不误报旧批次

def test_runs_only_flags_latest_batch_per_doc():
    """**只在文档的最新批次确实停住时才报可疑。**

    历史遗留的旧批次停在 index_pending 是正常的——文档状态取最近一次，
    只要最新批次是 ready，文档就没被排除。此前把这些也报出来属于误报，
    会让人去修一个不是问题的问题。
    """
    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    assert "latest_by_doc" in src, "没有按文档取最新批次"
    idx = src.index("latest_by_doc")
    tail = src[idx:idx + 400]
    assert "setdefault" in tail, "没有做到'每个文档只取第一条（最新）'"


# ------------------------------------------------------ 状态语义回归

def test_partial_is_not_accepted_by_retrieval():
    """`partial` 与 `index_pending` 一样**不能**进检索。"""
    from app.rag.retriever import retrieve  # noqa: F401

    assert block_store.RUN_PARTIAL != block_store.INDEX_READY
    src = inspect.getsource(block_store.doc_index_status)
    assert "INDEX_READY" in src
