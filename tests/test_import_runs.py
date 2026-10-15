"""导入批次与索引状态。

## 这一组测的是「索引过期时检索端知道」

`--no-index` 只写 PG 不写 Qdrant；导入中途失败时 PG 是新的、Qdrant 是部分的。
两种情况检索端都**必须排除该文档**，否则会把旧向量当最新结果返回。

另一个是**恢复路径**：一次崩溃让文档停在 `index_pending`，
如果 `--reindex` 不把状态置回 `ready`，文档会被**永久**排除——
而它其实已经修好了。
"""
import inspect

import pytest

from app.rag import block_store
from app.rag.block_store import (
    INDEX_PENDING,
    INDEX_READY,
    RUN_FAILED,
    RUN_PARTIAL,
    doc_index_status,
    finish_import_run,
    list_import_runs,
    set_index_status,
    start_import_run,
)


# ---------------------------------------------------------------- 状态语义

def test_ready_is_the_only_status_retrieval_accepts():
    """检索端**只认 ready**。这条是契约，写死成常量便于全局引用。"""
    assert INDEX_READY == "ready"
    assert INDEX_PENDING != INDEX_READY


def test_missing_run_record_defaults_to_ready():
    """没有批次记录的文档按 `ready` 处理。

    **这是刻意的保守选择**：早期导入的文档没有批次记录，
    把它们一律当成"待修复"会让检索整体失败；而它们当时确实可用。
    """
    assert doc_index_status("no-such-doc-xyz") == INDEX_READY


def test_unknown_doc_status_does_not_raise():
    """状态查不到/数据库不可用时不能抛——检索不该因此挂掉。"""
    assert isinstance(doc_index_status("whatever"), str)


# ---------------------------------------------------------------- 批次记录

@pytest.mark.real_db
def test_start_run_marks_doc_pending():
    """**导入一开始就要标 pending**。

    因为这一刻起 PG 就要变了，而 Qdrant 还没跟上——
    检索端必须立刻看到"这批还没好"，而不是等导入结束。
    """
    run_id = start_import_run("status-test-doc", source_sha256="a" * 64,
                              source_name="x.pdf", mode="document",
                              model="test-model")
    assert run_id
    assert doc_index_status("status-test-doc") == INDEX_PENDING

    finish_import_run(run_id, status=RUN_FAILED, pages_total=10, pages_ok=8,
                      failed_pages=[3, 7], error="test")
    assert doc_index_status("status-test-doc") == RUN_FAILED


@pytest.mark.real_db
def test_run_record_persists_sha_and_model():
    """SHA-256 与模型版本要**落库**，不能只打印在控制台。

    否则事后无法回答"PG 里这批内容来自哪个版本的 PDF、用的哪个模型"——
    同名文件内容变了也看不出来。
    """
    digest = "b" * 64
    run_id = start_import_run("sha-test-doc", source_sha256=digest,
                              source_name="book.pdf", mode="pages",
                              pages_spec="47",
                              model="deepseek-flash",
                              prompt_versions={"page_reader": "v1"})
    finish_import_run(run_id, status=INDEX_READY, pages_total=1, pages_ok=1)

    runs = list_import_runs("sha-test-doc", limit=1)
    assert runs, "批次记录没落库"
    r = runs[0]
    assert r["sha256"] == digest[:16]
    assert r["model"] == "deepseek-flash"
    assert r["mode"] == "pages"
    assert r["pages_spec"] == "47"
    assert r["status"] == INDEX_READY


@pytest.mark.real_db
def test_failed_pages_are_recorded():
    run_id = start_import_run("failpages-doc")
    finish_import_run(run_id, status=RUN_PARTIAL, pages_total=5, pages_ok=3,
                      failed_pages=[2, 4])
    r = list_import_runs("failpages-doc", limit=1)[0]
    assert r["pages_failed"] == 2
    assert r["failed_pages"] == [2, 4]


@pytest.mark.real_db
def test_finish_unknown_run_is_ignored():
    """收尾一个不存在的批次不该抛——它可能已经被清理了。"""
    finish_import_run("no-such-run-id", status=INDEX_READY)


@pytest.mark.real_db
def test_latest_run_wins():
    """文档状态取**最近一次**批次。

    崩溃的旧批次停在 `index_pending` 是正常的（它确实没完成），
    不该影响已经成功的新批次。
    """
    start_import_run("latest-doc")                       # 假装崩了
    rid = start_import_run("latest-doc")
    finish_import_run(rid, status=INDEX_READY)
    assert doc_index_status("latest-doc") == INDEX_READY


@pytest.mark.real_db
def test_set_index_status_restores_ready():
    """**恢复路径**：`--reindex` 修好后必须能把状态置回 ready。

    不接这条路的话，一次崩溃会让文档**永久**被检索排除。
    """
    rid = start_import_run("recover-doc")
    finish_import_run(rid, status=INDEX_PENDING)
    assert doc_index_status("recover-doc") == INDEX_PENDING

    set_index_status("recover-doc", INDEX_READY)
    assert doc_index_status("recover-doc") == INDEX_READY


@pytest.mark.real_db
def test_set_index_status_on_unknown_doc_is_noop():
    set_index_status("never-imported-xyz", INDEX_READY)   # 不该抛


# ---------------------------------------------------------------- 接线断言

def test_retriever_excludes_pending_documents():
    """**防"写了没接"**：检索端必须真的按状态过滤。

    这类失败模式（代码写了但没接上）普通单元测试测不出来——
    上一轮的 `verify_figure` 就是这么漏掉的。
    """
    from app.rag import retriever

    src = inspect.getsource(retriever.retrieve)
    assert "doc_index_status" in src, "检索端没有查文档索引状态"
    # 判定已从「只认 ready」的白名单改为**黑名单** `is_index_visible`——
    # 白名单把 `partial` 也排除了，实测导致 2 本书 963 页搜不到。
    assert "is_index_visible" in src, "检索端没有用可见性判定"
    assert "== _bs.INDEX_READY" not in src, "检索端还在用旧的白名单"


def test_cli_marks_pending_on_no_index_and_ready_on_reindex():
    """`--no-index` 要标 pending；`--reindex` 要标 ready。"""
    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    assert "INDEX_PENDING if args.no_index" in src or "args.no_index" in src
    assert "set_index_status" in src, "CLI 没有恢复状态的调用点"
    assert src.count("set_index_status") >= 2, "reindex 与 reconcile 都应能恢复状态"


# ---------------------------------------------------------------- 审计发现的四条

def test_batch_driver_wires_run_state():
    """**防"写了没接"**：分批驱动也必须开/收批次。

    它此前零调用点——于是 `--no-index` 或 Qdrant 写失败时文档状态停在旧的
    ready，检索端继续用旧向量；有失败页也返回 0。
    """
    from scripts import import_pdf_batch as batch

    src = inspect.getsource(batch)
    assert "start_import_run" in src, "分批驱动没开批次"
    assert "finish_import_run" in src, "分批驱动没收尾批次"
    assert "return 3" in src, "分批驱动有失败页却仍返回 0"


def test_run_is_created_after_lock_not_before():
    """**先取锁、再建批次。**

    反过来的话，锁被占用时异常分支直接返回，那个刚建出来的
    index_pending 批次永远不会收尾——**原本可用的文档会从检索里消失**。
    （数据库里那两条 0 页的 index_pending 记录就是证据。）
    """
    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    i_lock = src.index("with import_lock(doc_id)")
    i_run = src.index("start_import_run")
    assert i_run > i_lock, "start_import_run 出现在 import_lock 之前"


def test_vl_parse_error_marks_page_failed():
    """VL 解析失败必须标 failed，不能只记警告。

    这页的正文还在但**公式与插图全丢**；整文档替换会拿"有正文没公式"的
    新内容覆盖"都全"的旧内容——静默的数据丢失。
    """
    from app.rag import pdf_import

    src = inspect.getsource(pdf_import.collect_page)
    idx = src.index("parse_error")
    tail = src[idx:idx + 600]
    assert "result.failed = True" in tail, "parse_error 分支没有标 failed"


def test_reindex_marks_pending_before_rebuild():
    """重建**前**先标 pending。

    重建过程中 Qdrant 处于"删了一部分、还没写回"的状态，
    此时若状态仍是 ready，检索端会拿到部分索引。
    """
    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    i_pending = src.index("set_index_status(args.reindex, block_store.INDEX_PENDING)")
    i_rebuild = src.index("index_blocks(blocks", i_pending - 400)
    assert i_pending < i_rebuild, "pending 没有在重建之前设置"


def test_reconcile_apply_rechecks_and_sets_ready():
    """`--apply` 修完差异后要**重新判定**再置 ready。

    此前只在"原本就无差异"时置 ready——于是修好了差异的文档反而
    停在 index_pending 被检索排除。
    """
    from scripts import import_pdf as cli

    src = inspect.getsource(cli)
    assert "after = block_store.reconcile(args.reconcile, apply=False)" in src, \
        "apply 之后没有复查"
    assert "return 4" in src, "复查仍有差异时没有非零退出"
