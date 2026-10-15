"""批次收尾：被取代的未完成批次要标记，不能留成"可疑"。

## 这条测的是「不误报，也不掩盖」

未完成的旧批次留着会让 `--runs` 一直报可疑，**而文档其实早就 ready 了**
——误报会让人去修一个不是问题的问题。
但也不能删：**"这次为什么没导完"恰恰是排障时要回答的问题**。
"""
import pytest
from app.rag import block_store
from app.rag.block_store import (
    INDEX_PENDING,
    INDEX_READY,
    RUN_RUNNING,
    RUN_SUPERSEDED,
    supersede_stale_runs,
)


def test_superseded_status_is_distinct():
    """`superseded` 必须与 `index_pending` 分开。

    合并成一个状态的话，就看不出"这是被取代的旧批次"还是
    "这份文档当前的索引确实待修复"——后者必须保持可见。
    """
    assert RUN_SUPERSEDED != INDEX_PENDING
    assert RUN_SUPERSEDED != RUN_RUNNING


@pytest.mark.real_db
def test_stale_runs_are_marked_not_deleted():
    """**标记而不是删除**——审计轨迹要留住。"""
    stale = block_store.start_import_run("supersede-doc")     # 假装没跑完
    fresh = block_store.start_import_run("supersede-doc")
    block_store.finish_import_run(fresh, status=INDEX_READY)

    n = supersede_stale_runs("supersede-doc")
    assert n == 1, f"应标记 1 条，实际 {n}"

    runs = {r["id"]: r for r in block_store.list_import_runs("supersede-doc")}
    assert stale in runs, "旧批次被删了——审计轨迹丢了"
    assert runs[stale]["status"] == RUN_SUPERSEDED
    assert runs[fresh]["status"] == INDEX_READY


@pytest.mark.real_db
def test_latest_run_is_never_superseded():
    """**最新的批次绝不标记**。

    它代表文档当前状态。哪怕它是 `index_pending`（索引确实待修复），
    也不能被掩盖成"已被取代"——那会让文档带着坏索引却看起来正常。
    """
    rid = block_store.start_import_run("only-pending-doc")
    block_store.finish_import_run(rid, status=INDEX_PENDING)

    n = supersede_stale_runs("only-pending-doc")
    assert n == 0, "把唯一的（也是最新的）批次标记了"
    assert block_store.doc_index_status("only-pending-doc") == INDEX_PENDING
    assert block_store.list_import_runs("only-pending-doc")[0]["status"] == INDEX_PENDING


@pytest.mark.real_db
def test_supersede_does_not_change_doc_status():
    """清理标记**不影响检索**——文档状态取的是最新批次。"""
    stale = block_store.start_import_run("status-keep-doc")
    fresh = block_store.start_import_run("status-keep-doc")
    block_store.finish_import_run(fresh, status=INDEX_READY)

    assert block_store.doc_index_status("status-keep-doc") == INDEX_READY
    supersede_stale_runs("status-keep-doc")
    assert block_store.doc_index_status("status-keep-doc") == INDEX_READY


@pytest.mark.real_db
def test_completed_runs_are_left_alone():
    """已经成功的旧批次不该被标 `superseded`——它没有"未完成"这回事。"""
    done = block_store.start_import_run("done-doc")
    block_store.finish_import_run(done, status=INDEX_READY)
    fresh = block_store.start_import_run("done-doc")
    block_store.finish_import_run(fresh, status=INDEX_READY)

    supersede_stale_runs("done-doc")
    runs = {r["id"]: r for r in block_store.list_import_runs("done-doc")}
    assert runs[done]["status"] == INDEX_READY, "成功的旧批次被误标"


@pytest.mark.real_db
def test_supersede_all_docs_when_no_doc_id():
    """不传 doc_id 时处理全部文档。"""
    for name in ("multi-a-doc", "multi-b-doc"):
        block_store.start_import_run(name)
        rid = block_store.start_import_run(name)
        block_store.finish_import_run(rid, status=INDEX_READY)
    n = supersede_stale_runs(None)
    assert n >= 2


@pytest.mark.real_db
def test_delete_import_run_is_available_for_test_cleanup():
    """删除函数存在即可用——但**正常流程不该用它**。

    未完成的批次应当 `supersede` 而不是删除；这个函数主要给测试收尾。
    """
    rid = block_store.start_import_run("delete-me-doc")
    assert block_store.delete_import_run(rid) is True
    assert block_store.delete_import_run(rid) is False, "重复删除应返回 False"


def test_conftest_cleans_runs_after_tests():
    """**防"测试污染真实库"**。

    conftest 里必须有 autouse 的批次清理，否则每跑一次测试就在真实库里
    留一堆 `xxx-test-doc`，`--runs` 的告警会因此失去意义。
    """
    import inspect

    from tests import conftest

    src = inspect.getsource(conftest)
    assert "clean_import_runs" in src, "conftest 没有清理批次"
    assert "autouse=True" in src
    assert "delete_import_run" in src, "清理没有真的删除"
