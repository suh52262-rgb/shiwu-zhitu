"""导入安全：替换范围、对账、防数据丢失。

## 这一组测的是「不会丢数据」

此前 `persist_blocks` 默认**整文档替换**，于是
`import_pdf.py <pdf> --pages 47` 会把该文档其余 491 页的块删光——
而 Qdrant 侧只 upsert 不清旧点，两个存储当场分叉。

现在默认是 `"pages"`：**只替换本次写入覆盖的页**。
这一条规则同时挡掉局部导入清库、分批互相覆盖、孤儿向量三类事故。
"""
from unittest.mock import MagicMock

import pytest

from app.rag import block_store
from app.rag.block_store import (
    REPLACE_DOCUMENT,
    REPLACE_NONE,
    REPLACE_PAGES,
    _page_gaps,
    _pages_of,
)


def _blocks(pages: list[int], *, doc_id: str = "d") -> list[dict]:
    out = []
    for p in pages:
        for i in range(2):
            out.append({
                "id": f"{doc_id}-{p}-{i}", "doc_id": doc_id, "page": p,
                "block_index": i, "type": "text", "text": f"p{p} 的第 {i} 块内容",
                "embed_text": f"p{p} 的第 {i} 块内容",
            })
    return out


def test_pages_of_collects_unique_sorted_pages():
    assert _pages_of(_blocks([5, 3, 5, 1])) == [1, 3, 5]


def test_page_gaps_reports_missing_ranges():
    assert _page_gaps([1, 2, 3]) == []
    assert _page_gaps([1, 4, 5, 9]) == ["2-3", "6-8"]
    assert _page_gaps([]) == []


# ---------------------------------------------------------------- 默认语义

def test_default_replace_scope_is_pages_not_document():
    """**这条断言是防回退的护栏。**

    默认值一旦被改回"整文档替换"，局部导入就会静默删掉其他页——
    这是本项目踩过的真实数据丢失 bug。
    """
    import inspect

    sig = inspect.signature(block_store.persist_blocks)
    assert sig.parameters["replace"].default == REPLACE_PAGES
    sig_index = inspect.signature(block_store.index_blocks)
    assert sig_index.parameters["replace"].default == REPLACE_PAGES


def test_partial_import_only_deletes_its_own_pages(monkeypatch):
    """`--pages 47` 只能删第 47 页，其余页必须原样保留。"""
    executed: list = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def begin(self):
            return self

        def execute(self, stmt):
            executed.append(str(stmt))

        def merge(self, obj):
            return obj

    monkeypatch.setattr(block_store, "Session", lambda engine: FakeSession())
    monkeypatch.setattr(block_store, "get_engine", lambda: None)

    block_store.persist_blocks(_blocks([47]), doc_id="d", replace=REPLACE_PAGES)
    deletes = [s for s in executed if s.strip().upper().startswith("DELETE")]
    assert len(deletes) == 1, deletes
    # 删除语句必须带上页码条件，而不是只按 doc_id
    assert "page" in deletes[0].lower(), deletes[0]


def test_replace_document_deletes_without_page_filter(monkeypatch):
    """全量重导才允许整文档删——那时没有 page 条件是对的。"""
    executed: list = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def begin(self):
            return self

        def execute(self, stmt):
            executed.append(str(stmt))

        def merge(self, obj):
            return obj

    monkeypatch.setattr(block_store, "Session", lambda engine: FakeSession())
    monkeypatch.setattr(block_store, "get_engine", lambda: None)

    block_store.persist_blocks(_blocks([47]), doc_id="d", replace=REPLACE_DOCUMENT)
    deletes = [s for s in executed if s.strip().upper().startswith("DELETE")]
    assert len(deletes) == 1
    assert "page" not in deletes[0].lower()


def test_replace_none_never_deletes(monkeypatch):
    """`append` 模式一条都不删。"""
    executed: list = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def begin(self):
            return self

        def execute(self, stmt):
            executed.append(str(stmt))

        def merge(self, obj):
            return obj

    monkeypatch.setattr(block_store, "Session", lambda engine: FakeSession())
    monkeypatch.setattr(block_store, "get_engine", lambda: None)

    block_store.persist_blocks(_blocks([47]), doc_id="d", replace=REPLACE_NONE)
    assert not [s for s in executed if s.strip().upper().startswith("DELETE")]


def test_empty_blocks_is_a_noop(monkeypatch):
    """空输入不该触碰数据库——否则会误删。"""
    called = []
    monkeypatch.setattr(block_store, "get_engine", lambda: called.append(1))
    assert block_store.persist_blocks([], doc_id="d") == 0
    assert not called


# ---------------------------------------------------------------- 索引侧删除

def test_index_deletes_before_upsert_in_same_scope(monkeypatch):
    """**孤儿向量的根因修复**：upsert 之前必须按同样的范围先删。

    VL 输出不确定 → 同一页两次导入的公式条数会变 → 块数变少就留下孤儿点。
    只 upsert 不清旧点，重导同一份 PDF 仍会再产生孤儿。
    """
    deleted: list = []
    monkeypatch.setattr(block_store, "delete_points",
                        lambda doc_id, pages=None: deleted.append((doc_id, pages)))
    monkeypatch.setattr("app.rag.ingest._ensure_collection", lambda *a, **kw: None)

    fake_embedder = MagicMock()
    fake_embedder.embed.side_effect = lambda texts: [[0.0] * 4 for _ in texts]
    monkeypatch.setattr("app.rag.embedding.get_embedding_client",
                        lambda: fake_embedder)
    client = MagicMock()
    client.collection_exists.return_value = True
    monkeypatch.setattr(block_store, "get_qdrant", lambda: client)

    block_store.index_blocks(_blocks([47, 48]), replace=REPLACE_PAGES)
    assert deleted, "没有在写入前删除旧点（孤儿向量就是这么来的）"
    _, pages = deleted[0]
    assert set(pages) == {47, 48}, f"删除范围与写入范围不一致：{pages}"


def test_index_skips_deletion_in_append_mode(monkeypatch):
    deleted: list = []
    monkeypatch.setattr(block_store, "delete_points",
                        lambda doc_id, pages=None: deleted.append((doc_id, pages)))
    monkeypatch.setattr("app.rag.ingest._ensure_collection", lambda *a, **kw: None)
    fake_embedder = MagicMock()
    fake_embedder.embed.side_effect = lambda texts: [[0.0] * 4 for _ in texts]
    monkeypatch.setattr("app.rag.embedding.get_embedding_client",
                        lambda: fake_embedder)
    client = MagicMock()
    client.collection_exists.return_value = True
    monkeypatch.setattr(block_store, "get_qdrant", lambda: client)

    block_store.index_blocks(_blocks([47]), replace=REPLACE_NONE)
    assert not deleted


# ---------------------------------------------------------------- 对账

def test_reconcile_detects_orphans(monkeypatch):
    """对账要能报出"索引有、PG 无"的孤儿，并带上页码供判断。"""
    monkeypatch.setattr(block_store, "load_blocks",
                        lambda doc: _blocks([1], doc_id="d"))
    monkeypatch.setattr(block_store, "_scroll_points", lambda doc: [
        {"id": "d-1-0", "payload": {"page": 1, "type": "text", "text": "对齐的"}},
        {"id": "d-1-1", "payload": {"page": 1, "type": "text", "text": "对齐的"}},
        {"id": "孤儿子", "payload": {"page": 38, "type": "formula", "text": "$$10^1$$"}},
    ])

    rep = block_store.reconcile("d")
    assert rep["qdrant_only"] == ["孤儿子"]
    assert rep["pg_only"] == []
    assert rep["orphan_detail"][0]["page"] == 38
    assert rep["orphan_detail"][0]["type"] == "formula"


def test_reconcile_reports_missing_vectors(monkeypatch):
    monkeypatch.setattr(block_store, "load_blocks",
                        lambda doc: _blocks([1], doc_id="d"))
    monkeypatch.setattr(block_store, "_scroll_points", lambda doc: [
        {"id": "d-1-0", "payload": {"page": 1, "type": "text", "text": "a"}},
    ])
    rep = block_store.reconcile("d")
    assert rep["pg_only"] == ["d-1-1"]


def test_reconcile_figure_path_keeps_subdirectory(tmp_path, monkeypatch):
    """**图片存在性检查不能剥掉 `figures/` 子目录。**

    踩过：用 `Path(rel).name` 会丢掉目录，把 637 个真实存在的图片
    全报成缺失。figure_path 存的是相对**文档根**的路径。
    """
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "p1.png").write_bytes(b"x")
    blocks = _blocks([1], doc_id="d")
    blocks[0]["figure_path"] = "figures/p1.png"

    monkeypatch.setattr(block_store, "load_blocks", lambda doc: blocks)
    monkeypatch.setattr(block_store, "_scroll_points", lambda doc: [])
    rep = block_store.reconcile("d", figures_root=tmp_path)
    assert rep["missing_figures"] == [], rep["missing_figures"]


def test_reconcile_dry_run_does_not_modify(monkeypatch):
    """不加 `--apply` 一律只读——对账命令误改数据的代价太大。"""
    monkeypatch.setattr(block_store, "load_blocks", lambda doc: _blocks([1], doc_id="d"))
    monkeypatch.setattr(block_store, "_scroll_points", lambda doc: [])
    called = []
    monkeypatch.setattr(block_store, "get_qdrant", lambda: called.append(1))
    rep = block_store.reconcile("d")
    assert rep["applied"] is None
    assert not called


def test_page_coverage_reports_gaps(monkeypatch):
    """页覆盖断档要说清楚——"导入了 400 页"和"导入了 1~492 页"不是一回事。"""
    monkeypatch.setattr(block_store, "load_blocks",
                        lambda doc: _blocks([1, 2, 5], doc_id="d"))
    monkeypatch.setattr(block_store, "_scroll_points", lambda doc: [])
    rep = block_store.reconcile("d")
    assert rep["pages"] == {"count": 3, "first": 1, "last": 5, "gaps": ["3-4"]}


# ---------------------------------------------------------------- 列默认值

def test_boolean_column_has_server_default():
    """**防回退护栏**：新加的布尔列必须带 server default。

    根因回顾：`ALTER TABLE ... ADD COLUMN "indexable" BOOLEAN` 不带 DEFAULT 时，
    已存在的行全是 NULL；Python 侧 `bool(None)` 是 False，
    于是一个语义上默认 true 的列**静默变成"全部 false"**——
    22976 块全被标成不可索引，重建索引写了 0 块，
    同时按范围把 22976 个向量全删了。
    """
    from app.db.tables import DocumentBlock

    col = DocumentBlock.__table__.c.indexable
    assert col.server_default is not None, "布尔列缺 server default，迁移会给已有行留 NULL"


def test_migration_add_column_includes_default():
    """迁移语句本身要带 DEFAULT，否则上面的 server_default 白设。"""
    import inspect

    from app.db import migrate

    src = inspect.getsource(migrate.apply_schema_migrations)
    assert "server_default" in src, "迁移没读 server_default"
    assert "DEFAULT" in src, "ADD COLUMN 没带 DEFAULT 子句"
    assert "IS NULL" in src, "没有回填历史 NULL 行"
