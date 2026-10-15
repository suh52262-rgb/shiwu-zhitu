"""分批导入的覆盖判定与统计准确性（第三轮审计）。

这组问题的共同点：**用"计划的范围"代替"实际做到的事"**。
  · `covered` 比较外层的 `end`，而循环可能因 `--first-batch-only` 提前结束
  · `indexed` 记块数，而其中还有被跳过不索引的

两者都会让批次记录比事实乐观——而批次记录正是用来判断"能不能进检索"的。
"""
import inspect

from app.rag import block_store


def _src() -> str:
    from scripts import import_pdf_batch as batch

    return inspect.getsource(batch)


# ------------------------------------------------------ 覆盖判定

def test_covered_uses_actual_processed_page_not_planned_end():
    """**必须用"实际处理到第几页"判定覆盖，不能用外层的 `end`。**

    `--first-batch-only` 是在循环里 `break` 的，`end` 仍是完整范围，
    于是"只导了首批"会被判成覆盖全书并标成 `ready`——
    **未完成的文档进入检索**（审计发现）。
    """
    src = _src()
    assert "processed_through" in src, "没有记录实际处理进度"
    assert "processed_through >= total_pages" in src, "covered 没改用实际进度"
    assert "end >= total_pages" not in src, "covered 仍在比较外层的 end"


def test_processed_through_updated_before_break():
    """进度必须在 `break` **之前**更新，否则提前结束时它还是旧值。"""
    src = _src()
    i_update = src.index("processed_through = max(processed_through, stop)")
    i_break = src.index("if args.first_batch_only:")
    assert i_update < i_break, "进度更新在 break 之后，提前结束时会漏记"


def test_processed_through_initialised_before_loop():
    """初值必须是 `start - 1`（"一页都没处理"），不能是 0——
    否则 `--start 100` 时会被当成"处理了 0~..."。"""
    src = _src()
    assert "processed_through = args.start - 1" in src


def test_pages_ok_uses_processed_range():
    """`pages_ok` 也要按实际处理范围算，否则 `--first-batch-only` 会虚报。"""
    src = _src()
    assert "(processed_through - args.start + 1) - len(failures)" in src, \
        "pages_ok 仍按外层范围计算"


# ------------------------------------------------------ 统计准确性

def test_indexed_counts_actually_indexed_blocks():
    """`indexed` 要记**真正进了索引**的块数。

    `len(grand_blocks)` 里还有 `indexable=False` 被 `index_blocks` 跳过的
    （已隔离的坏公式），用块数当索引数会虚报。
    """
    src = _src()
    assert "grand_indexed" in src, "没有独立的索引计数"
    assert "grand_indexed += n_q" in src, "没有累加实际入索引的块数"
    assert "indexed=grand_indexed" in src, "indexed 没用实际值"
    assert "indexed=(0 if args.no_index else len(grand_blocks))" not in src


def test_no_index_records_zero_indexed():
    """`--no-index` 时一块都没入索引，`grand_indexed` 保持 0。

    用 `len(grand_blocks)` 的话会记成"全都入索引了"——与实际相反。
    """
    src = _src()
    # 只有 not args.no_index 分支里才累加
    i = src.index("grand_indexed += n_q")
    head = src[:i]
    assert "if not args.no_index:" in head, "累加不在 no_index 判断之内"


def test_per_batch_output_reports_published_blocks():
    """每批的输出要按**实际发布**的块报告。

    `result["blocks"]` 含失败页的块，而那些已被剔除——
    按它报告会虚报入库量（审计发现）。
    """
    src = _src()
    assert 'for b in result["blocks"]:\n            by_type' not in src, \
        "分类型统计仍用未过滤的块"
    assert "for b in publishable:" in src, "分类型统计没用 publishable"
    assert "{n_pg} 块入 PG" in src, "入库数仍用未过滤的块数"
    assert "{len(result['blocks'])} 块入 PG" not in src


def test_failed_page_blocks_are_reported_as_skipped():
    """被剔除的块数要**说出来**，不能静默少报。"""
    src = _src()
    assert "skipped" in src
    assert "来自失败页，未发布" in src


# ------------------------------------------------------ 端到端语义

def test_partial_status_is_not_ready():
    """`partial` 与 `ready` 必须区分——这是"能不能进检索"的开关。"""
    assert block_store.RUN_PARTIAL != block_store.INDEX_READY
    assert block_store.RUN_PARTIAL != block_store.INDEX_PENDING
