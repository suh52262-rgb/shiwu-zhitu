"""全量导入驱动：分批跑，每批报进度与实测成本。

**为什么要分批而不是一次跑完 3433 页**：
  1. 中途失败不至于全部重来（已落库的批次不用重跑）
  2. 能尽早看到"单页很美好、整本出问题"的故障
  3. 实测成本会随内容变化，分批才能给出可信的外推

**注意 `replace_doc=False`**：CLI 的单次导入默认会先删该文档的旧块，
分批跑时必须关掉，否则第二批会抹掉第一批。整本结束时才清一次。
"""
import argparse
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pymupdf  # noqa: E402

from app.rag import block_store, pdf_import  # noqa: E402
from app.services import vision  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="全量/分批导入")
    ap.add_argument("pdf")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0, help="0 = 到末页")
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--first-batch-only", action="store_true",
                    help="只跑第一批（用于先验证成本与质量）")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    # 分批驱动同样要校验：它也用 doc_id 拼输出目录
    from app.rag.import_guard import ImportGuardError, import_lock, validate_doc_id

    try:
        validate_doc_id(args.doc_id)
    except ImportGuardError as exc:
        print(f"  ✗ {exc}")
        return 2

    # 在函数开头导入：后面的"模型"那行就要用，不能等到锁里才导
    from app.services import vision

    with pymupdf.open(str(pdf)) as doc:
        total_pages = doc.page_count
    end = args.end or total_pages
    end = min(end, total_pages)

    print(f"  文档：{pdf.name}")
    print(f"  范围：第 {args.start}~{end} 页，每批 {args.batch} 页 "
          f"（共 {total_pages} 页）")
    print(f"  模型：{'(跳过 VL)' if False else ' / '.join(vision.vision_models())}")
    print()

    from app.rag.import_guard import file_sha256

    try:
        # **先取锁、再建批次**——与单文档导入同一原则：
        # 锁失败时不该留下一个永不收尾的 index_pending 批次。
        with import_lock(args.doc_id):
            from app.agent.prompt_registry import prompt_version_fields

            run_id = block_store.start_import_run(
                args.doc_id, source_sha256=file_sha256(pdf), source_name=pdf.name,
                mode="pages", pages_spec=f"{args.start}-{end}",
                model=",".join(vision.vision_models()),
                prompt_versions=prompt_version_fields(),
            )
            print(f"  导入批次 {run_id[:8]}…（状态 index_pending）")
            return _run_batches(args, pdf, end, total_pages, run_id=run_id)
    except ImportGuardError as exc:
        print(f"  ✗ {exc}")
        return 2


def _run_batches(args, pdf, end, total_pages, *, run_id: str) -> int:
    import time as _t
    grand_blocks: list[dict] = []
    #: 真正进了索引的块数。**不能用 len(grand_blocks) 代替**——
    #: 其中还有 `indexable=False` 被跳过的（已隔离的坏公式），
    #: 用块数当索引数会虚报（审计发现的第三处统计不准）
    grand_indexed = 0
    skipped_from_failed_pages = 0
    grand_tokens = 0
    failures: list[tuple[int, str]] = []
    t_all = time.perf_counter()

    page = args.start
    batch_no = 0
    # **实际处理到了第几页**。判"是否覆盖全书"必须用它，不能用外层的 `end`——
    # `--first-batch-only` 是在循环里 `break` 的，`end` 仍是完整范围，
    # 于是"只导了首批"会被判成覆盖全书并标成 ready（审计发现）。
    processed_through = args.start - 1
    while page <= end:
        batch_no += 1
        stop = min(page + args.batch - 1, end)
        pages = list(range(page, stop + 1))
        t0 = time.perf_counter()
        print(f"  ── 第 {batch_no} 批：第 {page}~{stop} 页 "
              f"（{len(pages)} 页）" + "─" * 20, flush=True)

        def progress(pno, total, pr):
            if pno % 25 == 0 or pno == total:
                print(f"      {pno}/{total} 页", flush=True)

        result = pdf_import.import_pdf(
            pdf, args.doc_id, pages=pages, use_vl=True,
            out_root=ROOT / "data" / "documents" / args.doc_id,
            progress=progress,
        )
        elapsed = time.perf_counter() - t0
        tokens = sum((pr.vl_meta or {}).get("prompt_tokens") or 0
                     for pr in result["page_results"])
        grand_tokens += tokens
        # **失败页的块一律不发布**（与单文档导入同一语义）。
        # 只挡"整文档替换"不够：`replace-pages` 仍会把失败页产出的**部分块**
        # 写进去并替换该页旧块——VL 解析失败时正文在、公式与插图没了，
        # 这一页的旧公式会被静默删掉（审计发现）。
        publishable = [b for pr in result["page_results"] if not pr.failed
                       for b in pr.blocks]
        skipped_from_failed_pages += len(result["blocks"]) - len(publishable)
        grand_blocks.extend(publishable)

        # **只认 pr.failed，不能遍历 notes。**
        # notes 里还有"图注锚点定位失败已回落模型"这类**不影响可用性**的提示，
        # 把它们当成失败会误报 partial 并非零退出（审计发现）。
        for pr in result["page_results"]:
            if pr.failed:
                failures.append((pr.page, "；".join(pr.notes) or "处理失败"))

        # 替换范围 = 本批覆盖的页。**不再用"是不是第一批"决定清不清库**——
        # 那个逻辑在第二批换进程跑时就会误判，而且第一批清库
        # 对"从中间页恢复"的场景是灾难。逐批按页替换天然幂等。
        n_pg = block_store.persist_blocks(publishable, doc_id=args.doc_id,
                                          replace=block_store.REPLACE_PAGES)
        n_q = 0
        if not args.no_index:
            n_q = block_store.index_blocks(publishable,
                                           replace=block_store.REPLACE_PAGES)
            grand_indexed += n_q

        # 按**实际发布**的块报告：`result["blocks"]` 含失败页的块，
        # 而那些已经被剔除了——按它报告会虚报入库量（审计发现）。
        by_type: dict[str, int] = {}
        for b in publishable:
            by_type[b["type"]] = by_type.get(b["type"], 0) + 1
        rate = elapsed / max(1, len(pages))
        skipped = len(result["blocks"]) - len(publishable)
        print(f"      ✓ {n_pg} 块入 PG / {n_q} 块入索引"
              + (f"（{skipped} 块来自失败页，未发布）" if skipped else "")
              + f"；{elapsed:.0f}s（{rate:.1f}s/页）；输入 {tokens} tok")
        print(f"        分类型：" + "、".join(f"{k}×{v}" for k, v in sorted(by_type.items())))
        print(f"        累计：{len(grand_blocks)} 块，{grand_tokens} tok，"
              f"估计 ${vision.estimate_cost(grand_tokens, 0):.3f}", flush=True)

        processed_through = max(processed_through, stop)

        if args.first_batch_only:
            print("\n  （--first-batch-only，到此为止）")
            break
        page = stop + 1

    wall = time.perf_counter() - t_all
    print("\n" + "=" * 74)
    print(f"  完成：{len(grand_blocks)} 块，耗时 {wall/60:.1f} 分钟")
    print(f"  输入 token：{grand_tokens}（约 ${vision.estimate_cost(grand_tokens, 0):.2f}）")
    if failures:
        print(f"  ⚠ {len(failures)} 页有注意事项：")
        for pno, note in failures[:10]:
            print(f"      第 {pno} 页：{note}")
    if not args.first_batch_only:
        docs = [d for d in block_store.list_documents() if d["doc_id"] == args.doc_id]
        if docs:
            d = docs[0]
            print(f"  库中：{d['pages']} 页 / {d['blocks']} 块  "
                  + "、".join(f"{k}×{v}" for k, v in sorted(d["by_type"].items())))

    # **收尾批次 + 失败页非零退出**（此前分批驱动两者都没做：
    # 状态会一直停在 index_pending，检索端永久排除该文档；
    # 有失败页也返回 0，上层以为成功了）
    # **这次导入是否覆盖了整份 PDF。**
    # 此前 `pages_total` 用的是本批页数，于是"只导了首批"的文档会显示
    # 1/1 并被标成 ready——**未完成的导入进入了检索**（审计发现）。
    # 现在用 PDF 总页数，且只要没导完就只能是 partial。
    covered = (args.start <= 1 and processed_through >= total_pages)
    if covered:
        status = (block_store.INDEX_PENDING if args.no_index
                  else (block_store.RUN_PARTIAL if failures else block_store.INDEX_READY))
    else:
        status = block_store.RUN_PARTIAL
        print(f"  ⚠ 本次只覆盖第 {args.start}~{processed_through} 页"
              f"（全书 {total_pages} 页），状态标为 partial —— "
              "**文档不会进入检索**，直到整本导完，或用 --reindex 显式重建索引")

    if skipped_from_failed_pages:
        print(f"  （{skipped_from_failed_pages} 块来自失败页，未发布；"
              "这些页的旧内容原样保留）")

    block_store.finish_import_run(
        run_id,
        status=status,
        pages_total=total_pages,
        # 按**实际处理**的页数算，不按外层范围——否则 --first-batch-only
        # 会虚报 pages_ok（审计发现）
        pages_ok=max(0, (processed_through - args.start + 1) - len(failures)),
        failed_pages=sorted({p for p, _ in failures}),
        blocks=len(grand_blocks),
        # `--no-index` 时一块都没入索引，记 0；
        # 否则记**真正入索引**的块数（grand_indexed 已扣掉不可索引的块）
        indexed=grand_indexed,
        prompt_tokens=grand_tokens,
    )
    if failures:
        print(f"  ✗ {len(failures)} 页有失败记录，状态标为 partial，返回非零")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
