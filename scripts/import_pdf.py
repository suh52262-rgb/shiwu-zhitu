"""导入一份 PDF 到知识库（PostgreSQL 事实 + Qdrant 索引）。

    python scripts/import_pdf.py <pdf> --doc-id my-textbook [--pages 1-20] [--no-vl]
    python scripts/import_pdf.py --list                  # 看已导入的文档
    python scripts/import_pdf.py --reindex <doc-id>      # 只重建索引，不碰 VL
    python scripts/import_pdf.py --delete <doc-id>

`--reindex` 是这套设计的关键收益：换 embedding 模型时只需重跑这一步，
**不必再花一千万 tokens 重调 VL**。
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
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.rag import block_store, pdf_import  # noqa: E402


def _parse_pages(spec: str) -> list[int] | None:
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="导入 PDF 到知识库")
    ap.add_argument("pdf", nargs="?", help="PDF 路径")
    ap.add_argument("--doc-id", help="文档标识（默认取文件名）")
    ap.add_argument("--doc-type", default="textbook",
                    choices=["textbook", "exam", "notes"])
    ap.add_argument("--pages", default="", help="页码，如 1-20,35")
    ap.add_argument("--no-vl", action="store_true",
                    help="跳过 VL（只入正文与图注，用于验证版面流程）")
    ap.add_argument("--no-index", action="store_true", help="只落 PG，不建向量索引")
    ap.add_argument("--list", action="store_true", help="列出已导入文档")
    ap.add_argument("--reindex", metavar="DOC_ID", help="只重建索引（不重新解析 PDF）")
    ap.add_argument("--delete", metavar="DOC_ID", help="删除文档及其索引")
    ap.add_argument("--reconcile", metavar="DOC_ID",
                    help="对账 PG 与 Qdrant（配 --apply 才真的修）")
    ap.add_argument("--apply", action="store_true", help="配合 --reconcile 执行修复")
    ap.add_argument("--runs", metavar="DOC_ID", nargs="?", const="",
                    help="列出导入批次（省略 DOC_ID 列全部）")
    ap.add_argument("--prune", action="store_true",
                    help="配合 --runs：把被取代的未完成批次标为 superseded（不删除）")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "replace-pages", "replace-document", "append"],
                    help="替换范围。auto：全量导入=整文档，--pages 局部导入=只替换这些页")
    args = ap.parse_args()

    if args.list:
        docs = block_store.list_documents()
        if not docs:
            print("  尚无已导入文档")
            return 0
        print(f"  {'doc_id':<28}{'类型':<10}{'页数':>6}{'块数':>7}   分类型")
        for d in docs:
            by = "、".join(f"{k}×{v}" for k, v in sorted(d["by_type"].items()))
            print(f"  {d['doc_id']:<28}{d['doc_type']:<10}{d['pages']:>6}"
                  f"{d['blocks']:>7}   {by}")
        return 0

    if args.runs is not None and args.prune:
        n = block_store.supersede_stale_runs(args.runs or None)
        print(f"  已把 {n} 个被取代的未完成批次标为 superseded"
              "（记录保留，`--runs` 仍可查看）")
        # 标记后继续往下打印列表，便于立刻确认结果

    if args.runs is not None:
        runs = block_store.list_import_runs(args.runs or None, limit=20)
        if not runs:
            print("  尚无导入批次记录")
            return 0
        print(f"  {'批次':<10}{'文档':<20}{'状态':<15}{'页':>8}{'块':>7}{'索引':>7}  开始")
        for r in runs:
            print(f"  {r['id'][:8]:<10}{r['doc_id'][:18]:<20}{r['status']:<15}"
                  f"{r['pages_ok']}/{r['pages_total']:<6}{r['blocks']:>7}"
                  f"{r['indexed']:>7}  {r['started_at'][:19]}")
            if r["failed_pages"]:
                print(f"      失败页：{r['failed_pages']}")
        # **只在文档的最新批次确实停住时才报可疑。**
        # 历史遗留的旧批次停在 index_pending 是正常的——文档状态取最近一次，
        # 只要最新批次是 ready，文档就没有被排除（此前把这些也报出来，
        # 属于误报，会让人去修一个不是问题的问题）。
        latest_by_doc: dict[str, dict] = {}
        for r in runs:                      # runs 按 started_at 倒序
            latest_by_doc.setdefault(r["doc_id"], r)
        stale = [r for r in latest_by_doc.values()
                 if r["status"] == block_store.INDEX_PENDING and r["blocks"] == 0]
        if stale:
            print(f"\n  ⚠ {len(stale)} 个批次停在 index_pending 且未产出任何块"
                  "（多半是中断的）：")
            for r in stale[:5]:
                print(f"      {r['id'][:8]}  {r['doc_id']}  {r['started_at'][:19]}")
            print("      处置：`--reindex <doc-id>` 或 `--reconcile <doc-id> --apply`")
            print("            会把状态置回 ready；若该文档本就可用，这步必要。")
        return 0

    if args.reconcile:
        if args.apply:
            # **修复动作开始前先置 pending**：修复过程中 Qdrant 处于
            # "删了一部分、还没补回"的状态，此时若状态仍是 ready，
            # 检索端会拿到不完整索引（审计发现）。
            block_store.set_index_status(args.reconcile, block_store.INDEX_PENDING)
        rep = block_store.reconcile(args.reconcile, apply=args.apply)
        print(f"  文档 {rep['doc_id']}")
        print(f"    PG 块数        {rep['pg_blocks']}")
        print(f"    Qdrant 点数    {rep['qdrant_points']}")
        print(f"    页覆盖         {rep['pages']['count']} 页"
              f"（{rep['pages']['first']}~{rep['pages']['last']}）"
              f"{'  断档 ' + str(rep['pages']['gaps']) if rep['pages']['gaps'] else ''}")
        print(f"    未索引（设计如此）{len(rep.get('unindexed_by_design', []))} 个"
              "  ← 已隔离的坏公式，原文保留在 PG")
        print(f"    PG 有索引无    {len(rep['pg_only'])} 个")
        for i in rep["pg_only"][:5]:
            print(f"        {i}")
        print(f"    索引有 PG 无   {len(rep['qdrant_only'])} 个（孤儿）")
        for o in rep.get("orphan_detail", [])[:8]:
            print(f"        p{o['page']} [{o['type']}] {o['text']!r}")
        print(f"    payload 不一致 {len(rep['payload_diff'])} 处")
        for d in rep["payload_diff"][:5]:
            print(f"        {d['id']} {d['field']}: qdrant={d['qdrant']!r} pg={d['pg']!r}")
        print(f"    图片缺失       {len(rep['missing_figures'])} 个")
        for m in rep["missing_figures"][:5]:
            print(f"        p{m['page']} {m['path']}")
        if args.apply:
            print(f"    已修复：{rep['applied']}")
            # **修完要重新判定，而不是"原本就无差异才置 ready"。**
            # 此前只在无差异时置 ready——于是"修好了差异"的文档反而
            # 停在 index_pending 被检索排除（审计发现）。
            after = block_store.reconcile(args.reconcile, apply=False)
            if not (after["pg_only"] or after["qdrant_only"] or after["payload_diff"]):
                block_store.set_index_status(args.reconcile, block_store.INDEX_READY)
                print("    复查无差异，文档索引状态已置为 ready")
            else:
                print(f"    复查仍有差异：PG缺失 {len(after['pg_only'])} / "
                      f"孤儿 {len(after['qdrant_only'])} / "
                      f"payload {len(after['payload_diff'])} —— 状态保持 index_pending")
                return 4
        else:
            print("    （--dry-run 语义：加 --apply 才真的修）")
        return 0

    if args.delete:
        stat = block_store.delete_document(args.delete)
        print(f"  已删除 {args.delete}：PG {stat['blocks']} 块，Qdrant 已清理")
        return 0

    if args.reindex:
        blocks = block_store.load_blocks(args.reindex)
        if not blocks:
            print(f"  ✗ PostgreSQL 里没有 {args.reindex} 的内容块")
            return 1
        print(f"  从 PostgreSQL 读回 {len(blocks)} 块，开始重建索引……")
        # **重建前先标 pending**：重建过程中 Qdrant 处于"删了一部分、
        # 还没写回"的状态，此时若状态仍是 ready，检索端会拿到部分索引。
        block_store.set_index_status(args.reindex, block_store.INDEX_PENDING)
        t0 = time.perf_counter()
        n = block_store.index_blocks(blocks, replace=block_store.REPLACE_DOCUMENT)
        # **恢复路径必须把状态置回 ready**：否则一次崩溃让文档停在
        # index_pending，检索端会永久排除它——而它其实已经修好了。
        block_store.set_index_status(args.reindex, block_store.INDEX_READY)
        print(f"  ✓ 重建索引 {n} 块，耗时 {time.perf_counter() - t0:.1f}s")
        print(f"  文档索引状态已置为 ready")
        return 0

    if not args.pdf:
        ap.print_help()
        return 1
    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"  ✗ 找不到 {pdf}")
        return 1
    from app.rag.import_guard import (
        ImportGuardError, file_sha256, import_lock, validate_doc_id,
    )

    doc_id = args.doc_id or pdf.stem
    try:
        validate_doc_id(doc_id)
    except ImportGuardError as exc:
        print(f"  ✗ {exc}")
        return 2
    page_list = _parse_pages(args.pages)

    # 记录源文件摘要：区分"同一文件重试"与"同名但内容不同的文件"。
    # 只看文件名的话，后者会被当成前者，用旧结果糊弄过去。
    digest = file_sha256(pdf)
    print(f"  源文件 SHA-256：{digest[:16]}…")

    print(f"  导入 {pdf.name} → doc_id={doc_id}  类型={args.doc_type}"
          f"{'  (跳过 VL)' if args.no_vl else ''}")
    t0 = time.perf_counter()
    try:
        # **先取锁、再建批次。** 反过来的话，锁被占用时异常分支直接返回，
        # 那个刚建出来的 index_pending 批次永远不会被收尾——
        # **会让原本可用的文档从检索里消失**（审计发现；数据库里那两条
        # 0 页的 index_pending 记录就是证据）。
        with import_lock(doc_id):
            from app.agent.prompt_registry import prompt_version_fields
            from app.services import vision

            run_id = block_store.start_import_run(
                doc_id, source_sha256=digest, source_name=pdf.name,
                doc_type=args.doc_type, mode=args.mode, pages_spec=args.pages,
                model=",".join(vision.vision_models()),
                prompt_versions=prompt_version_fields(),
            )
            print(f"  导入批次 {run_id[:8]}…（状态 index_pending，检索端会先排除本文件）")
            return _run_import(args, pdf, doc_id, page_list,
                               doc_type=args.doc_type, run_id=run_id)
    except ImportGuardError as exc:
        print(f"  ✗ {exc}")
        return 2


def _run_import(args, pdf, doc_id, page_list, *, doc_type: str, run_id: str) -> int:
    import time as _time
    t0 = _time.perf_counter()

    def progress(page_no, total, pr):
        if page_no % 5 == 0 or page_no == total:
            print(f"    第 {page_no}/{total} 页：{len(pr.blocks)} 块"
                  f"{'  ⚠ ' + pr.notes[0] if pr.notes else ''}", flush=True)

    result = pdf_import.import_pdf(
        pdf, doc_id, doc_type=args.doc_type, pages=page_list,
        use_vl=not args.no_vl, out_root=ROOT / "data" / "documents" / doc_id,
        progress=progress,
    )
    elapsed = time.perf_counter() - t0

    by_type: dict[str, int] = {}
    for b in result["blocks"]:
        by_type[b["type"]] = by_type.get(b["type"], 0) + 1
    tok = sum((pr.vl_meta or {}).get("prompt_tokens") or 0
              for pr in result["page_results"])
    print(f"\n  提取完成：{result['pages']} 页，{len(result['blocks'])} 块，"
          f"耗时 {elapsed:.1f}s，输入 token {tok}")
    print("    分类型：" + "、".join(f"{k}×{v}" for k, v in sorted(by_type.items())))

    warnings = [(pr.page, pr.notes) for pr in result["page_results"] if pr.notes]
    if warnings:
        print(f"  ⚠ {len(warnings)} 页有注意事项：")
        for page_no, notes in warnings[:5]:
            print(f"      第 {page_no} 页：{'；'.join(notes)}")

    # 替换范围：默认按"本次覆盖的页"，全量导入时等价于整文档替换。
    # **绝不能默认整文档**——`--pages 47` 会删掉其余 491 页（此前的真实 bug）。
    if args.mode == "append":
        replace = block_store.REPLACE_NONE
    elif args.mode == "replace-document":
        replace = block_store.REPLACE_DOCUMENT
    elif args.mode == "replace-pages":
        replace = block_store.REPLACE_PAGES
    else:                     # auto
        replace = block_store.REPLACE_PAGES if page_list else block_store.REPLACE_DOCUMENT
    # ---- 失败页的处置：**绝不拿"部分成功"去做整文档替换** ----
    failed_pages = [pr.page for pr in result["page_results"] if pr.failed]
    if failed_pages and replace == block_store.REPLACE_DOCUMENT:
        print(f"\n  ✗ {len(failed_pages)} 页处理失败，**拒绝执行整文档替换**。")
        print(f"     失败页：{failed_pages[:12]}{' …' if len(failed_pages) > 12 else ''}")
        print("     整文档替换会先删光该文档的全部旧块，再插回成功的那些——")
        print("     失败页的旧内容会就此消失，而脚本仍报成功。")
        print("     改法：修好失败页后重跑，或用")
        print(f"       --mode replace-pages --pages <成功页> 只替换成功页")
        block_store.finish_import_run(
            run_id, status=block_store.RUN_FAILED,
            pages_total=result["pages"],
            pages_ok=result["pages"] - len(failed_pages),
            failed_pages=failed_pages,
            blocks=len(result["blocks"]),
            error="整文档替换被拒绝：存在失败页")
        return 3

    # **失败页的块一律不发布**（任何模式都一样）。
    #
    # 只挡住"整文档替换"不够：`replace-pages` 仍会把失败页产出的**部分块**
    # 写进去并替换该页旧块——VL 解析失败时正文在、公式与插图没了，
    # 于是这一页的旧公式被静默删掉（审计发现）。
    #
    # 语义统一成一句话：**失败页不发布**。首次导入时该页就是缺失的
    # （且被明确报为失败），重导时旧内容原样保留。两种结果都比
    # "悄悄用残缺内容覆盖完整内容"好。
    publishable = [b for pr in result["page_results"] if not pr.failed
                   for b in pr.blocks]
    skipped_blocks = len(result["blocks"]) - len(publishable)
    if skipped_blocks:
        print(f"  （{skipped_blocks} 块来自失败页，不发布；这些页的旧内容原样保留）")

    print(f"\n  写入 PostgreSQL（事实来源）……替换范围={replace}")
    n = block_store.persist_blocks(publishable, doc_id=doc_id,
                                   doc_type=args.doc_type, replace=replace)
    print(f"  ✓ {n} 块")

    indexed = 0
    if not args.no_index:
        print("  写入 Qdrant（索引）……")
        t1 = time.perf_counter()
        indexed = block_store.index_blocks(publishable, replace=replace)
        print(f"  ✓ {indexed} 块，耗时 {time.perf_counter() - t1:.1f}s")
    else:
        print("  （--no-index）未建索引，该文档保持 index_pending，")
        print("   检索端会排除它——直到你跑 --reindex 或 --reconcile --apply。")

    block_store.finish_import_run(
        run_id,
        status=(block_store.INDEX_PENDING if args.no_index
                else (block_store.RUN_PARTIAL if failed_pages else block_store.INDEX_READY)),
        pages_total=result["pages"],
        pages_ok=result["pages"] - len(failed_pages),
        failed_pages=failed_pages, blocks=n, indexed=indexed,
        prompt_tokens=tok,
    )
    print(f"\n  完成。插图目录 data/documents/{doc_id}/figures/")
    if failed_pages:
        # **有失败页就不能返回 0**：上层脚本/CI 靠退出码判断成败，
        # 返回 0 会让"部分导入"被当成"导入成功"。
        print(f"  ✗ 但有 {len(failed_pages)} 页失败：{failed_pages[:12]}"
              f"{' …' if len(failed_pages) > 12 else ''}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
