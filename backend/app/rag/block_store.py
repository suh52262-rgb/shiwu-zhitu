"""内容块落库：PostgreSQL（事实）+ Qdrant（索引）。

**两者可以分开重跑**，这是关键：
  · `persist_blocks()` 便宜 —— 只是写结构化数据
  · `index_blocks()` 也不贵 —— 实测整本 22976 块约 2.6 分钟
  · 真正贵的是 `pdf_import.import_pdf()`（要调 VL，约 60 万 tokens/整本）

所以换 embedding 模型时只重跑 `index_blocks()`，**不必再碰 VL**。

## 替换范围 = 写入范围（一条规则，解决三个问题）

`replace` 的默认值是 `"pages"`：**只替换本次 blocks 覆盖到的那些页**。
这一条规则同时挡掉三种事故：

1. **`--pages 47` 不该删掉其余 491 页**。此前 `replace_doc` 默认 `True`，
   局部导入会把整个文档的 PG 行删光——**数据丢失级 bug**。
2. **分批导入不该互相覆盖**。以前靠"是不是第一批"决定清不清库，
   逻辑含糊且脆弱（第二批在别的进程跑就会误判）。
3. **孤儿向量**。Qdrant 只 upsert 不清旧点，页内块数变少就留下孤儿。
   索引侧按**同样的范围**删除，两个存储才对齐。

**为什么会有块消失**：VL 输出是**不确定**的，同一页两次导入的公式条数会变。
`block_id = uuid5(doc_id, page, block_index)`，块数从 10 变 8，
索引 8/9 的点就成了孤儿。**这是当前 Qdrant 里那 5 个孤儿的真实成因**
（只归因于"只 upsert、不清旧点"不够——即使清了整个文档，重导仍会再产生孤儿）。
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.postgres import get_engine
from app.db.qdrant import get_qdrant
from app.db.tables import DocumentBlock

logger = logging.getLogger(__name__)

#: `replace` 的三个取值
REPLACE_PAGES = "pages"          # 默认：只替换本次写入覆盖的页
REPLACE_DOCUMENT = "document"    # 整文档替换（全量重导才用）
REPLACE_NONE = "none"            # 只增不改


def _pages_of(blocks: list[dict]) -> list[int]:
    return sorted({int(b["page"]) for b in blocks if b.get("page") is not None})


def persist_blocks(blocks: list[dict], *, doc_id: str, doc_type: str = "textbook",
                   replace: str = REPLACE_PAGES) -> int:
    """写入 PostgreSQL。返回写入条数。

    `replace`（**默认 `"pages"`，注意不是整文档**）：
      · `"pages"`    —— 只删本次覆盖到的页的旧块。局部导入安全。
      · `"document"` —— 删该文档的全部旧块（只有全量重导才该用）。
      · `"none"`     —— 只增不改。

    ⚠️ **不要把默认值改回"整文档替换"**。此前正是它让
    `import_pdf.py --pages 47` 删掉了其余 491 页（见模块 docstring）。
    """
    if not blocks:
        return 0
    pages = _pages_of(blocks)
    with Session(get_engine()) as session, session.begin():
        if replace == REPLACE_DOCUMENT:
            session.execute(delete(DocumentBlock).where(DocumentBlock.doc_id == doc_id))
        elif replace == REPLACE_PAGES and pages:
            session.execute(
                delete(DocumentBlock).where(
                    DocumentBlock.doc_id == doc_id,
                    DocumentBlock.page.in_(pages),
                )
            )
        for b in blocks:
            session.merge(DocumentBlock(
                id=b["id"],
                doc_id=doc_id,
                doc_type=doc_type,
                page=b["page"],
                block_index=b["block_index"],
                type=b.get("type", "text"),
                text=b.get("text", ""),
                embed_text=b.get("embed_text", "") or b.get("text", ""),
                formula_number=b.get("formula_number", ""),
                bbox=b.get("bbox") or [],
                figure_path=b.get("figure_path", ""),
                figure_labels=b.get("figure_labels") or [],
                section=b.get("section", ""),
                source_kind=b.get("source_kind", "text_layer"),
                is_generated=bool(b.get("is_generated", False)),
                # 同理：只有显式 False 才不可索引，缺失/None 都当可索引
                indexable=b.get("indexable") is not False,
                note=b.get("note", ""),
                model=b.get("model", ""),
            ))
    logger.info("落库 %s：%d 块，替换范围=%s%s", doc_id, len(blocks), replace,
                f"（{len(pages)} 页）" if pages else "")
    return len(blocks)

    return len(blocks)


def load_blocks(doc_id: str) -> list[dict]:
    """从 PostgreSQL 读回内容块（重建索引时用，不必重新解析 PDF）。"""
    with Session(get_engine()) as session:
        rows = session.execute(
            select(DocumentBlock).where(DocumentBlock.doc_id == doc_id)
            .order_by(DocumentBlock.page, DocumentBlock.block_index)
        ).scalars().all()
    return [
        {
            "id": r.id, "doc_id": r.doc_id, "page": r.page,
            "block_index": r.block_index, "type": r.type,
            "text": r.text, "embed_text": r.embed_text,
            "formula_number": r.formula_number, "bbox": r.bbox,
            "figure_path": r.figure_path, "figure_labels": r.figure_labels,
            "section": r.section, "source_kind": r.source_kind,
            # NULL 视为 True：该列语义默认是"可索引"，
            # 不要用 bool()——bool(None) 是 False，会静默反过来。
            "is_generated": r.is_generated, "indexable": r.indexable is not False,
            "note": r.note, "model": r.model,
        }
        for r in rows
    ]


def load_page_blocks(doc_id: str, pages: list[int]) -> dict[tuple[str, int], list[dict]]:
    """一次取回若干页的全部内容块，按 (doc_id, page) 分组。

    供检索后的**上下文扩展**用：把命中块周围的同页块一并取回。
    用一次查询取整页，而不是每个命中各查一次——命中通常集中在少数几页。
    """
    if not pages:
        return {}
    with Session(get_engine()) as session:
        rows = session.execute(
            select(DocumentBlock)
            .where(DocumentBlock.doc_id == doc_id, DocumentBlock.page.in_(pages))
            .order_by(DocumentBlock.page, DocumentBlock.block_index)
        ).scalars().all()
    grouped: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r.doc_id, r.page), []).append({
            "block_index": r.block_index, "type": r.type, "text": r.text,
            # ⚠️ **必须带上过滤元数据**（2026-10-10 RAG 核查第 1 项）：
            #    原实现只返回 `block_index/type/text` ⇒ 扩展器**想过滤也没有依据**，
            #    于是隔离块（`indexable=False`，如"扫描件文字层疑似坏公式"）
            #    通过**邻块通道**重新进入 Tutor 提示词。
            #    实测：900 个候选窗口里 **283 个**含不可索引块，其中 242 个含已隔离的坏公式。
            #    `id` 一并带上是第 3 项要的（来源归属：说得清"实际发了哪些块"）。
            #    NULL 视为 True —— 与 `load_page_blocks_v2` 同口径，
            #    **不要用 bool()**（`bool(None)` 是 False，会静默反过来）。
            "indexable": r.indexable is not False,
            "is_generated": bool(r.is_generated),
            "note": r.note or "",
            "id": r.id,
            "source_kind": r.source_kind,
            # ⚠️ **`page` 也要带上**：拼接会把**下一页**的块接进来，
            #    而下游只标了命中页的单一页码 ⇒ "来源归属"就错了（核查第 3 项）。
            "page": r.page,
        })
        grouped[(r.doc_id, r.page)][-1].update(
            formula_number=r.formula_number or '', bbox=r.bbox or [], section=r.section or '')
    return grouped


def list_documents() -> list[dict]:
    """已导入的文档一览（按 doc_id 聚合）。"""
    with Session(get_engine()) as session:
        rows = session.execute(select(DocumentBlock)).scalars().all()
    agg: dict[str, dict] = {}
    for r in rows:
        d = agg.setdefault(r.doc_id, {"doc_id": r.doc_id, "doc_type": r.doc_type,
                                      "pages": set(), "blocks": 0, "by_type": {}})
        d["pages"].add(r.page)
        d["blocks"] += 1
        d["by_type"][r.type] = d["by_type"].get(r.type, 0) + 1
    out = []
    for d in agg.values():
        d["pages"] = len(d["pages"])
        out.append(d)
    return sorted(out, key=lambda x: x["doc_id"])


def delete_document(doc_id: str) -> dict:
    """删除一份文档的全部内容块与向量。返回删除统计。"""
    with Session(get_engine()) as session, session.begin():
        result = session.execute(
            delete(DocumentBlock).where(DocumentBlock.doc_id == doc_id)
        )
        removed = result.rowcount or 0
    points = _delete_points(doc_id)
    return {"blocks": removed, "points": points}


def _delete_points(doc_id: str) -> int:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    settings = get_settings()
    client = get_qdrant()
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(must=[FieldCondition(
            key="doc_id", match=MatchValue(value=doc_id))]),
    )
    return -1          # Qdrant 不回传删除数


def _scroll_points(doc_id: str, *, limit: int = 100_000) -> list[dict]:
    """滚动取出某文档的全部向量点（含 payload）。"""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    settings = get_settings()
    client = get_qdrant()
    if not client.collection_exists(settings.qdrant_collection):
        return []
    out: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(must=[FieldCondition(
                key="doc_id", match=MatchValue(value=doc_id))]),
            limit=1024, offset=offset, with_payload=True, with_vectors=False,
        )
        for p in points:
            out.append({"id": str(p.id), "payload": p.payload or {}})
        if offset is None or len(out) >= limit:
            break
    return out


def reconcile(doc_id: str, *, apply: bool = False,
              figures_root: str | Path | None = None) -> dict:
    """对账 PostgreSQL 与 Qdrant，可选修复。

    报告四类差异：
      · `pg_only`       —— PG 有、Qdrant 没有的块（索引缺失）
      · `qdrant_only`   —— Qdrant 有、PG 没有的点（**孤儿**）
      · `payload_diff`  —— 两边都有但 payload 关键字段不一致
      · `missing_figures` —— 引用的图片文件不存在

    `apply=True` 时：
      · 删掉 `qdrant_only` 的点（**只删该文档的，不动集合里其他文档**）
      · 为 `pg_only` 补建索引

    **不做"清空整个集合"**——集合里有演示教材点与缺 doc_id 的旧点，
    无差别清空会误伤。
    """
    blocks = load_blocks(doc_id)
    pg_by_id = {b["id"]: b for b in blocks}
    points = _scroll_points(doc_id)
    qd_by_id = {p["id"]: p for p in points}

    # **区分"故意不索引"与"索引确实缺失"**。
    # `indexable=False` 的块（如扫描件文字层里的坏公式）本来就不该进 Qdrant，
    # 把它们报成"索引缺失"会误导人去修复一个不是问题的问题。
    should_index = {i for i, b in pg_by_id.items() if b.get("indexable", True)}
    unindexed_by_design = sorted(set(pg_by_id) - should_index)
    pg_only = sorted(should_index - set(qd_by_id))
    qdrant_only = sorted(set(qd_by_id) - set(pg_by_id))
    payload_diff = []
    for pid in sorted(set(pg_by_id) & set(qd_by_id)):
        b, pl = pg_by_id[pid], qd_by_id[pid]["payload"]
        for key, actual in (("type", b.get("type")), ("page", b.get("page")),
                            ("figure_path", b.get("figure_path") or ""),
                            ("source_kind", b.get("source_kind"))):
            if pl.get(key) != actual:
                payload_diff.append({"id": pid, "field": key,
                                     "qdrant": pl.get(key), "pg": actual})

    root = Path(figures_root) if figures_root else Path("data/documents") / doc_id
    missing_figures = []
    for b in blocks:
        rel = b.get("figure_path")
        # **保留 `figures/` 子目录**：figure_path 存的是相对文档根的路径
        # （`figures/p0047_1.png`），取 `.name` 会把它剥掉、
        # 导致 637 个真实存在的图片全被报成缺失（踩过）。
        if rel and not (root / rel).is_file():
            missing_figures.append({"id": b["id"], "page": b["page"], "path": rel})

    # 孤儿点带上页码与内容摘要——只看一串 uuid 无法判断该怎么处置
    orphan_detail = [
        {"id": p["id"], "page": p["payload"].get("page"),
         "type": p["payload"].get("type"),
         "text": str(p["payload"].get("text", ""))[:60]}
        for p in (qd_by_id[i] for i in qdrant_only)
    ]

    pages = sorted({b["page"] for b in blocks})
    report = {
        "doc_id": doc_id,
        "pg_blocks": len(blocks),
        "qdrant_points": len(points),
        "pg_only": pg_only,
        "unindexed_by_design": unindexed_by_design,
        "qdrant_only": qdrant_only,
        "orphan_detail": orphan_detail,
        "payload_diff": payload_diff,
        "missing_figures": missing_figures,
        "pages": {"count": len(pages),
                  "first": pages[0] if pages else None,
                  "last": pages[-1] if pages else None,
                  "gaps": _page_gaps(pages)},
        "applied": None,
    }
    if not apply:
        return report

    actions = {}
    if qdrant_only:
        from qdrant_client.models import PointIdsList

        settings = get_settings()
        get_qdrant().delete(collection_name=settings.qdrant_collection,
                            points_selector=PointIdsList(points=qdrant_only))
        actions["deleted_orphans"] = len(qdrant_only)
    if pg_only:
        # 只补真正缺失的；`unindexed_by_design` 不动（那是设计如此）
        index_blocks([pg_by_id[i] for i in pg_only], replace=REPLACE_NONE)
        actions["reindexed"] = len(pg_only)
    if payload_diff:
        # **payload 不一致也要修**：此前 --apply 只删孤儿、补缺失，
        # 对 payload 差异只报告不处理——于是"对账跑完仍不一致"。
        # 重新 upsert 这些块即可（block id 相同，覆盖写）。
        diff_ids = {d["id"] for d in payload_diff}
        index_blocks([pg_by_id[i] for i in diff_ids if i in pg_by_id],
                     replace=REPLACE_NONE)
        actions["repayloaded"] = len(diff_ids)
    report["applied"] = actions or {"note": "无差异，未做改动"}
    return report


# ---------------------------------------------------------------- 导入批次

#: 索引与事实库一致，检索端可用
INDEX_READY = "ready"
#: 索引未就绪（导入中 / 索引没重建完）——**检索端必须排除**
INDEX_PENDING = "index_pending"
RUN_RUNNING = "running"
#: 已被更新批次取代的未完成批次。
#:
#: **标记而不是删除**——它是审计轨迹：说明"这次导入没跑完，后来被谁取代"。
#: 删掉的话，`--runs` 里那段历史就查不到了，而"为什么当时没导完"
#: 恰恰是排障时要回答的问题。
RUN_SUPERSEDED = "superseded"
#: 导入完成、索引已重建，**只是有页失败**。检索端**可见**——
#: 见 `INDEX_INVISIBLE_STATUSES`。
RUN_PARTIAL = "partial"
RUN_FAILED = "failed"

#: 检索端**排除**的状态（黑名单）。
#:
#: ⚠️ **用黑名单，而不是"只认 ready"的白名单。**
#: 原来的白名单把 `partial` 也排除了——而 `partial` 的索引与 PG 是**一致**的，
#: 只是有页失败（VL 输出无法解析）。实测代价：教材-2 下册与教材-5 电磁学
#: 各失败 **1 页**，**963 页内容全校搜不到**。代价明显大于收益。
#:
#: 用黑名单的另一个理由：**新增状态时默认可见**。
#: 白名单会让每个新状态都默认不可见，重演同一个 bug。
INDEX_INVISIBLE_STATUSES = frozenset({INDEX_PENDING, RUN_RUNNING, RUN_FAILED})


def is_index_visible(status: str) -> bool:
    """该文档当前是否可被检索。

    `partial` **可见**（索引一致、只是缺页）；
    `index_pending` / `running` / `failed` 不可见。
    """
    return status not in INDEX_INVISIBLE_STATUSES


def start_import_run(doc_id: str, *, source_sha256: str = "", source_name: str = "",
                     doc_type: str = "textbook", mode: str = "document",
                     pages_spec: str = "", model: str = "",
                     prompt_versions: dict | None = None) -> str:
    """开一次导入批次，返回 import_id。

    **同时把该文档标为 `index_pending`**——导入一开始，索引就不再与 PG
    一致了，检索端必须先看到"这批还没好"，而不是继续拿旧向量当最新结果。
    """
    import uuid as _uuid

    from app.db.tables import DocumentImportRun

    import_id = str(_uuid.uuid4())
    with Session(get_engine()) as session, session.begin():
        session.add(DocumentImportRun(
            id=import_id, doc_id=doc_id, source_sha256=source_sha256,
            source_name=source_name, doc_type=doc_type, mode=mode,
            pages_spec=pages_spec, model=model, prompt_versions=prompt_versions or {},
            status=INDEX_PENDING,
        ))
    return import_id


def finish_import_run(import_id: str, *, status: str, pages_total: int = 0,
                      pages_ok: int = 0, failed_pages: list[int] | None = None,
                      blocks: int = 0, indexed: int = 0, prompt_tokens: int = 0,
                      error: str = "") -> None:
    """收尾一次导入批次。**状态由调用方明确给出，不猜。**"""
    from datetime import datetime, timezone

    from app.db.tables import DocumentImportRun

    failed_pages = failed_pages or []
    with Session(get_engine()) as session, session.begin():
        run = session.get(DocumentImportRun, import_id)
        if run is None:
            logger.warning("导入批次 %s 不存在，收尾被忽略", import_id)
            return
        run.status = status
        run.pages_total = pages_total
        run.pages_ok = pages_ok
        run.pages_failed = len(failed_pages)
        run.failed_pages = failed_pages
        run.blocks = blocks
        run.indexed = indexed
        run.prompt_tokens = prompt_tokens
        run.error = error[:2000]
        run.finished_at = datetime.now(timezone.utc)


def set_index_status(doc_id: str, status: str) -> None:
    """把文档**最近一次**批次的状态改成 `status`。

    用于 `--no-index`（标 `index_pending`）与索引成功后（标 `ready`）。
    """
    from sqlalchemy import desc

    from app.db.tables import DocumentImportRun

    with Session(get_engine()) as session, session.begin():
        run = session.execute(
            select(DocumentImportRun).where(DocumentImportRun.doc_id == doc_id)
            .order_by(desc(DocumentImportRun.started_at)).limit(1)
        ).scalars().first()
        if run is None:
            return
        run.status = status


def doc_index_status(doc_id: str) -> str:
    """该文档最近一次批次的索引状态。没有批次记录时返回 `ready`。

    返回 `ready` 是**刻意的保守选择**：早期导入的文档没有批次记录，
    把它们一律当成"待修复"会让检索整体失效；而它们当时确实是可用的。
    """
    from sqlalchemy import desc

    from app.db.tables import DocumentImportRun

    try:
        with Session(get_engine()) as session:
            run = session.execute(
                select(DocumentImportRun).where(DocumentImportRun.doc_id == doc_id)
                .order_by(desc(DocumentImportRun.started_at)).limit(1)
            ).scalars().first()
    except Exception as exc:  # noqa: BLE001 事实库不可用不该让检索失败
        logger.info("无法读取索引状态，按 ready 处理：%s", type(exc).__name__)
        return INDEX_READY
    return run.status if run else INDEX_READY



def supersede_stale_runs(doc_id: str | None = None) -> int:
    """把**已被更新批次取代**的未完成批次标为 `superseded`，返回条数。

    什么叫"陈旧"：某文档有多个批次，**不是最新那个**、且状态还是
    `index_pending`。它们的处境是明确的——这次导入没跑完（进程被杀、
    锁冲突等），后来又有别的批次跑成了。

    为什么需要这个动作：
      · 它们让 `--runs` 一直报"可疑批次"，而文档其实早就 ready 了
        （误报会让人去修一个不是问题的问题）
      · 但**不能删**——"这次为什么没导完"是排障时要回答的问题
      · 文档状态取的是最新批次，所以标记它们**不影响检索**

    **最新的那个批次绝不标记**：它代表文档当前状态，
    哪怕是 `index_pending`（那说明索引确实待修复，不该被掩盖）。
    """
    from sqlalchemy import desc

    from app.db.tables import DocumentImportRun

    with Session(get_engine()) as session, session.begin():
        stmt = select(DocumentImportRun).order_by(desc(DocumentImportRun.started_at))
        if doc_id:
            stmt = (select(DocumentImportRun)
                    .where(DocumentImportRun.doc_id == doc_id)
                    .order_by(desc(DocumentImportRun.started_at)))
        runs = session.execute(stmt).scalars().all()

        seen: set[str] = set()
        n = 0
        for r in runs:                       # 按 started_at 倒序
            if r.doc_id not in seen:
                seen.add(r.doc_id)           # 每个文档的第一条 = 最新，跳过
                continue
            if r.status in (INDEX_PENDING, RUN_RUNNING):
                r.status = RUN_SUPERSEDED
                n += 1
    return n


def delete_import_run(run_id: str) -> bool:
    """删除一条批次记录。返回是否真的删掉了。

    **正常流程不该用它**——批次记录是审计轨迹，未完成的批次应当用
    `supersede_stale_runs()` 标记而不是删除。
    这里主要给测试收尾用：测试会往真实库里写批次，不清理就会累积。
    """
    from app.db.tables import DocumentImportRun

    with Session(get_engine()) as session, session.begin():
        run = session.get(DocumentImportRun, run_id)
        if run is None:
            return False
        session.delete(run)
    return True

def list_import_runs(doc_id: str | None = None, *, limit: int = 20) -> list[dict]:
    """导入批次一览（审计用：源文件摘要、模型版本、失败页）。"""
    from sqlalchemy import desc

    from app.db.tables import DocumentImportRun

    stmt = select(DocumentImportRun).order_by(desc(DocumentImportRun.started_at)).limit(limit)
    if doc_id:
        stmt = (select(DocumentImportRun)
                .where(DocumentImportRun.doc_id == doc_id)
                .order_by(desc(DocumentImportRun.started_at)).limit(limit))
    with Session(get_engine()) as session:
        rows = session.execute(stmt).scalars().all()
    return [
        {"id": r.id, "doc_id": r.doc_id, "status": r.status,
         "sha256": (r.source_sha256 or "")[:16], "mode": r.mode,
         "pages_spec": r.pages_spec, "model": r.model,
         "pages_total": r.pages_total, "pages_ok": r.pages_ok,
         "pages_failed": r.pages_failed, "failed_pages": (r.failed_pages or [])[:12],
         "blocks": r.blocks, "indexed": r.indexed, "prompt_tokens": r.prompt_tokens,
         "started_at": r.started_at.isoformat() if r.started_at else "",
         "finished_at": r.finished_at.isoformat() if r.finished_at else "",
         "error": r.error}
        for r in rows
    ]


def _page_gaps(pages: list[int]) -> list[str]:
    """页覆盖里的断档（连续缺失的区间）。"""
    if not pages:
        return []
    gaps = []
    for a, b in zip(pages, pages[1:]):
        if b - a > 1:
            gaps.append(f"{a + 1}-{b - 1}")
    return gaps


#: 短于这个长度的正文块，入库时补上同页邻块作为上下文。**0 = 关闭（当前默认）**。
#:
#: ## 问题（实测，尚未解决）
#: 全量导入后正文块长度中位数只有 **12 字，43% 不足 10 字**
#: （如 `'动量守恒'`、`'1. 角动量'`）——这种块被检索到时给 Tutor
#: 提供不了任何上下文。
#:
#: ## 试过两条路，都不成
#: 1. **调段落合并阈值**：8→20pt 时中位数从 12 掉到 10 字、短块占比从
#:    44% 升到 48%，**越调越差**。说明碎片不是合并太保守，而是这版 OCR
#:    文字层本身就碎（标题、公式碎片、图号、页码各占一部分）。
#: 2. **入库时补同页邻块**：正文检索确实变好了（"牛顿第二定律"从返回
#:    碎片变成返回完整句子），但**破坏了公式检索**——把 3 个邻块拼进
#:    `embed_text` 会让向量代表"一堆话题的混合"，具体查询的信号被稀释，
#:    "机械能守恒的条件" 变成命中 `l > l_*` 这种碎片公式。
#:
#: 因为没有余量把参数调好，**先关掉**：留一个未验证的权衡不如保持原状。
#: 正确方向可能是"分层检索"（先按标题/章节定位再检索），而不是简单地
#: 把邻居拼进同一个向量。
SHORT_BLOCK_CHARS = 0

#: 补上下文时，前后各取多少个同页邻块
CONTEXT_NEIGHBORS = 3


def _expand_embed_text(blocks: list[dict]) -> None:
    """给过短的正文块补上同页邻块，**就地修改 embed_text**。

    只动 `embed_text`，不动 `text`——检索靠前者，展示与引用靠后者。

    ⚠️ **当前默认关闭**（`SHORT_BLOCK_CHARS = 0`）。实测它会改善正文检索、
    但破坏公式检索，参数没调到两全（详见 `SHORT_BLOCK_CHARS` 的说明）。
    """
    if SHORT_BLOCK_CHARS <= 0:
        return
    by_page: dict[int, list[dict]] = {}
    for b in blocks:
        by_page.setdefault(b.get("page") or 0, []).append(b)

    for page_blocks in by_page.values():
        texts = [(b.get("type"), (b.get("text") or "").strip()) for b in page_blocks]
        for i, b in enumerate(page_blocks):
            if b.get("type") != "text":
                continue
            if len(texts[i][1]) >= SHORT_BLOCK_CHARS:
                continue
            lo = max(0, i - CONTEXT_NEIGHBORS)
            hi = min(len(page_blocks), i + CONTEXT_NEIGHBORS + 1)
            parts = [t for j, (ty, t) in enumerate(texts[lo:hi], start=lo)
                     if ty == "text" and t and j != i]
            if not parts:
                continue
            b["embed_text"] = (texts[i][1] + " ｜ " + " ".join(parts))[:600]


def delete_points(doc_id: str, pages: list[int] | None = None) -> None:
    """按范围删除向量点。

    **删除范围必须与写入范围一致**，否则会误删同一文档其他页的向量。
    `pages=None` 表示删该文档的全部点（全量重导才用）。
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    settings = get_settings()
    client = get_qdrant()
    if not client.collection_exists(settings.qdrant_collection):
        return
    must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if pages:
        must.append(FieldCondition(key="page", match=MatchAny(any=list(pages))))
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(must=must),
    )


def index_blocks(blocks: list[dict], *, batch_size: int = 64,
                 replace: str = REPLACE_PAGES) -> int:
    """把内容块向量化写入 Qdrant。返回写入条数。

    入库的文本是 `embed_text`（已按类型做过处理）：
      · 正文/图注 → 本身就是可检索文本
      · 公式 → `latex_to_readable` 转换 + 页码/章节/编号上下文的拼接结果

    `replace` 语义与 `persist_blocks` **保持一致**（默认只替换本次覆盖的页）：
    **upsert 之前先按同样的范围删除**，否则页内块数变少时旧点会成为孤儿。

    这一步是"去掉只 upsert 的隐患"的关键——此前孤儿向量就是这么来的。
    """
    from qdrant_client.models import PointStruct

    if not blocks:
        return 0
    settings = get_settings()
    client = get_qdrant()
    from app.rag.embedding import get_embedding_client
    from app.rag.ingest import _ensure_collection

    doc_ids = sorted({b.get("doc_id", "") for b in blocks if b.get("doc_id")})
    pages = _pages_of(blocks)

    embedder = get_embedding_client()
    # 先探一次维度，避免空集合时建错
    dim = len(embedder.embed(["维度探测"])[0])
    _ensure_collection(client, settings.qdrant_collection, dim)

    # **先删后写**：删除范围 = 写入范围。分文档做，避免跨文档误删。
    if replace == REPLACE_DOCUMENT:
        for doc_id in doc_ids:
            delete_points(doc_id)
    elif replace == REPLACE_PAGES:
        for doc_id in doc_ids:
            doc_pages = sorted({int(b["page"]) for b in blocks
                                if b.get("doc_id") == doc_id and b.get("page") is not None})
            delete_points(doc_id, doc_pages)

    written = 0
    skipped = 0
    for start in range(0, len(blocks), batch_size):
        batch = [b for b in blocks[start:start + batch_size]
                 if b.get("indexable", True)]
        skipped += len(blocks[start:start + batch_size]) - len(batch)
        if not batch:
            continue
        texts = [(b.get("embed_text") or b.get("text") or "").strip() for b in batch]
        vectors = embedder.embed(texts)
        points = []
        for b, vec in zip(batch, vectors):
            points.append(PointStruct(
                id=b["id"],
                vector=vec,
                payload={
                    # 兼容既有检索方（retriever 早就读这几个字段）
                    "text": b.get("text", ""),
                    "source": b.get("doc_id", ""),
                    "page": b.get("page"),
                    "chapter": b.get("section", ""),
                    # 新增：类型、图片、溯源
                    "type": b.get("type", "text"),
                    "doc_id": b.get("doc_id", ""),
                    "block_index": b.get("block_index", 0),
                    "figure_path": b.get("figure_path", ""),
                    "figure_labels": b.get("figure_labels") or [],
                    "formula_number": b.get("formula_number", ""),
                    "is_generated": bool(b.get("is_generated", False)),
                    "source_kind": b.get("source_kind", ""),
                },
            ))
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        written += len(points)
    if skipped:
        logger.info("入索引：%d 块；跳过 %d 块（已隔离，原文保留在 PG）", written, skipped)
    else:
        logger.info("入索引：%d 块", written)
    return written
