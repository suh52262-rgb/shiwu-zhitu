"""教材知识检索（Phase 4；P-导入轮补上上下文扩展）。

统一接口：retrieve(query, top_k) → [{"text", "source", "page", ..., "context"}, ...]
已被 LangGraph 的 retrieve_knowledge 节点与 /knowledge/search API 使用。

## 为什么要扩上下文（以及为什么不在索引时扩）

实测全量导入后正文块长度**中位数只有 12 字、43% 不足 10 字**
（`'动量守恒'`、`'1. 角动量'`）。这种块命中得再准，也**给 Tutor 提供不了
上下文**——它没法据此讲清一句话。

先试过在**索引时**把同页邻块拼进 `embed_text`：正文检索确实变好了，
但**破坏了公式检索**——把几个邻块拼进同一个向量会让它代表"一堆话题的
混合"，具体查询的信号被稀释（"机械能守恒的条件"变成命中 `l > l_*`）。

**正确的做法是扩「返回的文本」，不扩「被向量化的文本」**：
  · `embed_text` 保持干净 → 向量质量不受影响
  · 返回时从 PostgreSQL 取同页邻块 → Tutor 拿到有上下文的原文

这也正是"PostgreSQL 是事实、Qdrant 是索引"这套两层设计该发挥作用的地方：
向量库负责**找得准**，事实库负责**给得全**。
"""
import logging

from app.config import get_settings
from app.db.qdrant import get_qdrant
from app.rag.embedding import get_embedding_client

logger = logging.getLogger(__name__)

#: 命中块前后各取多少个同页块（命中块是"正文"时用这个，对称）
#: 短于这个长度的块被打折——**它结构上不可能回答"内容是什么"**。
#:
#: ## 为什么需要（实测）
#: `'3.动能'`（4 字）在查询「动能定理的内容是什么」上得 0.709，排在真正
#: 含答案的 193 字块（0.703）**前面**，把 `top_k=2` 的槽位占掉。
#:
#: 原因是**结构性**的：4 字块的向量几乎就是概念名本身，任何
#: "XX定理的内容是什么"式的问法都和它天然高分；而实质内容块的向量被稀释。
#:
#: 所以这不是调参凑答案，是把"**答案需要有实质内容**"这条先验编码进去。
#: 修复后四条检索用例全过（Hit Rate 75%→100%，MRR 46%→71%）。
SHORT_BLOCK_CHARS = 10

#: 打折因子。**刻意轻微**——短块也可能是对的（`'动量守恒：'` 配上下文
#: 扩展后是可用结果），只是不该压过有实质内容的块。
#: 不要为了某条用例把它调得更重：那会误伤图注、定义式这类合法的短结果。
SHORT_BLOCK_FACTOR = 0.9


def _length_factor(text: str) -> float:
    return SHORT_BLOCK_FACTOR if len((text or "").strip()) < SHORT_BLOCK_CHARS else 1.0


CONTEXT_NEIGHBORS = 4

#: 命中块短于这个长度（多半是**标题**），改用不对称取值：
#: **少取前面、多取后面**。
#:
#: 为什么：中文教材里标题后面跟的是本节内容，前面是上一节。
#: 实测 `'1. 角动量'`（p217）取前后对称邻块时，context 变成了上一节的
#: 陀螺章动——完全跑题。标题在页面底部时更严重，本节内容全在下一页。
HEADING_LIKE_CHARS = 12

#: 标题类命中的邻块取值：[前面几个, 后面几个]
HEADING_CONTEXT = (1, 8)

#: 扩出来的上下文最长多少字（防止一整页灌进提示词）
CONTEXT_MAX_CHARS = 600

#: 短于这个长度的邻块不并入上下文。
#:
#: 作用是滤掉**目录页与索引页**：实测 `'机械能守恒'`（p14）的邻块是
#: `'91 习题 93 第三章 机械能守恒 98 81.功和能· 98 1.1历史性的评述'`
#: ——那是目录条目，拼进去只会误导 Tutor。
DO_NOT_EXPAND_BELOW = 8


def _looks_like_content(text: str) -> bool:
    """邻块是不是"正文内容"。

    目录条目/页码的特征是**极短且几乎全是数字与标点**。
    这类块拼进上下文只会制造噪声，不如不要。

    占比按**非空白字符**算——按整串算的话，
    `'91 习题 93 第三章 98'` 会被空格稀释到刚好卡在阈值上（踩过）。
    """
    stripped = (text or "").strip()
    if len(stripped) < DO_NOT_EXPAND_BELOW:
        return False
    dense = "".join(c for c in stripped if not c.isspace())
    if len(dense) <= 20:
        digits = sum(1 for c in dense if c.isdigit())
        if digits / max(1, len(dense)) > 0.4:
            return False            # 短且数字占四成以上 → 多半是目录/页码
    return True


def _expand_context(hits: list[dict], *, include_generated: bool = False,
                    context_policy: str | None = None) -> None:
    """给命中块补上同页邻块，写进 `context` 字段（就地修改）。

    `text` 保持原样不动——引用与展示仍用原文，只有喂给 LLM 的上下文变长。

    **数据库不可用时静默跳过**（`context` 退化为 `text`）：
    检索本身不该因为事实库掉线而失败。这也是两层设计的意义——
    向量库负责**找得准**，事实库负责**给得全**。
    """
    import os
    context_policy = context_policy or os.getenv('RAG_CONTEXT_POLICY', 'legacy')
    if context_policy not in {'legacy', 'formula_refs_v1', 'formula_refs_v2'}:
        raise ValueError('Unknown RAG context policy')
    for h in hits:
        # Derived data belongs to this expansion, never to a previous policy.
        for key in ('context_blocks', 'context_link_candidates', 'context_link_omissions',
                    'context_policy', 'context_status', 'context_reason'):
            h.pop(key, None)
        h['context'] = h.get('text', '')
    targets = [h for h in hits if h.get("page") and h.get("source")]
    if not targets:
        return

    # **按文档分组**：命中可能跨多份文档，只取第一个 doc_id 会让其余命中
    # 取不到邻块（踩过——context 静默退化成原文，看起来像"扩展没生效"）。
    # **顺带把下一页也载入**：标题落在页面底部时，本节内容在下一页。
    by_doc: dict[str, set[int]] = {}
    for h in targets:
        page = h["page"]
        by_doc.setdefault(h["source"], set()).update({page, page + 1})
        if context_policy != 'legacy' and page > 1:
            by_doc[h['source']].add(page-1)

    grouped: dict = {}
    try:
        from app.rag import block_store

        for doc_id, pages in by_doc.items():
            grouped.update(block_store.load_page_blocks(doc_id, sorted(pages)))
    except Exception as exc:  # noqa: BLE001 事实库掉线不该让检索失败
        logger.info("上下文扩展跳过（无法读取内容块）：%s", type(exc).__name__)
        return

    for h in targets:
        # 把命中页与下一页的块接起来，标题落在页底时也能取到本节内容
        page_blocks = list(grouped.get((h["source"], h["page"])) or [])
        page_blocks += list(grouped.get((h["source"], h["page"] + 1)) or [])
        if not page_blocks:
            continue
        identity = h.get('block_id')
        if identity:
            matches = [i for i, b in enumerate(page_blocks) if b.get('id') == identity
                       and b.get('page', h['page']) == h['page']]
        else:
            # Legacy inputs without IDs are usable only when their text is unique.
            matches = [i for i, b in enumerate(page_blocks) if b.get('text') == h.get('text')
                       and b.get('page', h['page']) == h['page']]
        if len(matches) != 1:
            h.update(context_status='not_expanded', context_reason='hit_identity_missing_or_ambiguous')
            continue
        idx = matches[0]
        anchor = page_blocks[idx]
        if anchor.get('text') != h.get('text'):
            h.update(context_status='not_expanded', context_reason='hit_text_changed')
            continue
        if anchor.get('indexable') is False or anchor.get('is_generated') and not include_generated:
            h.update(context_status='not_expanded', context_reason='hit_fact_filtered')
            continue
        h['context_status'] = 'expanded' if identity else 'legacy_unique_text'


        hit_text = h.get("text") or ""
        if len(hit_text.strip()) < HEADING_LIKE_CHARS:
            before, after = HEADING_CONTEXT      # 标题：少取前、多取后
        else:
            before = after = CONTEXT_NEIGHBORS

        lo = max(0, idx - before)
        hi = min(len(page_blocks), idx + after + 1)
        # ---- ① 邻块拼接**沿用检索侧的过滤规则**（2026-10-10 RAG 核查第 1 项）----
        # ⚠️ 原实现只判"有没有文本" ⇒ **隔离块通过邻块通道重新进入提示词**：
        #    900 个候选窗口 **283 个**含不可索引块（其中 242 个含已隔离的坏公式）。
        #    ⚠️ 而且**主路是拦了的**（`retrieve()` 的 include_generated 过滤）——
        #    是这条**支路**绕过了它。判据与主路一致，不许各写一套。
        window = [b for b in page_blocks[lo:hi]
                  if b.get("text")
                  and b.get("indexable", True)
                  and not (b.get("is_generated") and not include_generated)
                  and (b["text"] == hit_text or _looks_like_content(b["text"]))]
        # ---- ② **命中优先的预算分配**（核查第 2 项）----
        # ⚠️ 原实现按"前邻块在前"顺序拼、再从**头部**截 600 ——
        #    命中若落在 600 字之后就被整段截掉（实测 **3 个**命中起点 >600），
        #    而 Tutor 还会再截 400（**20 个**命中起点 >400）。
        # ⇒ **先保命中，再补邻块**：命中放最前，保证它在任何下游截断里都活着。
        #    ⚠️ 这**改变了材料的阅读顺序**（命中块先出现）—— 有意为之，并在此写明：
        #    这段材料的用途是"**检索到的依据**"，命中块是锚点；顺序让位于"依据不能被截掉"。
        linked = []
        if context_policy != 'legacy':
            from app.rag.formula_links import linked_formulas
            pool = list(grouped.get((h['source'], h['page']-1)) or [])
            pool += list(grouped.get((h['source'], h['page'])) or [])
            anchors = [b for b in page_blocks[lo:hi] if b.get('page') == h['page']]
            linked, omissions = linked_formulas(anchors, pool, hit_page=h['page'],
                                               include_generated=include_generated,
                                               hit_id=page_blocks[idx].get('id') if context_policy == 'formula_refs_v2' else None)
            h['context_link_omissions'] = omissions
            h['context_policy'] = context_policy
            if context_policy == 'formula_refs_v2':
                h['context_link_candidates'] = linked
                linked = []  # Preserve original context; delivery allocates spare budget.
        ordered = [anchor] + linked
        linked_ids = {b['id'] for b in linked}
        ordered += [b for b in window if b['text'] != hit_text and b.get('id') not in linked_ids]
        parts, spans, cursor = [], [], 0
        for block in ordered:
            separator = 1 if parts else 0
            remaining = CONTEXT_MAX_CHARS - cursor - separator
            if remaining <= 0:
                break
            fragment = block["text"][:remaining]
            start = cursor + separator
            parts.append(fragment)
            spans.append({
                "id": block.get("id"), "page": block.get("page") or h.get("page"),
                "source": h.get("source"), "block_index": block.get("block_index"),
                "type": block.get("type"), "source_kind": block.get("source_kind"),
                "context_start": start, "context_end": start + len(fragment),
                "text_start": 0, "text_end": len(fragment),
                **({'context_link': block['context_link']} if block.get('context_link') else {}),
            })
            cursor = start + len(fragment)
        h["context"] = " ".join(parts)
        # Only catalog fragments actually present in the capped context, in
        # rendering order. The Tutor's smaller window intersects these spans;
        # blocks beyond that window must not be called model-delivered evidence.
        h["context_blocks"] = spans


def retrieve(query: str, top_k: int = 5, *, include_generated: bool = False,
             expand_context: bool = True) -> list[dict]:
    """从教材知识库检索与 query 最相关的内容。

    返回格式（保留原有 5 个字段以免破坏既有调用方，其余为新增）：
        [{"text": "...", "source": "...", "page": 85, "chapter": "机械能",
          "score": 0.83,
          "context": "...",          # text 加上同页邻块，供 LLM 读
          "type": "text|formula|figure_caption|figure",
          "figure_path": "figures/p0047_1.png", "figure_labels": ["图1-17"],
          "is_generated": False}, ...]

    **`include_generated=False` 是默认值**：机器生成的内容（VL 写的图注）
    不能冒充教材原文——Tutor 说"依据教材第 12 页"时，那句话必须真的在书上。

    知识库为空（尚未运行 build_rag）时返回 []，不抛异常，
    保证 Agent 工作流不因 RAG 缺席而阻塞。
    """
    if not query.strip():
        return []
    settings = get_settings()
    client = get_qdrant()
    if not client.collection_exists(settings.qdrant_collection):
        logger.info("知识库尚未构建（先运行 scripts/build_rag.py）")
        return []

    from time import perf_counter
    setup_started = perf_counter()
    embedder = get_embedding_client()
    embedding_setup_sec = perf_counter() - setup_started
    embed_started = perf_counter()
    vector = embedder.embed([query])[0]
    embed_sec = perf_counter() - embed_started
    # 多召回一些，给"按类型加权"留出重排空间
    search_started = perf_counter()
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=max(top_k * 3, top_k),
        with_payload=True,
    )
    retrieve_sec = perf_counter() - search_started

    #: 检索侧权重。**这几个数是估的，没有实测依据**，应在真实查询集上调。
    #: `figure`（VL 图注）最低，因为它是机器生成、可信度最低。
    weights = {"text": 1.0, "formula": 1.0, "figure_caption": 0.9, "figure": 0.5}

    scored: list[tuple[float, dict]] = []
    for point in response.points:
        payload = point.payload or {}
        if payload.get("is_generated") and not include_generated:
            continue
        btype = payload.get("type", "text")
        item = {
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
            "page": payload.get("page"),
            "chapter": payload.get("chapter", ""),
            "score": round(point.score, 3),
            # ---- ⚠️ **稳定身份**（2026-10-10 评测工具要求）----
            # `point.id` 是 Qdrant 的**稳定块身份**（与 PG 的 `document_blocks.id` 同源），
            # 必须透传：评测要用它判"直接命中"，而**不能用 `block_index` 拼 `bNone`**
            # （`block_index` 在命中字典里拿不到，拼出来的身份是假的）。
            # ⚠️ **取不到时给 None，不抛异常**（2026-10-11 修）：
            # 测试替身（SimpleNamespace）没有 `.id`；而生产里身份缺失是**可预期的**情况
            # ⇒ 评测侧据此标「**不可验证**」，而不是崩掉、也不是编一个身份出来。
            "block_id": str(getattr(point, "id", "") or "") or None,
            "block_index": payload.get("block_index"),
            # ---- 新增：类型与图片关联 ----
            "type": btype,
            "figure_path": payload.get("figure_path", ""),
            "figure_labels": payload.get("figure_labels") or [],
            "formula_number": payload.get("formula_number", ""),
            "is_generated": bool(payload.get("is_generated", False)),
            "source_kind": payload.get("source_kind", ""),
        }
        scored.append((point.score * weights.get(btype, 1.0)
                       * _length_factor(item.get("text", "")), item))

    scored.sort(key=lambda x: -x[0])

    # **排除索引过期的文档**：`--no-index` 或导入中途失败的文档，
    # PG 是新的而 Qdrant 是旧的/部分的。不排除就会把旧向量当最新结果返回。
    # 按命中的 doc_id 集合查一次状态即可（命中通常集中在少数文档）。
    from app.rag import block_store as _bs

    ready_cache: dict[str, bool] = {}
    kept = []
    # **边收集边去重**，凑够 top_k 条互不相同的。
    #
    # 原写法是"先取满 top_k 再去重"——一条重复就把结果削到 top_k−1。
    # 实测 `'机械能守恒的条件'` 因此只返回 7 条（8 条里有一对同文本）。
    # 用户要 8 条就该给 8 条不同的（池子里有的话）。
    seen_keys: set[tuple] = set()
    for score, item in scored:
        doc_id = item.get("source") or ""
        if doc_id not in ready_cache:
            try:
                # `partial`（导入完成、索引一致、只是有页失败）**可见**；
                # 只有 `index_pending` / `running` / `failed` 才排除。
                ready_cache[doc_id] = _bs.is_index_visible(
                    _bs.doc_index_status(doc_id))
            except Exception:  # noqa: BLE001 状态查不到时按可用处理
                ready_cache[doc_id] = True
        if not ready_cache[doc_id]:
            continue
        key = (item.get("source"), item.get("page"), (item.get("text") or "").strip())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(item)
        if len(kept) >= top_k:
            break
    # 去重已在上面完成。说明保留在此：
    #
    # 同一条图注在库里有两份：正文块（文字层抽段落时带上的）与图注块
    # （图注检测产出的）。实测真实教材 637 条图注里 **374 条（58.7%）重复**，
    # 另有 15 条 formula/text 重复。
    #
    # 不去重的话，`top_k=5` 命中图注时会有半个列表是同一条内容——
    # 有效结果数直接减半，而且前端展示也是复读。
    #
    # 放在检索侧而不是只靠导入侧拦截：**已有数据不用重导就能立刻生效**。

    hits = kept
    context_started = perf_counter()
    if expand_context:
        _expand_context(hits, include_generated=include_generated)
    context_sec = perf_counter() - context_started
    for hit in hits:
        hit.update(embedding_setup_sec=embedding_setup_sec, embed_sec=embed_sec,
                   retrieve_sec=retrieve_sec, context_sec=context_sec)
    return hits
