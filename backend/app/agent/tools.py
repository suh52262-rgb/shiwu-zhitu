"""Agent RAG Tool 层（P7，任务书第 10 节）。

RAG 不是每轮强制执行的步骤，而是 Agent 根据状态决定是否调用的 Tool：
任务书第 10 节的分工——成员 B 负责知识库内容质量，主框架负责 Tool 接入与调用策略。

当前调用策略（在 LangGraph 的条件路由中实现，见 agent/graph.py）：
diagnose 之后，仅当"存在错误 / 已知问题 / 薄弱知识点"时才调用 retrieve_knowledge。

未来扩展的工具（依赖成员 B 的数据，先占位）：
- retrieve_similar_questions：相似题检索（题库）
- retrieve_school_exam：历年真题检索（期中期末）
- retrieve_student_mistakes：学生错题检索（student_profiles 积累）
"""
import logging

from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


def retrieve_knowledge(
    query: str, knowledge_points: list[str] | None = None, top_k: int = 5,
    status: dict | None = None,
) -> list[dict]:
    """教材知识检索（主 Tool，任务书第 10 节统一接口）。

    ## 查询扩展（解决"学生口语 vs 教材术语"的鸿沟）

    `docs/检索失败归因分析.md` 的结论：golden set 里的未命中**不是索引问题**，
    而是用词不匹配——学生说"重量和高度乘起来"，书上写"重量 w 和高度 h 的乘积"；
    用 gold 原文去查时，两条都排第 1。

    所以这里**用原文和改写各检索一次，再融合**：
      · 只用改写 → 改写跑偏时会全错
      · 只用原文 → 就是扩展前的状态
      · **两路一起 → 交错排**（原查询固定占据 1、3、5… 位，
        所以**改写完全跑偏时 top-5 里仍有三条来自原查询**）
      · 实测「只用改写」比基线低 20 个百分点（55% vs 75%），
        说明改写**确实会跑偏**，这个保证不是理论担忧

    扩展或融合失败时**静默退回单路**，检索本身不受影响。

    Args:
        query: 检索语句（通常是诊断出的问题描述）。
        knowledge_points: 可选知识点约束，会拼入检索 query 提升召回针对性。
        top_k: 返回条数。

    Returns:
        [{"text", "source", "page", "chapter", "score"}, ...]；
        知识库为空 / 输入为空时返回 []，不抛异常（保证工作流不阻塞）。
    """
    if not query.strip() and not knowledge_points:
        return []
    from time import perf_counter
    total_started = perf_counter()
    final_query = " ".join(filter(None, [query.strip(), *(knowledge_points or [])]))

    from app.rag.query_expansion import ENABLED, expand_query, merge_interleave

    variants: list[str] = []
    if ENABLED:
        try:
            variants = expand_query(final_query)
        except Exception as exc:  # noqa: BLE001 扩展绝不能阻断检索
            logger.info("查询扩展异常，退化单路：%s", type(exc).__name__)

    if not variants:
        # **取更深的候选交给重排**：只取 top_k 的话，重排没有余地
        # （实测重排吃 top-10；取 top_k 会让它退化成 no-op）。
        results = retrieve(final_query, top_k=max(top_k, 10))
    else:
        # 每路多取一些再融合——只取 top_k 的话，某一路排第 6 的正确结果
        # 会因为另一路没召回到而拿不到融合分（实测踩过这个直觉）。
        depth = max(top_k, top_k * 2)
        lists = [retrieve(final_query, top_k=depth)]
        for v in variants:
            try:
                lists.append(retrieve(v, top_k=depth))
            except Exception as exc:  # noqa: BLE001 单路失败不影响其余
                logger.info("改写路检索失败，跳过：%s", type(exc).__name__)
        merged = merge_interleave(lists)
        results = merged[:max(top_k, 10)]
        logger.debug("查询扩展：%d 个变体，融合 %d 路 -> %d 条",
                     len(variants), len(lists), len(results))

    # ---- LLM 重排（P18 后续：唯一被实测证明有效的检索改动）----
    #
    # 实测（300 条 golden set，配对检验）：
    #   基线 76.0% → 重排 81.3%，净 +16 条，翻转 19 好 / 3 坏，p = 0.0009
    #
    # 而前三轮"缩放既有分数"的改动全部无效：
    #   查询扩展 +2.5%(p=0.18)、长度惩罚 +0.7%、标题降权 **+0.0%**
    # 原因：top-5 的**成员**由向量相似度决定，缩放改不动成员。
    # 重排引入的是 query-passage 交互——以往排序里**没有的信息**。
    #
    # 重排**不打乱长度**（只换序），失败时静默退回原序。
    from app.rag.rerank import rerank
    from time import perf_counter
    rerank_started = perf_counter()

    try:
        # ⚠️ `status` 是**可选出参**（2026-10-10 拒答分支）：
        #    重排要如实报告"明确判无相关 / 超时 / 解析不了 / 成功"四种，
        #    而**判无相关时它返回空列表** —— 空列表本身说明不了是哪一种。
        results = rerank(final_query, results, status=status)
    except Exception as exc:  # noqa: BLE001 重排绝不能阻断检索
        logger.info("重排异常，退回原序：%s", type(exc).__name__)
        if status is not None:
            status.clear()
            status.update({"rerank_status": "failed",
                           "rerank_reason": f"{type(exc).__name__}: {exc}"})

    rerank_sec = perf_counter() - rerank_started
    for item in results:
        item['rerank_sec'] = rerank_sec
        item['rerank'] = (status or {}).get('rerank_status', 'unknown')
    if status is not None:
        status['timings'] = {'rerank_stage_sec':rerank_sec}

    # **截回 top_k**：上面为了让重排有腾挪余地取了更深的候选，
    # 但下游契约是 top_k 条——不截会让提示词里多塞一倍内容。
    results = results[:top_k]
    total_sec = perf_counter() - total_started
    for item in results:
        item['total_sec'] = total_sec
    if status is not None:
        status.setdefault('timings', {})['total_sec'] = total_sec

    logger.debug(
        "retrieve_knowledge(query=%r, knowledge_points=%r, top_k=%d) -> %d hits",
        query, knowledge_points, top_k, len(results),
    )
    from app.services.rag_materials import policy
    if policy() == 'selected_evidence_v3' and results:
        from app.rag.retriever import _expand_context
        from app.services.rag_selection import select
        _expand_context(results, context_policy='formula_refs_v2')
        plan=select(results,final_query)
        results[0]['_selected_material']=plan
        if status is not None:
            status['material_selection_status']=plan['selection_status']
            status['timings']['material_selection_sec']=plan['selection_seconds']
            status['timings']['total_sec']=perf_counter()-total_started
        for item in results:
            item['total_sec']=perf_counter()-total_started
    return results


def retrieve_similar_questions(query: str, top_k: int = 3) -> list[dict]:
    """相似题检索。TODO：等待成员 B 提供题库数据后实现。"""
    raise NotImplementedError("retrieve_similar_questions 需要成员 B 的题目库数据")


def retrieve_school_exam(query: str, top_k: int = 3) -> list[dict]:
    """历年真题检索。TODO：等待成员 B 提供期中期末真题后实现。"""
    raise NotImplementedError("retrieve_school_exam 需要成员 B 的历年真题数据")


def retrieve_student_mistakes(student_id: str, top_k: int = 3) -> list[dict]:
    """学生错题检索。TODO：依赖 student_profiles / 错题记录积累后实现。"""
    raise NotImplementedError("retrieve_student_mistakes 需要错题记录积累")


# Tool 注册表：供调用策略、P9 Tracing 与未来的 tool-binding 使用
RAG_TOOLS = {
    "retrieve_knowledge": retrieve_knowledge,
    "retrieve_similar_questions": retrieve_similar_questions,
    "retrieve_school_exam": retrieve_school_exam,
    "retrieve_student_mistakes": retrieve_student_mistakes,
}
