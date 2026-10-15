"""证据块在投递链上的**落点分类**（纯函数，可测）。

## 为什么单独抽出来
```
原来这段逻辑埋在评测脚本里 ⇒ **测不到** ⇒
而它正是我出过错的地方：`delivered_rag_sources()` 没带 `id` 时，
`delivered` 集合恒空 ⇒ **每个块都被判成"未投递"** ⇒
「线2 满足任一充分证据集 = 0」等结论**全是假的**，而我把它当真实发现报了两次。
⇒ 抽成纯函数 + 补测试，是让这类错**下次跑测试就暴露**。
```

## 五类落点（用户口径：不得合并）
```
delivered                     该块**确实进了送给模型的片段**
necessary_content_truncated   进了**前 N 条**的扩展目录，但**落在预算之外**（真实截断）
rank_excluded                 召回到了，但**名次在前 N 条之外**
expansion_missed              命中在列，但**它自己的目录里没有这一块**
not_recalled                  **根本没召回**
unverifiable_identity         块身份缺失 ⇒ **不可验证**（⚠️ **不得记成"未投递"**）
```
"""
from __future__ import annotations

DELIVERED = "delivered"
TRUNCATED = "necessary_content_truncated"
RANK_EXCLUDED = "rank_excluded"
EXPANSION_MISSED = "expansion_missed"
NOT_RECALLED = "not_recalled"
UNVERIFIABLE = "unverifiable_identity"


def classify_block(block_id, *, delivered: set, top_ctx: set,
                   ctx_by_rank: dict, order: list, tutor_hits: int = 3) -> str:
    """判定**一个证据块**的落点。

    ⚠️ `block_id` 为空/None ⇒ **不可验证** —— 这是一种**独立结论**，
    **既不等于"未投递"，也不等于"未召回"**（身份缺失时我们**根本无从判断**）。
    """
    if not block_id:
        return UNVERIFIABLE
    bid = str(block_id)
    if bid in delivered:
        return DELIVERED
    if bid in top_ctx:
        return TRUNCATED
    for rank in sorted(ctx_by_rank):
        if bid in ctx_by_rank[rank]:
            # 只在**前 N 条**的目录里出现（top_ctx 之外不可能；此处兜底）
            return TRUNCATED if rank <= tutor_hits else RANK_EXCLUDED
    if bid in order:
        return EXPANSION_MISSED
    return NOT_RECALLED


def summarize(categories: list[str]) -> dict:
    out: dict = {}
    for c in categories:
        out[c] = out.get(c, 0) + 1
    return out
