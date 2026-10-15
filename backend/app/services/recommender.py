"""做完一道题后推荐下一题。

## 判据（为什么用知识点重合）
**不引入新模型、不用向量检索** —— 题库每道题都标了 `knowledge_points`，
学生画像里有 `weak_concepts`，**两者的交集就是最直接的信号**。

> 一个学生卡在"动量定理"，那就该推一道**用到动量定理**的题。

## 打分
```
得分 = |该题知识点 ∩ 薄弱知识点|
```
平手时的排序（按顺序依次比）：
1. **没做过的优先** —— 做过的题重推没有意义
2. **难度低者优先** —— 先易后难；刚被卡住就推一道更难的，只会更挫败
3. **题目 id** —— 保证结果稳定（不依赖字典顺序）

## 冷启动
学生还没暴露薄弱点（全对、或刚注册）时**不能返回空**。
退化为：**没做过且难度最低的几道** —— 至少是有用的，而不是一片空白。
"""
from __future__ import annotations

from typing import Iterable, Optional

#: 默认推几道
DEFAULT_LIMIT = 3

#: 推荐理由的分档（给 UI 显示用）
REASON_WEAK = "weak"        # 命中薄弱知识点
REASON_NEW = "new"          # 没做过
REASON_EASIER = "easier"    # 冷启动时的退路


def _norm(s: str) -> str:
    """知识点比对前的归一化：去空白、去全半角括号差异。

    **不做同义词归并** —— 那需要一张词表，属于另一个量级的工作。
    这里只处理"看起来一样但字符串不等"的情况。
    """
    return (s or "").strip().replace("（", "(").replace("）", ")") \
                    .replace(" ", "").replace("　", "")


def score_problem(problem_kps: Iterable[str],
                  weak: Iterable[str]) -> tuple[int, list[str]]:
    """返回 `(重合数, 重合的知识点)`。"""
    pk = {_norm(x): (x or "").strip() for x in (problem_kps or [])}
    wk = {_norm(x) for x in (weak or []) if (x or "").strip()}
    hit = [pk[k] for k in pk if k in wk]
    return len(hit), sorted(hit)


def recommend_problems(
    problems: list[dict],
    weak_concepts: Iterable[str],
    *,
    exclude_problem_id: str = "",
    done_problem_ids: Optional[set[str]] = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """推荐下一题。

    `problems` 每项需要 `problem_id` / `title` / `knowledge_points` /
    `difficulty`（缺 `difficulty` 当 1 处理，**不报错** —— 题库是数据，
    不该因为一条缺字段就让整个推荐挂掉）。

    返回 `[{"problem_id", "title", "score", "matched", "reason", "difficulty"}]`。
    """
    weak = [w for w in (weak_concepts or []) if (w or "").strip()]
    done = set(done_problem_ids or ())
    if exclude_problem_id:
        done.add(exclude_problem_id)      # 刚做完这道，不再推它

    rows = []
    for p in problems or []:
        pid = str(p.get("problem_id") or p.get("id") or "")
        if not pid or pid in done:
            continue
        kps = p.get("knowledge_points") or []
        sc, matched = score_problem(kps, weak)
        try:
            diff = int(p.get("difficulty") or 1)
        except (TypeError, ValueError):
            diff = 1
        rows.append({
            "problem_id": pid,
            "title": str(p.get("title") or ""),
            "score": sc,
            "matched": matched,
            "difficulty": diff,
            "reason": REASON_WEAK if sc else (REASON_NEW if not weak
                                              else REASON_EASIER),
        })

    # 有命中知识点的排前面；同分时**没做过的优先 → 难度低的优先 → id 稳定**
    rows.sort(key=lambda r: (-r["score"], r["difficulty"], r["problem_id"]))
    out = rows[:max(0, limit)]
    # 冷启动时把所有 reason 归成"没做过"，避免 UI 上说"命中薄弱知识点"却没命中
    if not weak:
        for r in out:
            r["reason"] = REASON_NEW
    return out


def explain(row: dict) -> str:
    """给 UI 用的一句话理由。**空话不如不说** —— 命中就列出命中的知识点。"""
    if row.get("reason") == REASON_WEAK and row.get("matched"):
        return "命中薄弱知识点：" + "、".join(row["matched"][:4])
    if row.get("reason") == REASON_NEW:
        return "还没做过"
    return "同章节的其他练习"
