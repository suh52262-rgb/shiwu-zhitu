"""题目推荐接口。

**放在独立文件里**，不动既有的 `students.py` / `problems.py` ——
推荐是附加能力，不该去改那些已经被测试保护着的路由。

前端用法（学生做完一道题后）：
    GET /students/{student_id}/recommendations?exclude=demo_03&limit=3
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.db import persistence
from app.services.problem_repository import get_problem_repository
from app.services.recommender import DEFAULT_LIMIT, explain, recommend_problems

router = APIRouter(tags=["recommend"])


@router.get("/students/{student_id}/recommendations")
def student_recommendations(
    student_id: str,
    exclude: str = Query("", description="刚做完的题号，不再推荐它"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=10),
) -> dict:
    """按**作答情况 + 学生画像**推荐下一题。

    ## 依据
    · **学生画像**：`student_profiles.weak_concepts`（跨题累积）
    · **本次作答**：该学生所有 `attempts.weak_concepts`（含刚做完那次）

    两者并起来作为"薄弱知识点"，再按**与题库各题知识点的重合度**排序。

    ## 冷启动
    学生还没暴露薄弱点时**不会返回空** —— 退化为"没做过且最简单的几道"。
    宁可推得保守，也不要给一片空白。
    """
    repo = get_problem_repository()

    # ---- 薄弱知识点 + 做过的题：`get_student_profile` 一次给全 ----
    # 它的 docstring 就是"学生画像 + 全部作答历史"，不用另查 Attempt 表。
    weak: list[str] = []
    done: set[str] = set()
    try:
        prof = persistence.get_student_profile(student_id) or {}
        weak += list(prof.get("observed_error_concepts") or [])
        for a in (prof.get("attempts") or []):
            if a.get("problem_id"):
                done.add(str(a["problem_id"]))
    except Exception:  # noqa: BLE001 DB 没起也不该让推荐整个挂掉
        pass

    problems = []
    for p in repo.list_problems():
        problems.append({
            "problem_id": p.problem_id,
            "title": p.title or p.problem_id,
            "knowledge_points": list(p.knowledge_points or []),
            "difficulty": p.difficulty or 1,
        })

    rows = recommend_problems(problems, weak, exclude_problem_id=exclude,
                              done_problem_ids=done, limit=limit)
    for r in rows:
        r["why"] = explain(r)
    return {"student_id": student_id, "weak_concepts": sorted(set(weak)),
            "recommendations": rows,
            "cold_start": not weak, "basis": "observed_writing_errors",
            "semantic_validated": False}
