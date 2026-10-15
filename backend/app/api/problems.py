"""题目管理接口（P16 接入 ProblemRepository）。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /problems | 题库列表（题目摘要） |
| GET | /problems/demo | 默认演示题（保留的兼容入口） |
| GET | /problems/{id} | 单题详情（含插图可访问 URL） |
| GET | /problems/{id}/graph | Problem Graph；404 = 该题 graphless |

路由声明顺序有讲究：`/demo` 必须在 `/{problem_id}` 之前，否则会被当成 problem_id。
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.problem import Problem
from app.services.problem_repository import ProblemDataError, get_problem_repository

router = APIRouter(prefix="/problems", tags=["problems"])

# 前端/演示默认题目
DEMO_PROBLEM_ID = "phy_001"
# 插图静态挂载前缀，见 app/main.py
ASSET_PREFIX = "/problem-assets"
assets_router = APIRouter()

@assets_router.get(ASSET_PREFIX + "/{problem_id}/{relative:path}")
def get_figure_asset(problem_id: str, relative: str):
    repo = get_problem_repository()
    path = repo.figure_path(problem_id, relative)
    if path is None:
        raise HTTPException(404, detail="题目配图不存在")
    problem = repo.get_problem(problem_id)
    if problem is None or relative not in problem.figures:
        raise HTTPException(404, detail="题目配图未登记")
    return FileResponse(path)


def _summary(problem: Problem, has_graph: bool) -> dict:
    return {
        "problem_id": problem.problem_id,
        "title": problem.title,
        "knowledge_points": problem.knowledge_points,
        "difficulty": problem.difficulty,
        "has_graph": has_graph,          # 前端据此显示 graph / graphless
        "figure_count": len(problem.figures),
        "question_family_id": problem.metadata.get("question_family_id", problem.problem_id),
        "question_revision": problem.metadata.get("question_revision", None),
        "parent_problem_id": problem.metadata.get("parent_problem_id", ""),
    }


def _detail(problem: Problem, has_graph: bool) -> dict:
    # exclude：Problem.answer 是参考答案，模型里明确注明"不下发给前端"
    data = problem.model_dump(exclude={"answer"})
    data["has_graph"] = has_graph
    # 把 figures 里的相对路径转成可直接访问的 URL（成员 C 的前端不必自己拼）
    data["figure_urls"] = [
        f"{ASSET_PREFIX}/{problem.problem_id}/{figure.lstrip('/')}"
        for figure in problem.figures
    ]
    return data


@router.get("")
def list_problems() -> dict:
    """题库列表。P16 起由 ProblemRepository 动态提供，不再写死。"""
    repo = get_problem_repository()
    items = []
    for problem in repo.list_problems():
        release = repo.graph_release(problem.problem_id)
        item = _summary(problem, release["graph"] is not None)
        item.update({key: value for key, value in release.items() if key != "graph"})
        items.append(item)
    return {"items": items, "total": len(items)}


@router.get("/demo")
def get_demo_problem() -> dict:
    """默认演示题。保留原有入口，内容改为从题库读取。"""
    try:
        problem = get_problem_repository().get_problem(DEMO_PROBLEM_ID)
    except ProblemDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if problem is None:
        raise HTTPException(status_code=404, detail=f"默认演示题 {DEMO_PROBLEM_ID} 不在题库中")
    return _detail(problem, get_problem_repository().has_graph(DEMO_PROBLEM_ID))


@router.get("/{problem_id}/graph")
def get_problem_graph(problem_id: str) -> dict:
    """获取题目的 Problem-Solving Graph。

    404 表示该题没有 Graph —— Agent 将以 graphless 模式运行（任务书第 11 节）。
    """
    try:
        release = get_problem_repository().graph_release(problem_id)
        graph = release["graph"]
    except ProblemDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail=f"题目 {problem_id} 没有 Problem Graph，将以 graphless 模式运行",
        )
    data = graph.model_dump()
    data.update({key: value for key, value in release.items() if key != "graph"})
    return data


@router.get("/{problem_id}")
def get_problem(problem_id: str) -> dict:
    """单题详情。"""
    repo = get_problem_repository()
    try:
        problem = repo.get_problem(problem_id)
    except ProblemDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if problem is None:
        raise HTTPException(status_code=404, detail=f"题目 {problem_id} 不存在")
    release = repo.graph_release(problem_id)
    data = _detail(problem, release["graph"] is not None)
    data.update({key: value for key, value in release.items() if key != "graph"})
    return data
