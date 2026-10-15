"""题目与 Problem Graph 存取的兼容入口（Phase 3 遗留，P16 已重构）。

P16 起统一走 `app.services.problem_repository.ProblemRepository`，
目录布局为 `data/problems/{problem_id}/{problem.json,graph.json,figures/}`。

本模块只保留原来的模块级函数名，供尚未迁移的调用方使用
（`scripts/run_eval.py`、若干测试仍直接 import 这些名字）。
**新代码请直接用 `get_problem_repository()`**，可以拿到 `list_problems` /
`save_graph` / `figure_path` 等本模块没有的能力。
"""
from app.models.problem import Problem
from app.models.problem_graph import ProblemGraph
from app.services.problem_repository import (
    FIGURES_DIR,
    GRAPH_FILE,
    PROBLEM_FILE,
    ProblemDataError,
    ProblemRepository,
    get_problem_repository,
)

__all__ = [
    "FIGURES_DIR",
    "GRAPH_FILE",
    "PROBLEM_FILE",
    "ProblemDataError",
    "ProblemRepository",
    "get_problem_repository",
    "load_problem",
    "load_problem_content",
    "load_problem_graph",
    "has_problem_graph",
]


def load_problem(problem_id: str) -> Problem | None:
    """加载题目，不存在返回 None。"""
    return get_problem_repository().get_problem(problem_id)


def load_problem_content(problem_id: str) -> str:
    """返回拼好标题的题干文本，供 LLM 节点使用；找不到题目返回空串。"""
    return get_problem_repository().get_problem_content(problem_id)


def load_problem_graph(problem_id: str) -> ProblemGraph | None:
    """加载题目的 Problem-Solving Graph；无 Graph 文件返回 None（graphless 模式）。"""
    return get_problem_repository().get_graph(problem_id)


def has_problem_graph(problem_id: str) -> bool:
    """题目是否配备 Problem Graph（P8：graph_mode / graphless_mode 的分流依据）。"""
    return get_problem_repository().has_graph(problem_id)
