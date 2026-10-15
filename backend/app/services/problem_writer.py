"""题库写入（带**覆盖护栏**）。

## 为什么需要
`problem_repository` 是只读的，按 `problem_id` 找目录。
直接往 `data/problems/{id}/` 写文件的后果是：

**同 id 会静默覆盖。** 再录一遍 demo题2（比如改了图），旧的那份**无声消失**。

而这个项目里已经反复吃过"静默发生的事没人知道"的亏：
`max_frames` 静默截断、`latest-full.json` 被单套件覆盖、
`indexable` 被 `bool(None)` 全标错、`results.json` 写着提示值当模型用。

> **一次说错话的覆盖，比一次报错的拒绝难查得多。**

## 用法
```python
write_problem(problem, graph)              # 新增；已存在则抛 ProblemExistsError
write_problem(problem, graph, force=True)  # 明确要覆盖
```
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.models.problem import Problem
from app.models.problem_graph import ProblemGraph
from app.services.knowledge_vocabulary import assert_in_vocabulary
from app.services.problem_repository import get_problem_repository

logger = logging.getLogger(__name__)

PROBLEM_FILE = "problem.json"
GRAPH_FILE = "graph.json"


class ProblemExistsError(RuntimeError):
    """`problem_id` 已存在，且没有明确要求覆盖。"""


def write_problem(problem: Problem,
                  graph: ProblemGraph | None = None,
                  *,
                  force: bool = False,
                  allow_new_concepts: bool = False,
                  root: Path | str | None = None) -> str:
    """把题目（可选带图）写入题库。

    返回 `"added"` 或 `"overwritten"`。

    **已存在时默认拒绝。** 要覆盖必须显式 `force=True`——
    覆盖是**破坏性**的，不该是默认行为。

    ## 校验
    · `problem_id` 不能为空
    · `graph.problem_id` 必须与 `problem.problem_id` 一致
      （读取端只会**警告**并"以目录名为准"，那意味着图会被挂到错题上）
    """
    pid = (problem.problem_id or "").strip()
    if not pid:
        raise ValueError("problem_id 不能为空")
    from app.services.problem_import import safe_id
    safe_id(problem.problem_id)

    if graph is not None and graph.problem_id and graph.problem_id != pid:
        raise ValueError(
            f"graph.problem_id={graph.problem_id!r} 与 problem.problem_id={pid!r} "
            f"不一致——写进去会把图挂到错题上"
        )

    # ---- 受控词表校验 ----
    # 题目级 + 节点级的知识点**都要过** —— 推荐用的正是这两处的并集。
    #
    # **为什么在写入口拦**：分叉不是一次意外，是"每题各自写"的必然结果
    # （现有 72 条里 6 组同义就是这么来的）。拦在入口，加新题时才不会静默污染。
    _kps = list(problem.knowledge_points or [])
    for _n in getattr(graph, "nodes", None) or []:
        _kps += list(getattr(_n, "knowledge_points", None) or [])
    assert_in_vocabulary(_kps, allow_new=allow_new_concepts)

    repo = get_problem_repository()
    base = Path(root) if root else repo.root
    target = base / pid
    existed = (target / PROBLEM_FILE).is_file()

    if existed and not force:
        raise ProblemExistsError(
            f"{pid} 已存在（{target}）。覆盖是破坏性的，"
            f"要覆盖请显式传 force=True。"
        )

    target.mkdir(parents=True, exist_ok=True)
    if graph is not None:
        graph.problem_id = pid          # 以 problem 的 id 为准，避免两者漂移
    (target / PROBLEM_FILE).write_text(
        json.dumps(problem.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    if graph is not None:
        (target / GRAPH_FILE).write_text(
            json.dumps(graph.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8")

    action = "overwritten" if existed else "added"
    logger.info("题库写入 %s：%s", action, pid)
    return action


def write_many(items: list[tuple[Problem, ProblemGraph | None]], *,
               force: bool = False,
               root: Path | str | None = None) -> list[tuple[str, str]]:
    """批量写入，**逐条报结果**。

    返回 `[(problem_id, "added"|"overwritten"), ...]`。

    **先全部预检再落盘吗？不。** 逐条写并逐条报——
    批量任务里"写了一半失败"比"一条都没写"更需要看得见。
    """
    out: list[tuple[str, str]] = []
    for problem, graph in items:
        out.append((problem.problem_id,
                    write_problem(problem, graph, force=force, root=root)))
    return out
