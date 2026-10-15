"""题库写入的覆盖护栏。

## 为什么必须有
`problem_repository` 按 `problem_id` 找目录 —— **同 id 直接覆盖，无声无息**。
再录一遍 demo题2（比如改了图），旧的那份就没了。

这个项目已经反复吃过"静默发生的事没人知道"的亏：
`max_frames` 静默截断、`latest-full.json` 被单套件覆盖、
`indexable` 被 `bool(None)` 全标错、`results.json` 写着提示值当模型用。

> **一次说错话的覆盖，比一次报错的拒绝难查得多。**
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from app.models.problem import Problem  # noqa: E402
from app.models.problem_graph import GraphNode, ProblemGraph  # noqa: E402
from app.services.problem_writer import (  # noqa: E402
    ProblemExistsError,
    write_many,
    write_problem,
)


def _prob(pid="t_001", title="测试题"):
    return Problem(problem_id=pid, title=title, statement="题干")


def _graph(pid="t_001"):
    return ProblemGraph(
        problem_id=pid,
        nodes=[GraphNode(id="S1", name="第一步", type="modeling",
                         description="描述")],
        edges=[],
        valid_paths=[["S1"]],
    )


# ---------------------------------------------------------------- 护栏

def test_writes_a_new_problem(tmp_path):
    assert write_problem(_prob(), _graph(), root=tmp_path) == "added"
    assert (tmp_path / "t_001" / "problem.json").is_file()
    assert (tmp_path / "t_001" / "graph.json").is_file()


def test_refuses_to_overwrite_by_default(tmp_path):
    """**核心护栏**：已存在就拒绝，不静默覆盖。"""
    write_problem(_prob(), _graph(), root=tmp_path)
    with pytest.raises(ProblemExistsError) as exc:
        write_problem(_prob(title="改了标题"), _graph(), root=tmp_path)
    assert "force" in str(exc.value), "报错信息没说清怎么覆盖"


def test_refusal_leaves_the_original_intact(tmp_path):
    """**拒绝之后原文件必须完好** —— 这是护栏的意义所在。"""
    write_problem(_prob(title="原版"), _graph(), root=tmp_path)
    before = (tmp_path / "t_001" / "problem.json").read_text(encoding="utf-8")
    with pytest.raises(ProblemExistsError):
        write_problem(_prob(title="新版"), _graph(), root=tmp_path)
    after = (tmp_path / "t_001" / "problem.json").read_text(encoding="utf-8")
    assert before == after, "被拒绝了却还是改了文件"
    assert "原版" in json.loads(after)["title"]


def test_force_overwrites_and_says_so(tmp_path):
    write_problem(_prob(title="原版"), _graph(), root=tmp_path)
    assert write_problem(_prob(title="新版"), _graph(),
                         force=True, root=tmp_path) == "overwritten"
    d = json.loads((tmp_path / "t_001" / "problem.json").read_text(encoding="utf-8"))
    assert d["title"] == "新版"


# ---------------------------------------------------------------- 校验

def test_rejects_empty_id(tmp_path):
    with pytest.raises(ValueError):
        write_problem(_prob(pid="  "), None, root=tmp_path)


def test_rejects_mismatched_graph_id(tmp_path):
    """**图挂错题**是最危险的一种错误 —— 读取端只会警告并"以目录名为准"。"""
    with pytest.raises(ValueError) as exc:
        write_problem(_prob("t_001"), _graph("t_999"), root=tmp_path)
    assert "不一致" in str(exc.value)


def test_graph_id_is_aligned_before_writing(tmp_path):
    """图没写 problem_id 时，以 problem 的为准，避免两者漂移。"""
    g = _graph("")
    write_problem(_prob("t_001"), g, root=tmp_path)
    d = json.loads((tmp_path / "t_001" / "graph.json").read_text(encoding="utf-8"))
    assert d["problem_id"] == "t_001"


def test_problem_without_graph_is_allowed(tmp_path):
    """graphless 题（如 phy_001）应当能录。"""
    assert write_problem(_prob("t_nog"), None, root=tmp_path) == "added"
    assert not (tmp_path / "t_nog" / "graph.json").exists()


# ---------------------------------------------------------------- 批量

def test_write_many_reports_each(tmp_path):
    """**逐条报结果** —— 批量任务里"写了一半"更要看得见。"""
    items = [(_prob("t_a"), _graph("t_a")), (_prob("t_b"), _graph("t_b"))]
    assert write_many(items, root=tmp_path) == [("t_a", "added"), ("t_b", "added")]
    # 再来一次，全部已存在 → 默认拒绝
    with pytest.raises(ProblemExistsError):
        write_many(items, root=tmp_path)
    # force 后逐条报 overwritten
    assert write_many(items, force=True, root=tmp_path) == [
        ("t_a", "overwritten"), ("t_b", "overwritten")]


def test_existing_demo_02_would_be_protected(tmp_path):
    """**对着真实数据自检**：`demo_02` 已存在，默认必须拒绝重录。

    这条是防止"以后重录 demo 例题时无声替换"。
    """
    from app.services.problem_repository import ProblemRepository
    repo_root = ProblemRepository().root
    assert (repo_root / "demo_02" / "problem.json").is_file()
    with pytest.raises(ProblemExistsError):
        write_problem(_prob("demo_02"), None, root=repo_root)
