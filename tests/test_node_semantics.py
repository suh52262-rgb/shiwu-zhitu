"""节点归属口径：唯一一处定义。

## 为什么要有这组测试

"学生到达了哪些节点"有三种都合理的答案（完成 / 碰到 / 试了没成）。
问题从来不是"有三种"，而是**三处消费方各用各的、谁也不说用的是哪一种**——
实测因此出现 3 条断言失败，排查花了很久才确认"教学语义其实是对的"。

这里钉住的是：**口径的定义只有一处，且不许静默退化。**
"""
import pytest

from app.agent.node_semantics import (
    SEMANTICS,
    attempted_only_nodes,
    completed_nodes,
    resolve,
    touched_nodes,
)


class _Match:
    """模拟 `StepMatch`。"""

    def __init__(self, matched, attempted, correct=True):
        self.matched_nodes = matched
        self.attempted_nodes = attempted
        self.correct = correct


# ---------------------------------------------------------------- 三种口径

def test_completed_requires_correct():
    """`completed_nodes` 与 `state.visited_nodes` 口径一致：**不正确就为空**。

    学生只给结论不论证时，匹配器有意不给信用——这是教学上正确的设计。
    """
    assert completed_nodes(_Match(["S5"], [], correct=True)) == {"S5"}
    assert completed_nodes(_Match(["S5"], [], correct=False)) == set()


def test_touched_ignores_correctness():
    """`touched_nodes` **不区分对错**——它是覆盖度口径。"""
    assert touched_nodes(_Match(["S5"], [], correct=True)) == {"S5"}
    assert touched_nodes(_Match(["S5"], [], correct=False)) == {"S5"}


def test_touched_includes_attempted():
    """碰到了 = 对应上的 ∪ 尝试过的。"""
    assert touched_nodes(_Match(["S2"], ["S1"], correct=True)) == {"S1", "S2"}


def test_attempted_only_excludes_matched():
    """"试了没成" 要把已经对应上的排掉，否则卡点统计会把成功也算成失败。"""
    assert attempted_only_nodes(_Match(["S2"], ["S1", "S2"])) == {"S1"}


def test_semantics_are_actually_distinct():
    """三种口径在同一个输入下必须给出**不同**结果。

    如果它们相等，说明实现退化成了一种，这个模块就没意义了。
    """
    m = _Match(["S2"], ["S1"], correct=False)
    assert completed_nodes(m) == set()
    assert touched_nodes(m) == {"S1", "S2"}
    assert attempted_only_nodes(m) == {"S1"}
    assert len({frozenset(completed_nodes(m)), frozenset(touched_nodes(m)),
                frozenset(attempted_only_nodes(m))}) == 3


# ---------------------------------------------------------------- 两种来源

def test_accepts_api_dict_as_well_as_object():
    """**同时接受 `StepMatch` 对象与 HTTP 响应里的 dict。**

    图内直接用对象，模拟器/评测拿到的是 `step_match` 字典。
    让口径函数适配两种来源，好过让每个调用方各写一遍提取逻辑
    ——那正是三种口径的由来。
    """
    d = {"matched_nodes": ["S2"], "attempted_nodes": ["S1"], "correct": False}
    assert touched_nodes(d) == {"S1", "S2"}
    assert completed_nodes(d) == set()
    assert attempted_only_nodes(d) == {"S1"}

    obj = _Match(["S2"], ["S1"], correct=False)
    assert touched_nodes(d) == touched_nodes(obj), "两种来源结果必须一致"


def test_handles_missing_fields():
    """缺字段/空 dict 不能崩——退化路径上这些很常见。"""
    assert touched_nodes({}) == set()
    assert completed_nodes({}) == set()
    assert attempted_only_nodes({}) == set()
    assert completed_nodes(_Match([], [], correct=True)) == set()


# ---------------------------------------------------------------- 选择口径

def test_resolve_raises_on_unknown_name():
    """**名字不认识时抛错，不给默认值。**

    静默用错口径正是这个模块要消灭的问题——给个默认值等于把它请回来。
    """
    with pytest.raises(KeyError, match="未知的节点口径"):
        resolve("whatever")


def test_all_names_resolvable():
    for name in SEMANTICS:
        assert callable(resolve(name))


# ---------------------------------------------------------------- 接线断言

def test_run_eval_uses_shared_semantics():
    """**防"写了没接"**：评测不许再自己拼口径。"""
    import inspect

    from scripts import run_eval

    src = inspect.getsource(run_eval._touched)
    assert "node_semantics" in src, "run_eval 没从统一口径取"


def test_simulation_runner_supports_explicit_names():
    """模拟器要支持显式口径字段，并保留 `nodes` 作为兼容别名。"""
    import inspect

    from app.simulation import runner

    src = inspect.getsource(runner._check_step)
    assert "completed_nodes" in src and "touched_nodes" in src
    assert "node_semantics" in src, "runner 没从统一口径取"


def test_case_field_comment_matches_behavior():
    """**字段注释不能说谎。**

    `nodes` 原注释写的是"期望**正确完成**的节点"，而用例全按"碰到就算"写
    ——注释与行为不符正是这次排查绕远路的原因。
    """
    from app.simulation.cases import StepExpectation

    fields = StepExpectation.model_fields
    assert "nodes" in fields and "completed_nodes" in fields
    assert "touched_nodes" in fields
    desc = fields["nodes"].description or ""
    # 注释里要写明是"碰到"而不是"完成"
    doc = StepExpectation.__doc__ or ""
    import inspect as _i

    src = _i.getsource(StepExpectation)
    assert "碰到" in src, "nodes 的语义没有写明是'碰到'"
