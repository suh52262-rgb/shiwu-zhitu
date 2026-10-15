"""假分叉检查（`branch_checker`）的测试。

## 为什么需要它
P14 会把题面要求的**并列输出**建成两条可选路径。实测（教材例题试点）：
```
题面：求此二粒子间的静电力**和**万有引力各为多大？   ← 两个都要算
建图：S2 → {S3 库仑定律 | S4 万有引力定律} → S5      ← 建成了二选一
```
**结构完全合法，P15 抓不到**（和 `demo_05` 是同一个错误类）。

## 判据的脆弱处（已知）
它靠**措辞匹配** —— "题面要求的量"必须和节点名/描述用词接近。
实测漏过一例：我喂 `杆下端达到的高度`，节点叫 `杆的质心上升高度`。
**所以量最好由同一次 VL 抽取给出**（同一模型、同一页文本 → 措辞天然接近）。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.models.problem_graph import GraphNode, ProblemGraph  # noqa: E402
from app.services.branch_checker import (  # noqa: E402
    check_and_explain,
    find_false_branches,
)


def _g(edges, nodes):
    return ProblemGraph(
        problem_id="t",
        nodes=[GraphNode(id=i, name=n, type="calculation", description=d)
               for i, n, d in nodes],
        edges=edges, valid_paths=[])


# ================================================================ 核心：假分叉

def test_detects_both_required_built_as_either_or():
    """**这就是要抓的**：两个后继各自命中题面要求的**不同**量。"""
    g = _g([["S1", "S2"], ["S2", "S3"], ["S2", "S4"], ["S3", "S5"], ["S4", "S5"]],
           [("S1", "建模", "识别"), ("S2", "选择定律", "分支"),
            ("S3", "库仑定律", "算静电力"), ("S4", "万有引力定律", "算万有引力"),
            ("S5", "比较", "得出结果")])
    bad = find_false_branches(g, ["静电力", "万有引力"])
    assert len(bad) == 1
    assert bad[0]["node"] == "S2"
    assert set(bad[0]["hit"]["S3"]) == {"静电力"}
    assert set(bad[0]["hit"]["S4"]) == {"万有引力"}


def test_real_branch_not_flagged():
    """**真分叉不能误报**：两个后继是同一件事的两种做法。

    实测对照（`demo_02` 隔离法/整体法、`demo_03` 动量定理/牛顿定律）都干净。
    """
    g = _g([["S1", "S2"], ["S2", "S3a"], ["S2", "S3b"], ["S3a", "S4"], ["S3b", "S4"]],
           [("S1", "建模", "识别"), ("S2", "求加速度", "分支"),
            ("S3a", "隔离法", "求加速度"), ("S3b", "整体法", "求加速度"),
            ("S4", "得张力", "结果")])
    assert find_false_branches(g, ["加速度", "绳中张力"]) == []


def test_no_branch_no_issue():
    g = _g([["S1", "S2"], ["S2", "S3"]],
           [("S1", "a", ""), ("S2", "b", ""), ("S3", "c", "")])
    assert find_false_branches(g, ["x", "y"]) == []


# ================================================================ 前提条件

def test_needs_at_least_two_asked_quantities():
    """**题面只要一个量 → 不存在"并列输出"的问题**，不该报。"""
    g = _g([["S1", "S2"], ["S2", "S3"], ["S2", "S4"], ["S3", "S5"], ["S4", "S5"]],
           [("S1", "a", ""), ("S2", "b", ""), ("S3", "静电力", ""),
            ("S4", "万有引力", ""), ("S5", "c", "")])
    assert find_false_branches(g, ["静电力"]) == []
    assert find_false_branches(g, []) == []


def test_unknown_quantity_not_counted():
    """题面要求的量在节点里**一个都没命中** → 不该报（宁缺勿滥）。

    误报会浪费人工时间 —— 比漏报更糟，因为它训练人忽略检查结果。
    """
    g = _g([["S1", "S2"], ["S2", "S3"], ["S2", "S4"], ["S3", "S5"], ["S4", "S5"]],
           [("S1", "a", ""), ("S2", "b", ""), ("S3", "甲", ""), ("S4", "乙", ""),
            ("S5", "c", "")])
    assert find_false_branches(g, ["静电力", "万有引力"]) == []


def test_same_quantity_on_both_successors_not_flagged():
    """两个后继命中**同一个**量 → 是同一件事的两种做法 → 真分叉。"""
    g = _g([["S1", "S2"], ["S2", "S3"], ["S2", "S4"], ["S3", "S5"], ["S4", "S5"]],
           [("S1", "a", ""), ("S2", "b", ""),
            ("S3", "静电力用力法", ""), ("S4", "静电力用能量法", ""), ("S5", "c", "")])
    assert find_false_branches(g, ["静电力", "万有引力"]) == []


# ================================================================ 措辞归一化

def test_normalizes_punctuation_and_spaces():
    g = _g([["S1", "S2"], ["S2", "S3"], ["S2", "S4"], ["S3", "S5"], ["S4", "S5"]],
           [("S1", "a", ""), ("S2", "b", ""),
            ("S3", "计算静电力（库仑）", ""), ("S4", "计算万有引力", ""),
            ("S5", "c", "")])
    assert find_false_branches(g, ["静电力（大小）", "万有引力"]) != []


def test_explain_is_empty_when_clean():
    g = _g([["S1", "S2"]], [("S1", "a", ""), ("S2", "b", "")])
    assert check_and_explain(g, ["x", "y"]) == ""


def test_malformed_edges_do_not_crash():
    """**题库是数据，畸形内容不该让检查崩。**"""
    g = _g([["S1", "S2"], ["S2"], ["S2", "S2", "S2"]],
           [("S1", "a", ""), ("S2", "b", "")])
    find_false_branches(g, ["x", "y"])      # 不抛就算过
